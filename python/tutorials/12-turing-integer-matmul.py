"""
Turing INT8 & INT4 Matrix Multiplication
========================================
In this tutorial, you will write integer matrix multiplication kernels for
NVIDIA Turing (sm75) GPUs that run on the Tensor Cores. We cover two precisions:

* **INT8** (``s8 x s8 -> s32`` via ``m8n8k16``) — the easy case: ordinary
  ``tl.dot`` on ``int8`` inputs. On Turing this comfortably beats cuBLAS INT8,
  whose ``imma`` path is far less tuned than its FP16 one.

* **INT4** (``s4 x s4 -> s32`` via ``m8n8k32``) — the advanced case, and the
  first usable pure-INT4 matmul in Triton (upstream marks this path "Not
  implemented"). It needs operand packing and ``tl.reinterpret_as_int4`` because
  PyTorch has no ``int4`` tensor and 4-bit data cannot live in shared memory.

You will specifically learn about:

* Integer Tensor Core GEMM and shared-memory budgeting on Turing's 64KB/CTA.

* Packing 4-bit operands into 32-bit words on the host.

* ``tl.reinterpret_as_int4``: a zero-cost register relabel from packed ``int32``
  to ``int4`` at the dot-operand level, keeping 4-bit data out of shared memory.

"""

# %%
# Motivations
# -----------
#
# Low-precision integer GEMM is the workhorse of quantized inference. Turing
# Tensor Cores accelerate both INT8 (``m8n8k16``) and INT4 (``m8n8k32``), the
# latter at double the K — twice the compute throughput and half the memory
# traffic of INT8. cuBLAS ships an under-optimized INT8 path on Turing and *no*
# INT4 path at all, so a Triton kernel is the only way to reach ``m8n8k32``.

import numpy as np
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# %%
# Part 1 — INT8
# =============
#
# INT8 GEMM is just the FP16 matmul tutorial with an ``int8`` input dtype and an
# ``int32`` accumulator: ``tl.dot`` lowers to the ``m8n8k16`` Tensor Core
# instruction automatically. INT8 tiles are 1 byte/element, so a tile costs
# ``(BLOCK_M + BLOCK_N) * BLOCK_K * stages`` bytes of shared memory; we prune the
# autotune space to Turing's 64KB/CTA limit.


def get_turing_int8_configs():
    configs = []
    for bm, bn in [(128, 128), (128, 64), (64, 128), (64, 64)]:
        for bk in [32, 64]:
            for s in [1, 2, 3]:
                for w in [4, 8]:
                    smem = (bm + bn) * bk * 1 * s  # int8 = 1 byte/element
                    if smem <= 64 * 1024:
                        configs.append(
                            triton.Config(
                                {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn,
                                 'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': 8},
                                num_stages=s, num_warps=w))
    return configs


@triton.autotune(configs=get_turing_int8_configs(), key=['M', 'N', 'K'])
@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)  # -> m8n8k16
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def int8_matmul(a, b):
    """a: int8 [M, K], b: int8 [K, N] -> int32 [M, N]."""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    int8_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c


# %%
# Part 2 — INT4
# =============
#
# INT4 is harder because PyTorch has no ``int4`` tensor and shared memory cannot
# hold 4-bit elements (``local_alloc`` requires an 8-bit-multiple element width).
# We keep the data **packed as ``int32``** everywhere it must move (global loads,
# shared memory) and only relabel it to ``int4`` at the register level — right
# before the dot — with ``tl.reinterpret_as_int4``. Eight nibbles live in each
# ``int32`` (LSB-first, two's complement), so 4 ``int32`` registers == 32
# ``int4`` == 128 bits per thread: the relabel moves no data.
#
# Packing
# -------
# The contraction dimension (K) is the one we pack along, so that after the
# relabel K lands where ``tl.dot`` expects it:
#
# * ``A`` is packed along its **last** axis: ``int8[M, K] -> int32[M, K/8]``,
#   relabeled with ``axis=1`` back to ``int4[M, K]``.
# * ``B`` is packed along its **first** axis: ``int8[K, N] -> int32[K/8, N]``,
#   relabeled with ``axis=0`` back to ``int4[K, N]``.
#
# Packing K into a different physical axis for A vs B is what lets us skip a
# transpose (``tl.trans`` on ``int4`` would need shared memory, which int4
# cannot use).


def pack_int4_lastdim(x):
    """int8 nibbles in [-8, 7], shape [R, C] with C % 8 == 0 -> int32 [R, C//8]."""
    R, C = x.shape
    u = (x.astype(np.int32) & 0xF).reshape(R, C // 8, 8)
    shifts = (np.arange(8) * 4).astype(np.int32)
    return np.bitwise_or.reduce(u << shifts, axis=2).astype(np.int32)


def pack_int4_firstdim(x):
    """int8 [R, C] with R % 8 == 0 -> int32 [R//8, C] (pack along the row/K dim)."""
    return pack_int4_lastdim(x.T.copy()).T.copy()


# %%
# Packed int32 uses half the bytes of int8 for the same logical K, so a tile
# costs ``(BLOCK_M + BLOCK_N) * (BLOCK_K/8) * 4 * stages`` bytes — larger K tiles
# fit under the same 64KB budget than INT8 allows.


def get_turing_int4_configs():
    configs = []
    for bm, bn in [(128, 128), (128, 64), (64, 128), (64, 64)]:
        for bk in [64, 128, 256]:
            for s in [1, 2, 3]:
                for w in [4, 8]:
                    smem = (bm + bn) * (bk // 8) * 4 * s  # packed int32
                    if smem <= 64 * 1024:
                        configs.append(
                            triton.Config(
                                {'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn,
                                 'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': 8},
                                num_stages=s, num_warps=w))
    return configs


@triton.autotune(configs=get_turing_int4_configs(), key=['M', 'N', 'K'])
@triton.jit
def int4_matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    # a is packed [M, K//8] (K last -> reinterpret axis=1 -> [M, K]).
    # b is packed [K//8, N] (K first -> reinterpret axis=0 -> [K, N]).
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    KP: tl.constexpr = BLOCK_SIZE_K // 8       # packed-K tile (int32 units)
    KK = K // 8                                # total packed-K length
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_kp = tl.arange(0, KP)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_kp[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_kp[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(KK, KP)):
        # A masked-out packed int32 of 0 unpacks to 8 zero nibbles -> no-op MAC.
        a = tl.load(a_ptrs, mask=offs_kp[None, :] < KK - k * KP, other=0)
        b = tl.load(b_ptrs, mask=offs_kp[:, None] < KK - k * KP, other=0)
        a4 = tl.reinterpret_as_int4(a, axis=1)   # i4 [BLOCK_M, BLOCK_K]
        b4 = tl.reinterpret_as_int4(b, axis=0)   # i4 [BLOCK_K, BLOCK_N]
        acc += tl.dot(a4, b4, out_dtype=tl.int32)  # -> m8n8k32
        a_ptrs += KP * stride_ak
        b_ptrs += KP * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def int4_matmul(a_packed, b_packed, M, N, K):
    """a_packed: int32 [M, K//8], b_packed: int32 [K//8, N] -> int32 [M, N]."""
    c = torch.empty((M, N), device=a_packed.device, dtype=torch.int32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    int4_matmul_kernel[grid](
        a_packed, b_packed, c, M, N, K,
        a_packed.stride(0), a_packed.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# %%
# Unit Test
# ---------
#
# Both kernels are bit-exact against an ``int32`` reference matmul (small
# integers never overflow int32).


def unit_test():
    np.random.seed(0)
    M = N = K = 512
    a_i = np.random.randint(-8, 8, (M, K)).astype(np.int8)
    b_i = np.random.randint(-8, 8, (K, N)).astype(np.int8)
    ref = a_i.astype(np.int32) @ b_i.astype(np.int32)

    a8 = torch.from_numpy(a_i).to(DEVICE)
    b8 = torch.from_numpy(b_i).to(DEVICE)
    out8 = int8_matmul(a8, b8).cpu().numpy()
    assert np.array_equal(out8, ref), "INT8 differs from the int32 reference"
    print("✅ Triton INT8 matches the int32 reference (EXACT)")

    a_p = torch.from_numpy(pack_int4_lastdim(a_i)).to(DEVICE)
    b_p = torch.from_numpy(pack_int4_firstdim(b_i)).to(DEVICE)
    out4 = int4_matmul(a_p, b_p, M, N, K).cpu().numpy()
    assert np.array_equal(out4, ref), "INT4 differs from the int32 reference"
    print("✅ Triton INT4 matches the int32 reference (EXACT)")


# %%
# Benchmark
# ---------
#
# We compare Triton INT4 against Triton INT8 (the same Tensor Core path, half the
# throughput in theory) and cuBLAS INT8 (``torch._int_mm``) as an external floor
# — cuBLAS has no INT4 path on Turing. Note: without a locked clock, the
# large-size tail is affected by thermal throttling.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["cublas-int8", "triton-int8", "triton-int4"],
        line_names=["cuBLAS INT8", "Triton INT8", "Triton INT4"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TOPS",
        plot_name="turing-integer-matmul-performance",
        args={},
    ))
def benchmark(M, N, K, provider):
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas-int8":
        a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=DEVICE)
        b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch._int_mm(a, b), quantiles=quantiles)
    elif provider == "triton-int8":
        a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=DEVICE)
        b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: int8_matmul(a, b), quantiles=quantiles)
    else:  # triton-int4
        a_i = np.random.randint(-8, 8, (M, K)).astype(np.int8)
        b_i = np.random.randint(-8, 8, (K, N)).astype(np.int8)
        a_p = torch.from_numpy(pack_int4_lastdim(a_i)).to(DEVICE)
        b_p = torch.from_numpy(pack_int4_firstdim(b_i)).to(DEVICE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: int4_matmul(a_p, b_p, M, N, K), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)  # logical MACs -> TOPS
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    unit_test()
    benchmark.run(show_plots=True, print_data=True)
