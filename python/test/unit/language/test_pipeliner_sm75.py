# Stress tests for the sm75 synchronous-copy software pipeline.
#
# These tests run on any CUDA arch (on Ampere+ they exercise the async copy
# path), but they were written to pin down Turing-specific risks:
# - masked-off pipeline stages still execute local_store unconditionally
#   (predicateOp whitelists it); the garbage they write must only ever land
#   in slots whose consumers are also masked off. Tiny and degenerate trip
#   counts are where this breaks.
# - multibuffer slot rotation (num_stages > 2) must never map two live
#   stages to the same slot. Deterministic integer data makes a clobbered
#   tile show up as an exact mismatch instead of a tolerance blip.
# - the sync path keeps the original tt.load alive, so mask + non-zero
#   `other` semantics must survive pipelining without the async path's
#   select special-case.
#
# All comparisons are exact: inputs are small integers, and the kernel
# accumulates losslessly. For fp16 inputs it accumulates in fp32 and the
# reference in fp64; for int8 inputs it accumulates in int32 and the
# reference in int64. Both representations are exact for these sums, so an
# int8 dot (m8n8k16, s32.s8.s8.s32) is pinned down the same way fp16 is.

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_padded_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                         stride_am, stride_ak, stride_bk, stride_bn,  #
                         stride_cm, stride_cn,  #
                         A_OTHER: tl.constexpr, B_OTHER: tl.constexpr,  #
                         IS_INT8: tl.constexpr,  #
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc_dtype = tl.int32 if IS_INT8 else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=rk[None, :] < k_rem, other=A_OTHER)
        b = tl.load(b_ptrs, mask=rk[:, None] < k_rem, other=B_OTHER)
        acc += tl.dot(a, b, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def run_and_check(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, device,
                  a_other=0.0, b_other=0.0, dtype="fp16"):
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0, "only K may be ragged"
    assert dtype in ("fp16", "int8")
    is_int8 = dtype == "int8"
    torch.manual_seed(0)
    if is_int8:
        a = torch.randint(-4, 5, (M, K), device=device, dtype=torch.int8)
        b = torch.randint(-4, 5, (K, N), device=device, dtype=torch.int8)
        c = torch.empty((M, N), device=device, dtype=torch.int32)
        ref_dtype = torch.int64
    else:
        a = torch.randint(-4, 5, (M, K), device=device).half()
        b = torch.randint(-4, 5, (K, N), device=device).half()
        c = torch.empty((M, N), device=device, dtype=torch.float32)
        ref_dtype = torch.float64
    grid = (M // BLOCK_M, N // BLOCK_N)
    matmul_padded_kernel[grid](
        a, b, c, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        A_OTHER=int(a_other) if is_int8 else a_other,  #
        B_OTHER=int(b_other) if is_int8 else b_other,  #
        IS_INT8=is_int8,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
        num_stages=num_stages)

    # The kernel pads the last partial K tile with A_OTHER/B_OTHER via the
    # load masks; replicate that padding exactly in the reference. int64
    # matmul isn't implemented on CUDA, so the int8 reference runs on CPU
    # (exact, and these matrices are tiny); fp16 stays on-device in fp64.
    K_ceil = triton.cdiv(K, BLOCK_K) * BLOCK_K if K > 0 else 0
    ref_device = "cpu" if is_int8 else device
    a_pad = torch.full((M, K_ceil), a_other, device=ref_device, dtype=ref_dtype)
    b_pad = torch.full((K_ceil, N), b_other, device=ref_device, dtype=ref_dtype)
    a_pad[:, :K] = a.to(ref_device).to(ref_dtype)
    b_pad[:K, :] = b.to(ref_device).to(ref_dtype)
    ref = a_pad @ b_pad

    mismatched = (c.to(ref_device).to(ref_dtype) != ref).sum().item()
    assert mismatched == 0, (
        f"{mismatched}/{c.numel()} elements wrong for K={K} "
        f"(trip count {triton.cdiv(K, BLOCK_K)}), num_stages={num_stages}, "
        f"dtype={dtype}")


# Trip counts around and below the pipeline depth: the prologue prefetches
# num_stages-1 tiles unconditionally (mask-predicated), so loops shorter than
# the pipeline exercise stores of masked-off stages and epilogue draining.
# K=33 adds a ragged final tile on top of a tiny trip count.
@pytest.mark.parametrize("dtype", ["fp16", "int8"])
@pytest.mark.parametrize("K", [0, 16, 32, 33, 64, 96, 160])
@pytest.mark.parametrize("num_stages", [2, 3, 4])
def test_edge_trip_counts(K, num_stages, dtype, device):
    run_and_check(64, 64, K, 64, 64, 32, num_stages, device, dtype=dtype)


# Multibuffer rotation with unique deterministic tiles: a single slot
# collision corrupts an entire BLOCK_K contribution and fails the exact
# compare. Shared memory: (64x32 + 32x64) fp16 = 8KB per stage (int8 is half
# that), so even num_stages=5 fits Turing's 64KB/CTA. Multiple CTAs via a
# 2x2 grid.
@pytest.mark.parametrize("dtype", ["fp16", "int8"])
@pytest.mark.parametrize("num_stages", [2, 3, 4, 5])
def test_multibuffer_slot_rotation(num_stages, dtype, device):
    run_and_check(128, 128, 640, 64, 64, 32, num_stages, device, dtype=dtype)


# Non-zero `other` on masked loads: the sync path keeps the original tt.load
# (mask and other included) as the data source, unlike the async path which
# needs a select special-case. The padding contribution 3*2*pad_len is
# replicated exactly by the padded reference.
@pytest.mark.parametrize("dtype", ["fp16", "int8"])
@pytest.mark.parametrize("K", [33, 80])
@pytest.mark.parametrize("num_stages", [2, 3])
def test_masked_load_nonzero_other(K, num_stages, dtype, device):
    run_and_check(64, 64, K, 64, 64, 32, num_stages, device,
                  a_other=3.0, b_other=2.0, dtype=dtype)
