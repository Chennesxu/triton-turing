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
#
# int4 (m8n8k32, s32.s4.s4.s32) is pinned the same way, but its shared buffer
# holds int32 words of eight packed nibbles -- eight times narrower than the
# dot operand. That asymmetry is what broke the prefetch pass in 65b2501fc,
# and it only exists at >= 2 ring slots, so the int4 multibuffer test asserts
# the real slot count rather than trusting num_stages.

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


@triton.jit
def matmul_padded_int4_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  #
                              stride_am, stride_ak, stride_bk, stride_bn,  #
                              stride_cm, stride_cn,  #
                              A_OTHER: tl.constexpr, B_OTHER: tl.constexpr,  #
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr):
    # K counts nibbles; a is packed int32 [M, K//8] and b is packed int32
    # [K//8, N], so A packs along its last axis and B along its first. That is
    # what lets both relabels put K where tl.dot wants it without a transpose
    # (tl.trans on int4 would need shared memory, which int4 cannot use).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    KP: tl.constexpr = BLOCK_K // 8  # packed-K tile, in int32 words
    rkp = tl.arange(0, KP)
    KK = K // 8
    a_ptrs = a_ptr + rm[:, None] * stride_am + rkp[None, :] * stride_ak
    b_ptrs = b_ptr + rkp[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(KK, KP)):
        kp_rem = KK - k * KP
        a = tl.load(a_ptrs, mask=rkp[None, :] < kp_rem, other=A_OTHER)
        b = tl.load(b_ptrs, mask=rkp[:, None] < kp_rem, other=B_OTHER)
        acc += tl.dot(tl.reinterpret_as_int4(a, axis=1),
                      tl.reinterpret_as_int4(b, axis=0), out_dtype=tl.int32)
        a_ptrs += KP * stride_ak
        b_ptrs += KP * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def pack_int4(x, axis):
    """int8 nibbles in [-8, 7] -> int32, eight per word, LSB first.

    axis=1 packs the last dim ([R, C] -> [R, C // 8]), axis=0 the first.
    """
    if axis == 0:
        return pack_int4(x.T.contiguous(), 1).T.contiguous()
    R, C = x.shape
    nibbles = (x.to(torch.int64) & 0xF).reshape(R, C // 8, 8)
    shifts = torch.arange(8, device=x.device, dtype=torch.int64) * 4
    # Nibbles do not overlap, so summing the shifted lanes is a bitwise or.
    # int64 keeps the top nibble off the sign bit until the truncating cast.
    return (nibbles << shifts).sum(dim=2).to(torch.int32)


def splat_nibble(v):
    """The packed int32 whose eight nibbles all equal v -- the `other` a masked
    int4 load needs, since it reads whole words."""
    word = (int(v) & 0xF) * 0x11111111
    return word - (1 << 32) if word >= (1 << 31) else word


def run_and_check(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, device,
                  a_other=0.0, b_other=0.0, dtype="fp16"):
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0, "only K may be ragged"
    assert dtype in ("fp16", "int8", "int4")
    is_int8 = dtype == "int8"
    is_int4 = dtype == "int4"
    torch.manual_seed(0)
    if is_int4:
        assert K % 8 == 0 and BLOCK_K % 8 == 0, "int4 packs eight nibbles per word"
        # The full [-8, 7] nibble range, so that a sign-extension slip in the
        # relabel shows up as a mismatch rather than surviving on small values.
        a = torch.randint(-8, 8, (M, K), device=device, dtype=torch.int8)
        b = torch.randint(-8, 8, (K, N), device=device, dtype=torch.int8)
        c = torch.empty((M, N), device=device, dtype=torch.int32)
        ref_dtype = torch.int64
    elif is_int8:
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
    if is_int4:
        a_packed = pack_int4(a, axis=1)
        b_packed = pack_int4(b, axis=0)
        compiled = matmul_padded_int4_kernel[grid](
            a_packed, b_packed, c, M, N, K,  #
            a_packed.stride(0), a_packed.stride(1),  #
            b_packed.stride(0), b_packed.stride(1),  #
            c.stride(0), c.stride(1),  #
            A_OTHER=splat_nibble(a_other), B_OTHER=splat_nibble(b_other),  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            num_stages=num_stages)
    else:
        compiled = matmul_padded_kernel[grid](
            a, b, c, M, N, K,  #
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            A_OTHER=int(a_other) if is_int8 else a_other,  #
            B_OTHER=int(b_other) if is_int8 else b_other,  #
            IS_INT8=is_int8,  #
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,  #
            num_stages=num_stages)

    # Numerical correctness alone can't tell a Tensor Core dot from an FMA
    # fallback, so on Turing pin the integer paths to the actual imma
    # instruction. K=0 compiles the loop away (no dot), so skip that case.
    imma = {"int8": "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32",
            "int4": "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32"}
    if dtype in imma and K > 0 and torch.cuda.get_device_capability() == (7, 5):
        assert imma[dtype] in compiled.asm["ptx"], \
            f"{dtype} dot did not lower to the Tensor Core path"

    # The kernel pads the last partial K tile with A_OTHER/B_OTHER via the
    # load masks; replicate that padding exactly in the reference. int64
    # matmul isn't implemented on CUDA, so the int8 reference runs on CPU
    # (exact, and these matrices are tiny); fp16 stays on-device in fp64.
    K_ceil = triton.cdiv(K, BLOCK_K) * BLOCK_K if K > 0 else 0
    ref_device = "cpu" if (is_int8 or is_int4) else device
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
    return compiled


# Trip counts around and below the pipeline depth: the prologue prefetches
# num_stages-1 tiles unconditionally (mask-predicated), so loops shorter than
# the pipeline exercise stores of masked-off stages and epilogue draining.
# K=33 adds a ragged final tile on top of a tiny trip count. num_stages=1 is
# the single-buffer (no pipeline) baseline.
@pytest.mark.parametrize("dtype", ["fp16", "int8"])
@pytest.mark.parametrize("K", [0, 16, 32, 33, 64, 96, 160])
@pytest.mark.parametrize("num_stages", [1, 2, 3, 4])
def test_edge_trip_counts(K, num_stages, dtype, device):
    run_and_check(64, 64, K, 64, 64, 32, num_stages, device, dtype=dtype)


# Multibuffer rotation with unique deterministic tiles: a single slot
# collision corrupts an entire BLOCK_K contribution and fails the exact
# compare. Shared memory: (64x32 + 32x64) fp16 = 8KB per stage (int8 is half
# that), so even num_stages=5 fits Turing's 64KB/CTA. Multiple CTAs via a
# 2x2 grid.
@pytest.mark.parametrize("dtype", ["fp16", "int8"])
@pytest.mark.parametrize("num_stages", [1, 2, 3, 4, 5])
def test_multibuffer_slot_rotation(num_stages, dtype, device):
    run_and_check(128, 128, 640, 64, 64, 32, num_stages, device, dtype=dtype)


# Non-zero `other` on masked loads: the sync path keeps the original tt.load
# (mask and other included) as the data source, unlike the async path which
# needs a select special-case. The padding contribution 3*2*pad_len is
# replicated exactly by the padded reference.
@pytest.mark.parametrize("dtype", ["fp16", "int8"])
@pytest.mark.parametrize("K", [33, 80])
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_masked_load_nonzero_other(K, num_stages, dtype, device):
    run_and_check(64, 64, K, 64, 64, 32, num_stages, device,
                  a_other=3.0, b_other=2.0, dtype=dtype)


# The same three groups for int4. K is in nibbles and must be a multiple of 8,
# the pack unit, so a ragged tile ends on a partial count of int32 words rather
# than a partial word; BLOCK_K=64 is the smallest tile the int4 autotune list
# ships. Trip counts match the fp16/int8 sets above: 0, 1, 1, 2, 2, 3, 5.
@pytest.mark.parametrize("K", [0, 32, 64, 72, 128, 192, 320])
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_int4_edge_trip_counts(K, num_stages, device):
    run_and_check(64, 64, K, 64, 64, 64, num_stages, device, dtype="int4")


# Shared memory: (64 + 64) x 64/8 int32 = 4KB per stage, so every slot count
# here fits. The int4 ring is the one case where the buffer is narrower than
# its dot operand, which the prefetch pass has to refuse; that only happens at
# >= 2 slots, so assert the real slot count rather than trusting num_stages --
# the latency clamp rewrites it silently and `shared` does not show it.
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_int4_multibuffer_slot_rotation(num_stages, device):
    compiled = run_and_check(128, 128, 1280, 64, 64, 64, num_stages, device,
                             dtype="int4")
    if num_stages > 1 and torch.cuda.get_device_capability() == (7, 5):
        assert compiled.metadata.sm75_pipeline_slots == num_stages - 1, (
            f"num_stages={num_stages} got "
            f"{compiled.metadata.sm75_pipeline_slots} ring slots")


# Non-zero `other` on a masked int4 load: the load reads whole int32 words, so
# `other` is a packed word and every nibble in it lands in the dot.
@pytest.mark.parametrize("K", [72, 160])
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_int4_masked_load_nonzero_other(K, num_stages, device):
    run_and_check(64, 64, K, 64, 64, 64, num_stages, device,
                  a_other=3.0, b_other=2.0, dtype="int4")
