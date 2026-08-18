# Correctness guards for the sm75 Tensor Core path.
#
# Both cases below were live defects until 2026-08-04, and neither announced
# itself: one produced wrong numbers with no diagnostic, the other reached
# codegen as an illegal op instead of falling back. They exist because
# upstream deprecated sm75 MMA, so the shared constraint checks
# (supportMMA(), the MMAv2 lowering) only encode what Ampere and later can do
# -- anything Turing lacks has to be re-stated here.
#
# Comparisons are exact where the arithmetic allows it: inputs are small
# integers, fp16 dots accumulate in fp32 against an fp64 reference, int8 dots
# accumulate in int32 against an int64 reference. A clobbered accumulator then
# shows up as an exact mismatch rather than a tolerance blip -- which matters,
# because the batched defect was hidden for a while behind "one element out of
# 8192 is slightly over atol".

import pytest
import torch

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def batched_dot_kernel(a_ptr, b_ptr, c_ptr,  #
                       stride_ab, stride_am, stride_ak,  #
                       stride_bb, stride_bk, stride_bn,  #
                       stride_cb, stride_cm, stride_cn,  #
                       IS_INT8: tl.constexpr,  #
                       BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_b = tl.arange(0, BLOCK_B)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_b[:, None, None] * stride_ab +
                      offs_m[None, :, None] * stride_am +
                      offs_k[None, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_b[:, None, None] * stride_bb +
                      offs_k[None, :, None] * stride_bk +
                      offs_n[None, None, :] * stride_bn)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    acc = tl.dot(a, b, out_dtype=tl.int32 if IS_INT8 else tl.float32)
    c_ptrs = c_ptr + (offs_b[:, None, None] * stride_cb +
                      offs_m[None, :, None] * stride_cm +
                      offs_n[None, None, :] * stride_cn)
    tl.store(c_ptrs, acc)


@pytest.mark.parametrize("B", [2, 4, 8])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_batched_dot_more_batches_than_warps(B, num_warps, dtype, device):
    """Every batch element must land in its own accumulator registers.

    warpsPerTileV2() spreads warps along the batch axis for a rank-3 dot, so a
    warp covers more than one batch element exactly when B > num_warps. The
    Turing MMA helpers used to drop the batch stride when indexing the
    accumulator, sending every element into batch 0's registers; the only
    guard was an assert() that release builds compile away. With num_warps >= B
    the bug is invisible, which is why this parametrization crosses the
    boundary in both directions.
    """
    if not is_cuda():
        pytest.skip("sm75 Tensor Core path is CUDA-only")

    M = N = K = 32
    torch.manual_seed(0)
    if dtype == "int8":
        a = torch.randint(-4, 5, (B, M, K), dtype=torch.int8, device=device)
        b = torch.randint(-4, 5, (B, K, N), dtype=torch.int8, device=device)
        c = torch.empty((B, M, N), dtype=torch.int32, device=device)
        # bmm has no int64 CUDA kernel; the reference is exact either way.
        ref = torch.bmm(a.cpu().to(torch.int64), b.cpu().to(torch.int64)).to(device)
    else:
        a = torch.randint(-4, 5, (B, M, K), device=device).to(torch.float16)
        b = torch.randint(-4, 5, (B, K, N), device=device).to(torch.float16)
        c = torch.empty((B, M, N), dtype=torch.float32, device=device)
        ref = torch.bmm(a.to(torch.float64), b.to(torch.float64))

    batched_dot_kernel[(1, )](
        a, b, c,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        IS_INT8=(dtype == "int8"),
        BLOCK_B=B, BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,
        num_warps=num_warps,
    )
    torch.testing.assert_close(c.to(ref.dtype), ref, atol=0, rtol=0)


@triton.jit
def f32_dot_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs_m[:, None] * BLOCK + offs_n[None, :])
    b = tl.load(b_ptr + offs_m[:, None] * BLOCK + offs_n[None, :])
    tl.store(c_ptr + offs_m[:, None] * BLOCK + offs_n[None, :], tl.dot(a, b))


def test_f32_dot_does_not_use_mma(device):
    """f32 dots must fall back to FMA on Turing.

    supportMMA() only rejects f32 operands when the dot asks for something
    other than TF32 -- and tf32 is tl.dot's default -- so on Ampere these
    become TF32 MMA. Turing has no TF32 tensor core, and its instruction table
    has no FP32_TF32_TF32_FP32 entry, so admitting the layout here produced an
    illegal tt.dot at codegen rather than an FMA fallback.
    """
    if not is_cuda():
        pytest.skip("sm75 Tensor Core path is CUDA-only")
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("Ampere and later legitimately use TF32 MMA here")

    BLOCK = 32
    torch.manual_seed(0)
    a = torch.randint(-4, 5, (BLOCK, BLOCK), device=device).to(torch.float32)
    b = torch.randint(-4, 5, (BLOCK, BLOCK), device=device).to(torch.float32)
    c = torch.empty((BLOCK, BLOCK), dtype=torch.float32, device=device)

    k = f32_dot_kernel[(1, )](a, b, c, BLOCK=BLOCK)
    torch.testing.assert_close(c.to(torch.float64),
                               torch.mm(a.to(torch.float64), b.to(torch.float64)),
                               atol=0, rtol=0)
    assert "mma.sync" not in k.asm["ptx"], \
        "Turing has no TF32 tensor core; an f32 dot must lower to FMA"
