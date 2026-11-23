import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def _nested_while_in_for_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    total = 0
    for base in range(0, N, BLOCK):
        cnt = 0
        row = tl.load(x_ptr + pid * N + base + offs)[None, :]
        acc = tl.broadcast_to(row, (16, BLOCK))
        while cnt < 4 and tl.sum(acc) > 0.0:
            cnt += 1
            acc = tl.broadcast_to(row - cnt, (16, BLOCK))
        total += cnt
    tl.store(out_ptr + pid, total)


@triton.jit
def _weighted_bincount_kernel(
    idx_ptr,
    val_ptr,
    out_ptr,
    stride_seg,
    stride_out,
    N,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_base = idx_ptr + pid * stride_seg
    val_base = val_ptr + pid * stride_seg
    bins = tl.arange(0, BINS)[:, None]
    acc = tl.zeros([BINS, BLOCK], dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    for base in range(0, N, BLOCK):
        idx = tl.load(idx_base + base + offs)
        val = tl.load(val_base + base + offs).to(tl.float32)
        hit = bins == (idx & (BINS - 1)).to(tl.int32)[None, :]
        acc += tl.where(hit, val[None, :], 0.0)
    res = tl.sum(acc, axis=1)
    tl.store(out_ptr + pid * stride_out + tl.arange(0, BINS), res)


@pytest.mark.parametrize("cc", [75, 80])
def test_nested_while_in_for_compile(cc):
    """Regression: nested scf.while with fewer results than beforeArguments."""
    sig = {"x_ptr": "*fp16", "out_ptr": "*i32", "N": "i32"}
    src = triton.compiler.ASTSource(
        _nested_while_in_for_kernel,
        signature=sig,
        constexprs={"BLOCK": 1024},
    )
    compiled = triton.compile(
        src, target=GPUTarget("cuda", cc, 32), options={"num_warps": 4}
    )
    assert compiled.asm["cubin"]


def test_nested_while_in_for_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rows, n = 4, 2048
    # Row values: 2.5 -> while runs 3 iterations per 1024-tile (total=6);
    # 0.5 -> 1 iter/tile (total=2); -1.0 -> 0 iters (total=0); 10.0 -> capped at 4/tile (total=8).
    vals = torch.tensor([2.5, 0.5, -1.0, 10.0], dtype=torch.float16, device="cuda")
    x = vals[:, None].expand(rows, n).contiguous()
    out = torch.empty(rows, dtype=torch.int32, device="cuda")
    _nested_while_in_for_kernel[(rows,)](x, out, n, BLOCK=1024, num_warps=4)
    expected = torch.tensor([6, 2, 0, 8], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(out, expected)


def test_loop_component_sm75_target_gate():
    """SM75 activates loop-component layout selection; SM80 remains unchanged."""
    sig = {
        "idx_ptr": "*i16",
        "val_ptr": "*fp16",
        "out_ptr": "*fp32",
        "stride_seg": "i32",
        "stride_out": "i32",
        "N": "i32",
    }
    src = triton.compiler.ASTSource(
        _weighted_bincount_kernel,
        signature=sig,
        constexprs={"BINS": 16, "BLOCK": 1024},
        attrs={(i,): [["tt.divisibility", 16]] for i in range(len(sig))},
    )
    res75 = triton.compile(
        src, target=GPUTarget("cuda", 75, 32), options={"num_warps": 4}
    )
    res80 = triton.compile(
        src, target=GPUTarget("cuda", 80, 32), options={"num_warps": 4}
    )

    assert res75.metadata.shared == 256
    assert "sizePerThread = [1, 8]" in res75.asm["ttgir"]

    assert res80.metadata.shared == 2048
    assert "sizePerThread = [1, 8]" not in res80.asm["ttgir"]
