

# Triton-Turing

**Triton-Turing** is a community-maintained fork of [Triton](https://github.com/triton-lang/triton) focused on restoring high-performance Tensor Core support for NVIDIA Turing GPUs (SM75: RTX 2080 Ti, Titan RTX).

Upstream Triton supports Turing's MMA instructions, but critical optimizations were gated to SM80+ (Ampere and later). The biggest one is the software pipeline, which exclusively uses `cp.async`, an Ampere-only instruction. As a result, Turing performance degrades significantly compared to its tensor-core potential.

## Goals

1. **Software pipelining without `cp.async`**: a multi-stage `ld.global → st.shared → bar.sync` path to overlap memory loads with MMA on Turing (`num_stages` ≥ 2, not just double-buffering)
2. **Turing-specific autotune**: configs tuned for 64 KB/CTA shared memory and native instruction shapes (fp16: `m16n8k8`, int8: `m8n8k16`)
3. **int4 MMA support**: implement the `m8n8k32` instruction path for int4 precision (hardware-supported but not implemented in upstream Triton)

## Status

| Feature | Status |
|---|---|
| Software pipelining (multi-stage `ld.global + bar.sync`) — first ever for Turing | ✅ Done |
| Turing-specific autotune configs | ✅ Done |
| int8 GEMM (`m8n8k16`) | ✅ Done |
| int4 MMA (`m8n8k32`) — first usable pure-int4 matmul in Triton | ✅ Done |
| bf16 dot on the fp16 Tensor Core — opt-in, see below | ✅ Done |

## Performance

All numbers below are from a Titan RTX (sm75), each operator against the
strongest existing implementation in its domain. The clock is not locked, so
every comparison is made within a single process; the very largest GEMM sizes
are still throttle-prone.

### FlashAttention-2 forward — faster than a hand-written CUDA kernel

![FlashAttention-2 forward](.github/assets/benchmarks/fa2-forward.png)

The Triton FA2 forward kernel (tutorial `06-fused-attention.py` plus our sm75
pipeline) is the fastest at every size measured. It is ahead of a from-scratch
CUDA/CUTLASS FlashAttention for Turing by **+21–26%** at head dim 64 and
**+8–10%** at head dim 128, and ahead of PyTorch SDPA (xformers backend) by
**1.8–2.1×**. Attention benefits from the pipeline because the softmax
dependency chain leaves the Tensor Cores idle, and the pipeline uses that window
to prefetch K/V.

### FlashAttention-2 backward — a mixed result, and our one weakness

![FlashAttention-2 backward](.github/assets/benchmarks/fa2-backward.png)

At head dim 64 our kernel beats the CUDA/CUTLASS implementation by **+35–40%**,
from Turing-specific block sizes plus a codegen change that lets a transposed
dot operand read the shared buffer its untransposed sibling already filled,
instead of paying a scratch round trip every loop iteration.

At head dim 128 it **trails by 15–16%**, the only place we lose. Upstream's
d=128 backward blocks need ~82 KB of shared memory, well past Turing's hard
**64 KB/CTA** limit, so `BLOCK_N1` and `BLOCK_M2` are halved to 64 to fit. An
exhaustive sweep of the 216-configuration block/stage/warp space confirms that
fallback is the fastest option available, not a tuning oversight: every larger
tile that fits forces `num_stages=1`, and losing the pipeline costs more than
the tile gains.

### Integer GEMM — INT4 doubles INT8, and cuBLAS has no INT4 path

![Integer GEMM](.github/assets/benchmarks/integer-gemm.png)

INT4 (`m8n8k32`) runs **2.1–2.5× faster than INT8** over the mid-to-large
range and peaks at **258 TOPS**. It gets both 2× the Tensor Core throughput and
half the shared-memory traffic, since operands stay packed as `int32`. cuBLAS
has **no INT4 GEMM at all** on Turing, so this is the first usable pure-int4
matmul in Triton; upstream marks the path "Not implemented". Triton INT8 is
**~2.1× faster than cuBLAS INT8**.

### FP16 GEMM — matching NVIDIA's hand-tuned cuBLAS

![FP16 GEMM](.github/assets/benchmarks/fp16-gemm.png)

For plain FP16 GEMM the Triton kernel reaches **≈ 87–89 % of cuBLAS**, NVIDIA's
hand-tuned vendor library, across the mid-to-large size range (three passes,
`benchmarks/gemm/21`; the figure shows the middle one).

### bf16 — an opt-in Tensor Core path

Turing's `mma.sync` has no bf16 form, so a bf16 `tl.dot` falls back to CUDA-core
FMA. bf16 is the default dtype for vLLM, SGLang and most Hugging Face
checkpoints, so a large share of real workloads never touch the Tensor Cores.

`TRITON_SM75_BF16_DOT_AS_F16=1` converts bf16 dot operands to fp16 and issues
`m16n8k8`, accumulating in fp32:

| GEMM | FMA | Tensor Core | Speedup |
|---|---|---|---|
| 1024³ | 0.742 ms | 0.060 ms | **12.4×** |
| 2048³ | 7.682 ms | 0.433 ms | **17.7×** |
| 4096³ | 63.481 ms | 4.828 ms | **13.1×** |

It is **off by default because it changes numerics.** bf16's 8 mantissa bits fit
fp16 exactly, so operands inside fp16's normal range, 6.1e-5 to 65504, convert
losslessly. Outside it both ends degrade: smaller values fall into fp16
subnormals and lose precision (relative error reaches 2.5e-2 at magnitude 1e-6),
larger ones become `inf`. A model that uses bf16 *because* fp16 overflows is
exactly what this breaks, and upstream's dot tests pass either way. Check your
own outputs before enabling it.

### When the software pipeline helps

![Pipeline regime](.github/assets/benchmarks/pipeline-regime.png)

This is the first software pipeline implemented for Turing. It helps kernels
that stall on memory: FlashAttention forward gains **25 %** at head_dim=128 and
**8 %** at 64, grouped/MoE GEMM **22 %**. It makes FlashAttention backward at
head_dim=64 **9 % slower**. Layernorm, softmax and elementwise have no reduction
loop, so it does not apply to them.

`num_stages` is autotuned per kernel and size, and the 64 KB/CTA shared memory
limits how deep it can go. With two or more slots (`num_stages` ≥ 3) each K-tile
costs one `bar.sync`, and `tl.dot` is split so `ldmatrix` runs between `mma`
instructions instead of bursting at the loop head. On a 128×256 tile that moved
us from 1.7× slower than cuBLAS to **85 % of it**, and it is now the fastest
dense-GEMM config from 2048³ up. With one slot (`num_stages=2`) each K-tile
costs two barriers, which is still faster when a third slot would cost more than
it returns: on FlashAttention forward at head_dim=64 it halves occupancy.

Because the 64 KB cap is applied silently, the depth you ask for is often not the
depth you get. On a 128×128×64 tile, `num_stages` 3 and 4 compile to the same
kernel: 4 asks for 3 slots at 98304 B, over the 64 KB limit, and gets clamped
back to 2, so any difference you measure between them is noise. Two ways to
see the real depth:

```shell
TRITON_SM75_DUMP_PIPELINE_DEPTH=1 python your_script.py
# sm75 pipeline: matmul_kernel at your_script.py:12
#   num_stages=4 -> prefetching 2 iterations ahead, 2 shared slots (65536 B)
#     [CLAMPED from 3; that would need 98304 B, over the 65536 B available]
#   identical codegen to num_stages=3
```

```python
kernel.metadata.sm75_pipeline_slots            # what you got
kernel.metadata.sm75_pipeline_slots_requested  # what you asked for
```

A runnable INT8/INT4 example is in
[`python/tutorials/12-turing-integer-matmul.py`](python/tutorials/12-turing-integer-matmul.py).

## Installation

```shell
git clone https://github.com/Chennesxu/triton-turing.git
cd triton-turing
pip install -r python/requirements.txt
pip install -e .
```

Requires a Turing GPU (sm75) and CUDA 11+. For full build instructions see the [upstream docs](https://triton-lang.org).

The repository also contains a separate `triton_kernels` package; if you need it, install it with `pip install -e python/triton_kernels`.

**Windows:** the [`windows` branch](https://github.com/Chennesxu/triton-turing/tree/windows)
carries the same sm75 work on top of
[triton-windows](https://github.com/triton-lang/triton-windows) and builds with
MSVC. Untested by us; we have no Turing card in a Windows machine.
