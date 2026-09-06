# Triton-Turing (Windows)

A Turing (SM75) build of Triton: [triton-windows](https://github.com/triton-lang/triton-windows)
as the base, plus this fork's Tensor Core work for NVIDIA Turing GPUs
(RTX 2080 Ti, Titan RTX).

Upstream Triton gates its most important optimization — the software pipeline —
behind SM80+, because it is built on `cp.async`, an Ampere-only instruction.
Upstream also does not build on Windows. This branch addresses both.

> **Not tested on Windows by us.** We develop on Linux and have no Turing card
> in a Windows machine, so this branch is unverified on the platform it targets.
> Please report what breaks.

## What it adds

| Feature | Status |
|---|---|
| Software pipelining (multi-stage `ld.global + bar.sync`) — first ever for Turing | ✅ |
| Turing-specific autotune configs | ✅ |
| int8 GEMM (`m8n8k16`) | ✅ |
| int4 MMA (`m8n8k32`) — first usable pure-int4 matmul in Triton | ✅ |
| FlashAttention-2 forward + backward (pipelined) | ✅ |
| bf16 dot on the fp16 Tensor Core — opt-in, `TRITON_SM75_BF16_DOT_AS_F16=1` | ✅ |

Benchmarks and the reasoning behind each result are on the
[`main` branch](https://github.com/Chennesxu/triton-turing). **They were all
measured on Linux, on a Titan RTX** — nothing here has been benchmarked on
Windows, and the numbers should not be read as Windows results.

Turing has no bf16 Tensor Core, so a bf16 `tl.dot` falls back to CUDA-core FMA.
`TRITON_SM75_BF16_DOT_AS_F16=1` converts the operands to fp16 and issues
`m16n8k8` instead — 12-18x faster on Linux, but **off by default because it
changes numerics**: operands below 6.1e-5 lose precision to fp16 subnormals and
operands above 65504 become `inf`. See the `main` branch README.

Turing's 64 KB/CTA shared memory caps how deep the software pipeline can go,
and the cap is applied silently — so the `num_stages` you ask for is often not
the depth you get. On a 128×128×64 tile, `num_stages` 3 and 4 compile to the
same kernel: 4 asks for 3 slots at 98304 B, over the limit, and gets clamped
back to 2. Any difference you measure between them is noise. Two ways to see
the real depth:

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

Build from source with MSVC. From an **x64 Native Tools Command Prompt for VS 2022**:

```shell
git clone -b windows https://github.com/Chennesxu/triton-turing.git
cd triton-turing
python setup.py bdist_wheel -v
pip install dist\*.whl
```

Requires a Turing GPU (sm75), CUDA 11+, and MSVC v143. LLVM is downloaded
automatically. For build details and troubleshooting see
[triton-windows' BUILD.md](https://github.com/triton-lang/triton-windows/blob/readme/BUILD.md).

## Differences from `main`

Beyond Windows support itself, this branch inherits three things from its
triton-windows base:

- **Triton 3.7.1** instead of 3.7.0. `main` tracks a mid-cycle snapshot of
  upstream `main`; this branch tracks the 3.7.x release line, so the trees
  differ by more than the version string suggests.
- **A different pinned LLVM** (`1f126a6`, via `cmake/llvm-hash.txt` rather than
  `cmake/llvm-info.json`).
- **Different fp16→fp8e5m2 rounding.** triton-windows carries its own rewrite of
  that conversion to a proper RTNE; it is not an upstream change. sm75 has no
  native FP8, so this path is live on Turing.
