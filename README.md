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

Benchmarks and the reasoning behind each result are on the
[`main` branch](https://github.com/Chennesxu/triton-turing). **They were all
measured on Linux, on a Titan RTX** — nothing here has been benchmarked on
Windows, and the numbers should not be read as Windows results.

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
