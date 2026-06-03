# Triton — Turing (SM 7.5) Fork

This fork restores and extends [Triton](https://github.com/triton-lang/triton)'s support for NVIDIA Turing GPUs (sm75, e.g. RTX 2080 Ti / Titan RTX).

Upstream Triton dropped Turing support after v2.x: the MMA path was gated to sm80+, and the software pipeline exclusively uses `cp.async`, which is an Ampere-only instruction. As a result, Turing GPUs silently fall back to FMA, losing all tensor-core acceleration.

## Goals

1. **Restore MMA acceleration** — re-enable `mma.sync.aligned.m16n8k8` for sm75
2. **Enable layout optimization** — `optimize_dot_operands` pass for sm75 (layout hoisting before MMA)
3. **Software double-buffering without `cp.async`** — implement a `ld.global → st.shared → bar.sync` pipeline path to overlap memory loads with MMA on Turing, replacing the Ampere-only `cp.async` path
4. **Turing-specific autotune** — autotune configs tuned for 96 KB shared memory and `m16n8k8` instruction shape

## Status

| Feature | Status |
|---|---|
| `mma.sync.aligned.m16n8k8` re-enabled for sm75 | Done |
| `optimize_dot_operands` for sm75 | Done |
| Verified on Titan RTX (fp16 matmul) | Done |
| Software double-buffering (`ld.global + bar.sync` pipeline) | Planned |
| Turing-specific autotune configs | Planned |

## Installation

```shell
git clone <this repo>
cd triton
pip install -r python/requirements.txt
pip install -e .
```

Requires a Turing GPU (sm75) and CUDA 11+. For full build instructions see the [upstream docs](https://triton-lang.org).
