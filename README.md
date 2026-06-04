# Triton-Turing

**Triton-Turing** is a community-maintained fork of [Triton](https://github.com/triton-lang/triton) focused on restoring high-performance Tensor Core support for NVIDIA Turing GPUs (SM75: RTX 2080 Ti, Titan RTX).

Upstream Triton supports Turing's MMA instructions, but critical optimizations were gated to SM80+ (Ampere and later). Specifically, the `optimize_dot_operands` pass (which hoists layout conversions before matmul) was disabled for sm75, and the software pipeline exclusively uses `cp.async`, an Ampere-only instruction. As a result, Turing performance degrades significantly compared to its tensor-core potential.

## Goals

1. **Enable layout optimization** — `optimize_dot_operands` pass for sm75 (layout hoisting before MMA)
2. **Software double-buffering without `cp.async`** — implement a `ld.global → st.shared → bar.sync` pipeline path to overlap memory loads with MMA on Turing
3. **Turing-specific autotune** — configs tuned for 96 KB shared memory and native instruction shapes (fp16: `m16n8k8`, int8: `m8n8k16`)
4. **int4 MMA support** — implement `m8n8k32` instruction path for int4 precision (hardware-supported but not implemented in upstream Triton)

## Status

| Feature | Status |
|---|---|
| `optimize_dot_operands` pass enabled for sm75 | ✅ Done |
| Verified on Titan RTX (fp16 matmul) | ✅ Done |
| Software double-buffering (`ld.global + bar.sync` pipeline) | 🚧 Planned |
| Turing-specific autotune configs | 🚧 Planned |
| int4 MMA (`m8n8k32`) support | 🚧 Planned |

## Installation

```shell
git clone <this repo>
cd triton
pip install -r python/requirements.txt
pip install -e .
```

Requires a Turing GPU (sm75) and CUDA 11+. For full build instructions see the [upstream docs](https://triton-lang.org).
