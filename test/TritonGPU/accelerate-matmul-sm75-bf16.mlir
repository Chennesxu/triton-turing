// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s --check-prefix=OFF
// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul='sm75-bf16-dot-as-f16=true' | FileCheck %s --check-prefix=ON

// Turing's mma.sync has no bf16 form, so getMMAVersionSafe refuses an MMA
// layout for a bf16 dot and decomposeMixedModeDotOp promotes the operands to
// f32 for the FMA path. sm75-bf16-dot-as-f16 opts into the fp16 tensor core
// instead: the dot keeps an MMA layout and its operands are converted to fp16.
//
// bf16's 8 mantissa bits are exactly representable in fp16, so the conversion
// is lossless; what it gives up is range (fp16 stops at 65504, bf16 reaches
// ~3e38). That is why this is an option rather than unconditional, and why
// both directions are pinned here -- flipping the default would silently
// change numerics for every bf16 kernel on sm75.
//
// The conversion goes through f32: bf16 -> f16 is not an arith.extf (same
// width) and the NVIDIA lowering tables have no direct bf16 -> f16 entry.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // The Turing MMA layout is versionMinor 1; pin it so a future default flip
  // or a version bump is visible here.
  // ON: #mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1
  // OFF-LABEL: @bf16_dot_sm75
  // ON-LABEL: @bf16_dot_sm75
  tt.func public @bf16_dot_sm75(%a: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                                %b: tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>)
      -> tensor<16x8xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>

    // Default: no MMA layout. The operands are widened to f32 and the dot
    // result stays on #blocked, i.e. the FMA path.
    // OFF-NOT: tt.fp_to_fp
    // OFF: tt.dot {{.*}} tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> {{.*}} -> tensor<16x8xf32, #blocked>

    // Enabled: operands go bf16 -> f32 -> f16 (RTNE) and the dot lands on a
    // Turing MMA layout (versionMajor 2, versionMinor 1), accumulating in f32.
    // ON: arith.extf {{.*}} : tensor<16x16xbf16, {{.*}}parent = #mma{{.*}} to tensor<16x16xf32
    // ON: tt.fp_to_fp {{.*}}, rounding = rtne : tensor<16x16xf32, {{.*}} -> tensor<16x16xf16
    // ON: arith.extf {{.*}} : tensor<16x8xbf16, {{.*}}parent = #mma{{.*}} to tensor<16x8xf32
    // ON: tt.fp_to_fp {{.*}}, rounding = rtne : tensor<16x8xf32, {{.*}} -> tensor<16x8xf16
    // ON: tt.dot {{.*}} tensor<16x16xf16, {{.*}} -> tensor<16x8xf32, #mma>
    %d = tt.dot %a, %b, %cst, inputPrecision = tf32 :
        tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> *
        tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> ->
        tensor<16x8xf32, #blocked>
    tt.return %d : tensor<16x8xf32, #blocked>
  }
}

// -----

// The option is scoped to sm75. Ampere has a native bf16 MMA, so the operands
// must stay bf16 there even with the option on.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // ON-LABEL: @sm80_keeps_native_bf16
  tt.func public @sm80_keeps_native_bf16(%a: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                                         %b: tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>)
      -> tensor<16x8xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32, #blocked>
    // ON-NOT: tt.fp_to_fp
    // ON: tt.dot {{.*}} tensor<16x16xbf16, {{.*}} -> tensor<16x8xf32, #mma>
    %d = tt.dot %a, %b, %cst, inputPrecision = tf32 :
        tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> *
        tensor<16x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> ->
        tensor<16x8xf32, #blocked>
    tt.return %d : tensor<16x8xf32, #blocked>
  }
}
