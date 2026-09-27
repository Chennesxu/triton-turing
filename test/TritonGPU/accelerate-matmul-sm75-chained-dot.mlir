// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// Chained dots on sm75 whose accumulator is wider than tall get the warp grid
// that minimises the operand data each warp loads, not upstream's
// [1, numWarps] (accelerate-matmul.mlir @chained_dot pins that on cuda:80).
// The cost per MxNxK dot is max(M/wm, 16) * K + K * max(N/wn, 8), summed over
// the function's chained dots.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // 1x8: 9216 + 5120; 2x4: 6144 + 4096; 4x2: 6144 + 5120; 8x1: 10240 + 9216.
  // CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
  // CHECK-LABEL: chained_dot_w8
  tt.func public @chained_dot_w8(
    %arg0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$MMA]]>
    %r = tt.dot %c, %arg2, %cst_1 :
      tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// The FA2 backward dQ shape at 4 warps: 1x4: 9216 + 3072; 2x2: 6144 + 3072;
// 4x1: 6144 + 4608.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
  // CHECK-LABEL: chained_dot_w4
  tt.func public @chained_dot_w4(
    %arg0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
  // CHECK: tt.dot {{.*}} -> tensor<64x32xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x32xf32, #blocked>
    %t = arith.truncf %d : tensor<64x32xf32, #blocked> to tensor<64x32xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$MMA]]>
    %r = tt.dot %c, %arg2, %cst_1 :
      tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// No accumulator is wider than tall: upstream's [numWarps, 1] stands, so the
// first result feeds the second dot from registers (FA2 at d64, forward at
// BLOCK_M 128).
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
  // CHECK-LABEL: chained_dot_tall
  tt.func public @chained_dot_tall(
    %arg0: tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x64xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 :
      tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$MMA]]>
    %r = tt.dot %c, %arg2, %cst_1 :
      tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x64xf32, #blocked1>
    tt.return %r : tensor<64x64xf32, #blocked1>
  }
}

// -----

// M = 16 is one instruction tall: splitting it replicates warps instead of
// shrinking their tiles, so 1x4 stays (1x4: 4096 + 3072; 2x2: 6144 + 5120).
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
  // CHECK-LABEL: chained_dot_m16
  tt.func public @chained_dot_m16(
    %arg0: tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<16x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked1>
  // CHECK: tt.dot {{.*}} -> tensor<16x64xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 :
      tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x64xf32, #blocked>
    %t = arith.truncf %d : tensor<16x64xf32, #blocked> to tensor<16x64xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<16x64xf16, #blocked> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: tt.dot {{.*}} -> tensor<16x128xf32, #[[$MMA]]>
    %r = tt.dot %c, %arg2, %cst_1 :
      tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<16x128xf32, #blocked1>
    tt.return %r : tensor<16x128xf32, #blocked1>
  }
}

// -----

// Two chains in one function share a grid even when each alone would differ,
// the way the FA2 backward's masked and unmasked loops carry one accumulator.
// Alone, the K=16 chain would take 4x1 (6400 vs 6656 for 2x2); summed with the
// K=32 chain (10752 vs 9216) both take 2x2.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
  // CHECK-NOT: #ttg.nvidia_mma
  // CHECK-LABEL: two_chains
  tt.func public @two_chains(
    %a0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %a1: tensor<128x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %a2: tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
    %b0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %b1: tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %b2: tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> (tensor<64x128xf32, #blocked1>, tensor<64x128xf32, #blocked1>) {
    %ca0 = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %ca1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    %cb0 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %cb1 = arith.constant dense<1.000000e+00> : tensor<64x128xf32, #blocked1>
  // CHECK: tt.dot {{.*}} -> tensor<64x16xf32, #[[$MMA]]>
    %da = tt.dot %a0, %a1, %ca0 :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x16xf32, #blocked>
    %ta = arith.truncf %da : tensor<64x16xf32, #blocked> to tensor<64x16xf16, #blocked>
    %xa = ttg.convert_layout %ta : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$MMA]]>
    %ra = tt.dot %xa, %a2, %ca1 :
      tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
  // CHECK: tt.dot {{.*}} -> tensor<64x32xf32, #[[$MMA]]>
    %db = tt.dot %b0, %b1, %cb0 :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x32xf32, #blocked>
    %tb = arith.truncf %db : tensor<64x32xf32, #blocked> to tensor<64x32xf16, #blocked>
    %xb = ttg.convert_layout %tb : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$MMA]]>
    %rb = tt.dot %xb, %b2, %cb1 :
      tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %ra, %rb : tensor<64x128xf32, #blocked1>, tensor<64x128xf32, #blocked1>
  }
}
