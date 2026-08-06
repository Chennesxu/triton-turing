// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

// A value that feeds both an untransposed dot operand and a transposed one
// only needs to reach shared memory once: the transposed operand is a
// transposed *view* of the same buffer. Without this, the transposed use
// lowers to its own shared-memory round trip -- a scratch buffer plus a
// store, a barrier and a load, per loop iteration.
//
// The IR below is what the Flash-Attention backward kernel actually presents
// to this pass at HEAD_DIM=64: qT converted once into a dot operand for
// dot(k, qT), and separately into a linear "transpose preimage" layout whose
// register transpose is the dot operand for dot(dsT, trans(qT)).

// CHECK-LABEL: reuse_buffer_for_transposed_operand
//       CHECK:   %[[BUF:.*]] = ttg.local_alloc %arg0
//       CHECK:   ttg.local_load %[[BUF]]
//       CHECK:   %[[VIEW:.*]] = ttg.memdesc_trans %[[BUF]] {order = array<i32: 1, 0>}
//       CHECK:   ttg.local_load %[[VIEW]]
//   CHECK-NOT:   tt.trans

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reuse_buffer_for_transposed_operand(%arg0: tensor<64x32xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #linear>
    %1 = ttg.convert_layout %arg0 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %2 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<64x32xf16, #linear> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
}

// -----

// Same shape, but the transposed operand is 128 wide (HEAD_DIM=128). Keeping
// the buffer live until the second dot costs more than the conversion it
// saves there -- see kMaxTransposedOperandWidth in ReduceDataDuplication.cpp
// for the measurements. The transpose must survive untouched.

// CHECK-LABEL: too_wide_to_reuse
//       CHECK:   ttg.local_alloc %arg0
//       CHECK:   tt.trans
//   CHECK-NOT:   ttg.memdesc_trans

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [32, 0], [64, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"ttg.target" = "cuda:75", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @too_wide_to_reuse(%arg0: tensor<128x32xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #linear>
    %1 = ttg.convert_layout %arg0 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %2 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<128x32xf16, #linear> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
}

// -----

// Turing only (versionMinor = 1). Ampere would hit the same shape but is
// untested here, and MMAv3+ folds transposes into the instruction instead.

// CHECK-LABEL: ampere_is_left_alone
//       CHECK:   tt.trans
//   CHECK-NOT:   ttg.memdesc_trans

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ampere_is_left_alone(%arg0: tensor<64x32xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #linear>
    %1 = ttg.convert_layout %arg0 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %2 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<64x32xf16, #linear> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    tt.return
  }
}
