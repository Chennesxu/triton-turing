// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// Turing (sm75) int4 Tensor Core: end-to-end layout assignment for a packed
// int4 dot. The frontend emits `tt.reinterpret_as_int4(%packed_i32)` feeding a
// tt.dot whose operands carry a blocked dot-operand layout. BlockedToMMA must
// NOT give the i4 tensor a dot-operand layout via a convert_layout on i4 — that
// routes through shared memory, and i4 local_alloc is illegal (shared element
// width must be a multiple of 8).
//
// The interception in convertDotOperandForMMA instead applies the dot-operand
// layout to the int32 SOURCE (kWidth=1, legal for 32-bit) and re-creates the
// reinterpret on top, so the i4 tensor only ever exists in dot-operand form
// (kWidth=8). This pins that invariant: int32 convert_layout + i4 reinterpret
// at the dot-operand level, an i4 dot fed to an nvidia_mma result, and crucially
// NO i4 local_alloc / no i4 convert_layout.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, {{.*}}instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @int4_dot
  tt.func public @int4_dot(%a_packed: tensor<16x8xi32, #blocked>, %b_packed: tensor<8x8xi32, #blocked>) -> tensor<16x8xi32, #blocked6> {
    %cst = arith.constant dense<0> : tensor<16x8xi32, #blocked6>

    // A: int32 source gets the dot-operand layout (kWidth=1), then reinterpret
    // relabels it to i4 dot-operand (kWidth=8) — no i4 convert_layout.
    // CHECK: ttg.convert_layout %{{.*}} : tensor<16x8xi32, #blocked> -> tensor<16x8xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK: tt.reinterpret_as_int4 %{{.*}} {axis = 1 : i32} : tensor<16x8xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> -> tensor<16x64xi4, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %a = tt.reinterpret_as_int4 %a_packed {axis = 1 : i32} : tensor<16x8xi32, #blocked> -> tensor<16x64xi4, #blocked5>

    // B: same on opIdx=1.
    // CHECK: ttg.convert_layout %{{.*}} : tensor<8x8xi32, #blocked> -> tensor<8x8xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    // CHECK: tt.reinterpret_as_int4 %{{.*}} {axis = 0 : i32} : tensor<8x8xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x8xi4, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %b = tt.reinterpret_as_int4 %b_packed {axis = 0 : i32} : tensor<8x8xi32, #blocked> -> tensor<64x8xi4, #blocked>

    %a_dot = ttg.convert_layout %a : tensor<16x64xi4, #blocked5> -> tensor<16x64xi4, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>>
    %b_dot = ttg.convert_layout %b : tensor<64x8xi4, #blocked> -> tensor<64x8xi4, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>>

    // i4 dot operands (kWidth=8) feed an nvidia_mma result.
    // CHECK: tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x64xi4, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x8xi4, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x8xi32, #mma>
    %d = tt.dot %a_dot, %b_dot, %cst, inputPrecision = tf32 : tensor<16x64xi4, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<64x8xi4, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<16x8xi32, #blocked6>

    // The i4 tensor never routes through shared memory.
    // CHECK-NOT: ttg.local_alloc
    tt.return %d : tensor<16x8xi32, #blocked6>
  }
}
