// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s

// Turing (sm75) int4 Tensor Core: s4 x s4 -> s32 via m8n8k32.
// Verifies the lowering added in MMAv2.cpp emits the m8n8k32 imma instruction
// for i4 dot operands. versionMinor=1 selects the Turing path; kWidth=8 packs
// 8 int4 values per 32-bit register; instrShape=[16,8] is the fixed MMAv2 tile.
//
// NOTE: i4 cannot go through ttg.local_alloc (shared-memory element width must
// be a multiple of 8). Real int4 data flow keeps data packed as int32 in shared
// memory and reinterprets to i4 at the dot-operand register level (frontend TBD).
// This test feeds i4 dot operands directly to isolate the MMA lowering.

#mma0 = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=8}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_int4
  tt.func @convert_dot_int4(%A: tensor<16x64xi4, #dot_operand_a>, %B: tensor<64x16xi4, #dot_operand_b>) {
    %cst0 = arith.constant dense<0> : tensor<16x16xi32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32
    %D = tt.dot %A, %B, %cst0 : tensor<16x64xi4, #dot_operand_a> * tensor<64x16xi4, #dot_operand_b> -> tensor<16x16xi32, #mma0>

    tt.return
  }
}
