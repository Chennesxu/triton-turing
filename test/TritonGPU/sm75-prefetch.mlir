// RUN: triton-opt %s -split-input-file -tritongpu-prefetch -canonicalize | FileCheck %s

// The prefetch pass on the sm75 synchronous-copy pipeline. There are no async
// tokens on Turing, so the pass accepts the dot when its operands come from
// >= 2-slot ring buffers AND the single per-iteration ttg.barrier sits between
// the dot and the first local_store (LowerLoops with
// TRITON_SM75_BARRIER_BEFORE_WRITE=1). The remainder loads are then issued
// before the barrier and the next tile's head loads after it, which keeps the
// ring's WAR/RAW invariants intact.

#A_RING = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B_RING = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#BLK_A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLK_B = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// CHECK-LABEL: tt.func @sm75_ring_split
// Head of tile 0 prefetched before the loop from the init slot.
// CHECK-DAG: %[[A0_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK-DAG: %[[A0:.+]] = ttg.local_load %[[A0_SMEM]]
// CHECK-DAG: %[[B0_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK-DAG: %[[B0:.+]] = ttg.local_load %[[B0_SMEM]]
// CHECK: scf.for
// Remainder K-slice read before the barrier, first dot on the carried head.
// CHECK-DAG: %[[A1_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 16]
// CHECK-DAG: %[[A1:.+]] = ttg.local_load %[[A1_SMEM]]
// CHECK-DAG: %[[B1_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[16, 0]
// CHECK-DAG: %[[B1:.+]] = ttg.local_load %[[B1_SMEM]]
// CHECK: %[[DOT0:.+]] = tt.dot %{{.+}}, %{{.+}}, %{{.+}} :
// CHECK-NOT: ttg.local_load
// CHECK: ttg.barrier local
// CHECK-NOT: ttg.local_load
// CHECK: ttg.local_store
// CHECK: ttg.local_store
// Next tile's head read only after the barrier and the refill of this slot.
// CHECK: ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK: ttg.local_load
// CHECK: ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK: ttg.local_load
// CHECK: tt.dot %[[A1]], %[[B1]], %[[DOT0]]
// CHECK: scf.yield
tt.func @sm75_ring_split(%lb : i32, %ub : i32, %step : i32,
                         %a_ptrs : tensor<128x32x!tt.ptr<f16>, #BLK_A>,
                         %b_ptrs : tensor<32x128x!tt.ptr<f16>, #BLK_B>) -> tensor<128x128xf32, #C> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #A_RING, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #B_RING, #smem, mutable>
  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst, %ins = %c1_i32, %ext = %c0_i32) -> (tensor<128x128xf32, #C>, i32, i32) : i32 {
    %a_view = ttg.memdesc_index %a[%ext] : !ttg.memdesc<2x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %a_val = ttg.local_load %a_view : !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable> -> tensor<128x32xf16, #A_OP>
    %b_view = ttg.memdesc_index %b[%ext] : !ttg.memdesc<2x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %b_val = ttg.local_load %b_view : !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable> -> tensor<32x128xf16, #B_OP>
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>
    %ins_p1 = arith.addi %ins, %c1_i32 : i32
    %ins_cmp = arith.cmpi sge, %ins_p1, %c2_i32 : i32
    %ins_next = arith.select %ins_cmp, %c0_i32, %ins_p1 : i32
    %a_next = tt.load %a_ptrs : tensor<128x32x!tt.ptr<f16>, #BLK_A>
    %a_ins = ttg.memdesc_index %a[%ins_next] : !ttg.memdesc<2x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %b_next = tt.load %b_ptrs : tensor<32x128x!tt.ptr<f16>, #BLK_B>
    %b_ins = ttg.memdesc_index %b[%ins_next] : !ttg.memdesc<2x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    ttg.barrier local
    ttg.local_store %a_next, %a_ins {ttg.sm75_ring_store} : tensor<128x32xf16, #BLK_A> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    ttg.local_store %b_next, %b_ins {ttg.sm75_ring_store} : tensor<32x128xf16, #BLK_B> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %ext_p1 = arith.addi %ext, %c1_i32 : i32
    %ext_cmp = arith.cmpi sge, %ext_p1, %c2_i32 : i32
    %ext_next = arith.select %ext_cmp, %c0_i32, %ext_p1 : i32
    scf.yield %acc_next, %ins_next, %ext_next : tensor<128x128xf32, #C>, i32, i32
  }
  tt.return %loop#0 : tensor<128x128xf32, #C>
}
}

// -----

#A_RING = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B_RING = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#BLK_A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLK_B = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// Barrier after the reads and before the dot (the default LowerLoops layout):
// remainder loads would land after the barrier, racing with the refill of
// their slot. The dot must be left alone.
// CHECK-LABEL: tt.func @sm75_ring_barrier_before_dot_untouched
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
tt.func @sm75_ring_barrier_before_dot_untouched(%lb : i32, %ub : i32, %step : i32,
                         %a_ptrs : tensor<128x32x!tt.ptr<f16>, #BLK_A>,
                         %b_ptrs : tensor<32x128x!tt.ptr<f16>, #BLK_B>) -> tensor<128x128xf32, #C> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #A_RING, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #B_RING, #smem, mutable>
  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst, %ins = %c1_i32, %ext = %c0_i32) -> (tensor<128x128xf32, #C>, i32, i32) : i32 {
    %a_view = ttg.memdesc_index %a[%ext] : !ttg.memdesc<2x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %a_val = ttg.local_load %a_view : !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable> -> tensor<128x32xf16, #A_OP>
    %b_view = ttg.memdesc_index %b[%ext] : !ttg.memdesc<2x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %b_val = ttg.local_load %b_view : !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable> -> tensor<32x128xf16, #B_OP>
    ttg.barrier local
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>
    %ins_p1 = arith.addi %ins, %c1_i32 : i32
    %ins_cmp = arith.cmpi sge, %ins_p1, %c2_i32 : i32
    %ins_next = arith.select %ins_cmp, %c0_i32, %ins_p1 : i32
    %a_next = tt.load %a_ptrs : tensor<128x32x!tt.ptr<f16>, #BLK_A>
    %a_ins = ttg.memdesc_index %a[%ins_next] : !ttg.memdesc<2x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    ttg.local_store %a_next, %a_ins {ttg.sm75_ring_store} : tensor<128x32xf16, #BLK_A> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %b_next = tt.load %b_ptrs : tensor<32x128x!tt.ptr<f16>, #BLK_B>
    %b_ins = ttg.memdesc_index %b[%ins_next] : !ttg.memdesc<2x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    ttg.local_store %b_next, %b_ins {ttg.sm75_ring_store} : tensor<32x128xf16, #BLK_B> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %ext_p1 = arith.addi %ext, %c1_i32 : i32
    %ext_cmp = arith.cmpi sge, %ext_p1, %c2_i32 : i32
    %ext_next = arith.select %ext_cmp, %c0_i32, %ext_p1 : i32
    scf.yield %acc_next, %ins_next, %ext_next : tensor<128x128xf32, #C>, i32, i32
  }
  tt.return %loop#0 : tensor<128x128xf32, #C>
}
}

// -----

#A_RING = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B_RING = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#BLK_A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLK_B = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// Single-slot buffers: the next tile's head would be read from the slot this
// iteration refills. Never split.
// CHECK-LABEL: tt.func @sm75_single_slot_untouched
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
tt.func @sm75_single_slot_untouched(%lb : i32, %ub : i32, %step : i32,
                         %a_ptrs : tensor<128x32x!tt.ptr<f16>, #BLK_A>,
                         %b_ptrs : tensor<32x128x!tt.ptr<f16>, #BLK_B>) -> tensor<128x128xf32, #C> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #A_RING, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #B_RING, #smem, mutable>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #C>) : i32 {
    %a_view = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<1x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %a_val = ttg.local_load %a_view : !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable> -> tensor<128x32xf16, #A_OP>
    %b_view = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<1x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %b_val = ttg.local_load %b_view : !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable> -> tensor<32x128xf16, #B_OP>
    ttg.barrier local
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>
    %a_next = tt.load %a_ptrs : tensor<128x32x!tt.ptr<f16>, #BLK_A>
    ttg.local_store %a_next, %a_view : tensor<128x32xf16, #BLK_A> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %b_next = tt.load %b_ptrs : tensor<32x128x!tt.ptr<f16>, #BLK_B>
    ttg.local_store %b_next, %b_view : tensor<32x128xf16, #BLK_B> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    ttg.barrier local
    scf.yield %acc_next : tensor<128x128xf32, #C>
  }
  tt.return %loop : tensor<128x128xf32, #C>
}
}
