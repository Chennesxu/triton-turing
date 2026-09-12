// RUN: triton-opt %s -split-input-file -tritongpu-prefetch -canonicalize | FileCheck %s

// The prefetch pass on the sm75 synchronous-copy pipeline. There are no async
// tokens on Turing, so the pass accepts the dot when its operands come from
// >= 2-slot ring buffers AND the single per-iteration ttg.barrier sits between
// the dot and the first local_store (the placement LowerLoops uses for >= 2
// slots). The remainder loads are then issued
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

// -----

// int4 packs eight i4 into one i32 in shared memory and reinterprets at the
// register level, so the ring's memdesc is 8x narrower than the dot operand
// type. The prefetch subslice is sized in dot-operand elements, so slicing
// this buffer runs past the end of the allocation. The pass must leave it
// alone even though every other sm75 condition holds: >= 2 slots and the
// barrier between the dot and the first local_store.

#A4_RING = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#B4_RING = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#BLK4_A = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLK4_B = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#C4 = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#A4_PACKED = #ttg.dot_op<{opIdx = 0, parent = #C4, kWidth = 1}>
#B4_PACKED = #ttg.dot_op<{opIdx = 1, parent = #C4, kWidth = 1}>
#A4_OP = #ttg.dot_op<{opIdx = 0, parent = #C4, kWidth = 8}>
#B4_OP = #ttg.dot_op<{opIdx = 1, parent = #C4, kWidth = 8}>
#smem4 = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// CHECK-LABEL: tt.func @sm75_int4_packed_ring_untouched
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: scf.for
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.dot
// CHECK-NOT: tt.dot
// CHECK: scf.yield
tt.func @sm75_int4_packed_ring_untouched(%lb : i32, %ub : i32, %step : i32,
                         %a_ptrs : tensor<128x8x!tt.ptr<i32>, #BLK4_A>,
                         %b_ptrs : tensor<8x128x!tt.ptr<i32>, #BLK4_B>) -> tensor<128x128xi32, #C4> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant dense<0> : tensor<128x128xi32, #C4>
  %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x8xi32, #A4_RING, #smem4, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<2x8x128xi32, #B4_RING, #smem4, mutable>
  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst, %ins = %c1_i32, %ext = %c0_i32) -> (tensor<128x128xi32, #C4>, i32, i32) : i32 {
    %a_view = ttg.memdesc_index %a[%ext] : !ttg.memdesc<2x128x8xi32, #A4_RING, #smem4, mutable> -> !ttg.memdesc<128x8xi32, #A4_RING, #smem4, mutable>
    %a_packed = ttg.local_load %a_view : !ttg.memdesc<128x8xi32, #A4_RING, #smem4, mutable> -> tensor<128x8xi32, #A4_PACKED>
    %a_val = tt.reinterpret_as_int4 %a_packed {axis = 1 : i32} : tensor<128x8xi32, #A4_PACKED> -> tensor<128x64xi4, #A4_OP>
    %b_view = ttg.memdesc_index %b[%ext] : !ttg.memdesc<2x8x128xi32, #B4_RING, #smem4, mutable> -> !ttg.memdesc<8x128xi32, #B4_RING, #smem4, mutable>
    %b_packed = ttg.local_load %b_view : !ttg.memdesc<8x128xi32, #B4_RING, #smem4, mutable> -> tensor<8x128xi32, #B4_PACKED>
    %b_val = tt.reinterpret_as_int4 %b_packed {axis = 0 : i32} : tensor<8x128xi32, #B4_PACKED> -> tensor<64x128xi4, #B4_OP>
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x64xi4, #A4_OP> * tensor<64x128xi4, #B4_OP> -> tensor<128x128xi32, #C4>
    %ins_p1 = arith.addi %ins, %c1_i32 : i32
    %ins_cmp = arith.cmpi sge, %ins_p1, %c2_i32 : i32
    %ins_next = arith.select %ins_cmp, %c0_i32, %ins_p1 : i32
    %a_next = tt.load %a_ptrs : tensor<128x8x!tt.ptr<i32>, #BLK4_A>
    %a_ins = ttg.memdesc_index %a[%ins_next] : !ttg.memdesc<2x128x8xi32, #A4_RING, #smem4, mutable> -> !ttg.memdesc<128x8xi32, #A4_RING, #smem4, mutable>
    %b_next = tt.load %b_ptrs : tensor<8x128x!tt.ptr<i32>, #BLK4_B>
    %b_ins = ttg.memdesc_index %b[%ins_next] : !ttg.memdesc<2x8x128xi32, #B4_RING, #smem4, mutable> -> !ttg.memdesc<8x128xi32, #B4_RING, #smem4, mutable>
    ttg.barrier local
    ttg.local_store %a_next, %a_ins {ttg.sm75_ring_store} : tensor<128x8xi32, #BLK4_A> -> !ttg.memdesc<128x8xi32, #A4_RING, #smem4, mutable>
    ttg.local_store %b_next, %b_ins {ttg.sm75_ring_store} : tensor<8x128xi32, #BLK4_B> -> !ttg.memdesc<8x128xi32, #B4_RING, #smem4, mutable>
    scf.yield %acc_next, %ins_next, %ins : tensor<128x128xi32, #C4>, i32, i32
  }
  tt.return %loop#0 : tensor<128x128xi32, #C4>
}
}
