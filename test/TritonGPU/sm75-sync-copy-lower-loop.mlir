// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-test-pipeline-lower-loop -canonicalize | FileCheck %s

// Turing (sm75) has no cp.async. LowerLoops routes loads to the synchronous
// copy path: the global tt.load is kept alive as the data source, its result
// is staged through shared memory with local_store/local_load, and CTA-wide
// ttg.barrier ops synchronize the producer and consumer sides.

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// The local_store must consume the tt.load result (not the local_load
// result), and both barriers must carry the schedule of their side.
// CHECK-LABEL: @sync_copy_dataflow
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK: scf.for
// CHECK:   %[[LOAD:.*]] = tt.load %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS:.*]] = ttg.memdesc_index %[[ALLOC]]{{\[}}%{{.*}}{{\]}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.local_store %[[LOAD]], %[[INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.barrier local {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[EXT:.*]] = ttg.memdesc_index %[[ALLOC]]{{\[}}%{{.*}}{{\]}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[VAL:.*]] = ttg.local_load %[[EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   ttg.barrier local {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[VAL]])
// CHECK: ttg.local_dealloc %[[ALLOC]]
tt.func @sync_copy_dataflow(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 16]> : tensor<2xi32>}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// When the load feeds a local_alloc with a matching shared encoding,
// replaceUsesWithLocalLoad folds the alloc into the pipeline buffer: the
// consumer reads the buffer view directly, no local_load is created, and the
// original local_alloc disappears. This path used to crash with a dangling
// insertion point because the alloc (the load's first use) is erased.
// CHECK-LABEL: @sync_copy_local_alloc_user
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK: scf.for
// CHECK:   %[[LOAD:.*]] = tt.load %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.local_store %[[LOAD]], %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.barrier local {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[EXT:.*]] = ttg.memdesc_index %[[ALLOC]]{{\[}}%{{.*}}{{\]}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK-NOT: ttg.local_load
// CHECK-NOT: ttg.local_alloc
// CHECK:   ttg.barrier local {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[EXT]])
tt.func @sync_copy_local_alloc_user(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 16]> : tensor<2xi32>}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %sh = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> !ttg.memdesc<128x32xf16, #shared, #smem>
    "use"(%sh) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (!ttg.memdesc<128x32xf16, #shared, #smem>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// Multibuffering beyond double buffering: a stage distance of 3 must allocate
// 3 slots and rotate both the insert and the extract index modulo 3, on
// distinct index chains.
// CHECK-LABEL: @sync_copy_three_stages
// CHECK-DAG: %[[BUFS:.*]] = arith.constant {{.*}} 3 : i32
// CHECK-DAG: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x32
// CHECK: scf.for
// CHECK:   arith.cmpi sge, %{{.*}}, %[[BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INSIDX:.*]] = arith.select %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   arith.cmpi sge, %{{.*}}, %[[BUFS]] {loop.cluster = 0 : i32, loop.stage = 3 : i32}
// CHECK:   %[[EXTIDX:.*]] = arith.select %{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
// CHECK:   ttg.memdesc_index %[[ALLOC]]{{\[}}%[[INSIDX]]{{\]}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.local_store
// CHECK:   ttg.barrier local {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.memdesc_index %[[ALLOC]]{{\[}}%[[EXTIDX]]{{\]}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}
// CHECK:   ttg.local_load
// CHECK:   ttg.barrier local {loop.cluster = 0 : i32, loop.stage = 3 : i32}
tt.func @sync_copy_three_stages(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 16]> : tensor<2xi32>}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}
