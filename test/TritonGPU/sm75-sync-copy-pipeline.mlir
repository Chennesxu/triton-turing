// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-pipeline | FileCheck %s

// Full software pipelining on sm75: the expander must peel a prologue that
// prefills the buffer slots, predicate the loads via their mask operand, and
// leave every ttg.barrier unpredicated at the top level of the loop body.
// bar.sync must execute uniformly across the CTA: wrapping it in control flow
// would deadlock, and predicateOp would abort compilation if it did not
// whitelist BarrierOp.

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:75"} {
// CHECK-LABEL: @sync_copy_expand
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32

// Prologue: two prefill iterations, each load masked by a trip-count guard.
// CHECK: tt.load %{{.*}}, %{{.*}} : tensor<128x32x!tt.ptr<f16>
// CHECK: ttg.local_store
// CHECK: ttg.barrier local
// CHECK: tt.load %{{.*}}, %{{.*}} : tensor<128x32x!tt.ptr<f16>
// CHECK: ttg.local_store
// CHECK: ttg.barrier local

// Kernel: consumer side first (local_load + barrier), then the next tile's
// masked load and store. No op may be wrapped in control flow, and no
// unresolved ttg.mask may survive.
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK-NOT: ttg.mask
// CHECK:   %[[VAL:.*]] = ttg.local_load
// CHECK-NOT: scf.if
// CHECK:   ttg.barrier local
// CHECK-NOT: scf.if
// CHECK:   "use"(%[[VAL]])
// CHECK-NOT: scf.if
// CHECK:   tt.load %{{.*}}, %{{.*}} : tensor<128x32x!tt.ptr<f16>
// CHECK-NOT: scf.if
// CHECK:   ttg.local_store
// CHECK-NOT: scf.if
// CHECK:   ttg.barrier local
// CHECK-NOT: scf.if
// CHECK-NOT: ttg.mask
// CHECK:   scf.yield
// CHECK: ttg.local_dealloc %[[ALLOC]]
tt.func @sync_copy_expand(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 16]> : tensor<2xi32>}) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}
