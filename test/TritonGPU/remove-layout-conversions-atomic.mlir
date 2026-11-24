// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// Regression: tt.atomic_rmw / tt.atomic_cas are layout anchors that also carry
// SameOperandsAndResultEncoding. LayoutPropagation must not propagate an
// operand's layout (e.g. #mma) across an atomic anchor into its result and
// downstream converts; otherwise resolving the atomic to #atomic and its
// downstream convert_layout to #mma hits assert(operandEnc) in
// rewriteGenericOpInPlace.

// Case 1: SM75 minimal repro (atomic_rmw fadd with #mma value and #store save).
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#atomic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @atomic_save(
  // CHECK-SAME: %[[PTRS:.*]]: tensor<64x64x!tt.ptr<f32>, #blocked>,
  // CHECK-SAME: %[[ACC:.*]]: tensor<64x64xf32, #mma>,
  // CHECK-SAME: %[[SAVE:.*]]: tensor<64x64x!tt.ptr<f32>, #blocked1>
  // CHECK: %[[VAL:.*]] = ttg.convert_layout %[[ACC]] : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
  // CHECK: %[[OLD:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %[[PTRS]], %[[VAL]] : (tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked>
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[OLD]] : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
  // CHECK: tt.store %[[SAVE]], %[[CVT]] : tensor<64x64x!tt.ptr<f32>, #blocked1>
  // CHECK: tt.return
  tt.func @atomic_save(%ptrs: tensor<64x64x!tt.ptr<f32>, #atomic>, %acc: tensor<64x64xf32, #mma>, %save: tensor<64x64x!tt.ptr<f32>, #store>) {
    %val = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #atomic>
    %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %val : (tensor<64x64x!tt.ptr<f32>, #atomic>, tensor<64x64xf32, #atomic>) -> tensor<64x64xf32, #atomic>
    %converted = ttg.convert_layout %old : tensor<64x64xf32, #atomic> -> tensor<64x64xf32, #store>
    tt.store %save, %converted : tensor<64x64x!tt.ptr<f32>, #store>
    tt.return
  }
}

// -----

// Case 2: SM80 target (non-SM75 resolveConflicts path).
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#atomic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @atomic_save_sm80(
  // CHECK: %[[VAL:.*]] = ttg.convert_layout %arg1 : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
  // CHECK: %[[OLD:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %[[VAL]] : (tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked>
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[OLD]] : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
  // CHECK: tt.store %arg2, %[[CVT]] : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.func @atomic_save_sm80(%ptrs: tensor<64x64x!tt.ptr<f32>, #atomic>, %acc: tensor<64x64xf32, #mma>, %save: tensor<64x64x!tt.ptr<f32>, #store>) {
    %val = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #atomic>
    %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %val : (tensor<64x64x!tt.ptr<f32>, #atomic>, tensor<64x64xf32, #atomic>) -> tensor<64x64xf32, #atomic>
    %converted = ttg.convert_layout %old : tensor<64x64xf32, #atomic> -> tensor<64x64xf32, #store>
    tt.store %save, %converted : tensor<64x64x!tt.ptr<f32>, #store>
    tt.return
  }
}

// -----

// Case 3: Predicated tt.atomic_rmw preserves mask, side effects, and old value.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#atomic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @atomic_save_masked(
  // CHECK-SAME: %[[PTRS:.*]]: tensor<64x64x!tt.ptr<f32>, #blocked>,
  // CHECK-SAME: %[[ACC:.*]]: tensor<64x64xf32, #mma>,
  // CHECK-SAME: %[[MASK:.*]]: tensor<64x64xi1, #blocked>,
  // CHECK-SAME: %[[SAVE:.*]]: tensor<64x64x!tt.ptr<f32>, #blocked1>,
  // CHECK-SAME: %[[SMASK:.*]]: tensor<64x64xi1, #blocked1>
  // CHECK: %[[VAL:.*]] = ttg.convert_layout %[[ACC]] : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
  // CHECK: %[[OLD:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %[[PTRS]], %[[VAL]], %[[MASK]] : (tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xf32, #blocked>, tensor<64x64xi1, #blocked>) -> tensor<64x64xf32, #blocked>
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[OLD]] : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
  // CHECK: tt.store %[[SAVE]], %[[CVT]], %[[SMASK]] : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.func @atomic_save_masked(%ptrs: tensor<64x64x!tt.ptr<f32>, #atomic>, %acc: tensor<64x64xf32, #mma>, %mask: tensor<64x64xi1, #atomic>, %save: tensor<64x64x!tt.ptr<f32>, #store>, %smask: tensor<64x64xi1, #store>) {
    %val = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #atomic>
    %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %val, %mask : (tensor<64x64x!tt.ptr<f32>, #atomic>, tensor<64x64xf32, #atomic>, tensor<64x64xi1, #atomic>) -> tensor<64x64xf32, #atomic>
    %converted = ttg.convert_layout %old : tensor<64x64xf32, #atomic> -> tensor<64x64xf32, #store>
    tt.store %save, %converted, %smask : tensor<64x64x!tt.ptr<f32>, #store>
    tt.return
  }
}

// -----

// Case 4: tt.atomic_cas anchor with #mma value and #store consumer.
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#atomic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @atomic_cas_save(
  // CHECK: %[[VAL:.*]] = ttg.convert_layout %arg2 : tensor<64x64xi32, #mma> -> tensor<64x64xi32, #blocked>
  // CHECK: %[[OLD:.*]] = tt.atomic_cas acq_rel, gpu, %arg0, %arg1, %[[VAL]] : (tensor<64x64x!tt.ptr<i32>, #blocked>, tensor<64x64xi32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xi32, #blocked>
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %[[OLD]] : tensor<64x64xi32, #blocked> -> tensor<64x64xi32, #blocked1>
  // CHECK: tt.store %arg3, %[[CVT]] : tensor<64x64x!tt.ptr<i32>, #blocked1>
  tt.func @atomic_cas_save(%ptrs: tensor<64x64x!tt.ptr<i32>, #atomic>, %cmp: tensor<64x64xi32, #atomic>, %acc: tensor<64x64xi32, #mma>, %save: tensor<64x64x!tt.ptr<i32>, #store>) {
    %val = ttg.convert_layout %acc : tensor<64x64xi32, #mma> -> tensor<64x64xi32, #atomic>
    %old = tt.atomic_cas acq_rel, gpu, %ptrs, %cmp, %val : (tensor<64x64x!tt.ptr<i32>, #atomic>, tensor<64x64xi32, #atomic>, tensor<64x64xi32, #atomic>) -> tensor<64x64xi32, #atomic>
    %converted = ttg.convert_layout %old : tensor<64x64xi32, #atomic> -> tensor<64x64xi32, #store>
    tt.store %save, %converted : tensor<64x64x!tt.ptr<i32>, #store>
    tt.return
  }
}
