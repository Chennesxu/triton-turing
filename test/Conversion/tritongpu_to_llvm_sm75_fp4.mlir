// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=75 -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --check-prefix=SM75
// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm=compute-capability=90 -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --check-prefix=SM90

// e2m1 -> fp16 upcast (the dot_scaled path when the other operand is fp16).
// The original sequence goes through cvt.rn.f16x2.e4m3x2, which needs sm_89,
// so below that the lowering must use the cvt-free byte-table variant.
// sm_89 and up keep the original sequence.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // SM75-LABEL: fp4_to_fp16
  // SM75-NOT: e4m3x2
  // SM75: prmt.b32 {{.*}} 0x3E3C3800, 0x46444240
  // SM75: prmt.b32 {{.*}} 0x1404
  // SM75-NOT: e4m3x2
  // SM90-LABEL: fp4_to_fp16
  // SM90: cvt.rn.f16x2.e4m3x2
  tt.func @fp4_to_fp16(%arg0: tensor<32x16xi8, #blocked>, %ptr: tensor<32x32x!tt.ptr<f16>, #blocked1>) {
    %0 = ttg.fp4_to_fp %arg0 {axis = 1 : i32} : tensor<32x16xi8, #blocked> -> tensor<32x32xf16, #blocked1>
    tt.store %ptr, %0 : tensor<32x32x!tt.ptr<f16>, #blocked1>
    tt.return
  }

  // The bf16 sequence never needed the fp8 cvt and is the same on both.
  // SM75-LABEL: fp4_to_bf16
  // SM75-NOT: e4m3x2
  // SM90-LABEL: fp4_to_bf16
  // SM90-NOT: e4m3x2
  tt.func @fp4_to_bf16(%arg0: tensor<32x16xi8, #blocked>, %ptr: tensor<32x32x!tt.ptr<bf16>, #blocked1>) {
    %0 = ttg.fp4_to_fp %arg0 {axis = 1 : i32} : tensor<32x16xi8, #blocked> -> tensor<32x32xbf16, #blocked1>
    tt.store %ptr, %0 : tensor<32x32x!tt.ptr<bf16>, #blocked1>
    tt.return
  }
}
