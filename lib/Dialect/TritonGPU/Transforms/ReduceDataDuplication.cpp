#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUREDUCEDATADUPLICATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

/// `src` has just been placed in shared memory as `buffer`, to serve a
/// `convert_layout` into a dot operand. Look for a *second* use of `src` of
/// the shape
///
///   trans(convert_layout(src) -> #linear) -> #dot_operand
///
/// and serve it from the same buffer, as `local_load(memdesc_trans(buffer))`.
///
/// The walk below skips that inner conversion because its result is a linear
/// layout, not a dot operand -- so it survives to codegen as a shared-memory
/// round trip of its own: a scratch buffer plus a store, a barrier and a load
/// on every loop iteration, spent entirely on landing the value in a layout
/// whose register-level transpose happens to be the dot operand. But the value
/// is already in shared memory, and shared memory has no preferred
/// orientation. A transposed *view* of the buffer we just allocated reads the
/// same bytes, so the conversion, its scratch and its barrier all disappear.
///
/// The Flash-Attention backward kernel builds exactly this three times per
/// inner iteration -- trans(qT) for dK, trans(do) for dP, trans(kT) for dQ --
/// each alongside an untransposed dot use of the same value.
///
/// Turing only. MMAv3 and newer fold transposes into the MMA instruction
/// itself (see FuseTransMMAV3Plus in OptimizeDotOperands), so they never
/// produce this pattern; Ampere would, but it is out of scope here and
/// untested.
///
/// There is a width cutoff, and it is empirical. Serving the transpose from
/// the buffer keeps that buffer live until the second dot, which both raises
/// peak shared memory and changes how ptxas schedules the loop. The
/// Flash-Attention backward kernel runs at the 255-register cap, so it sits on
/// a spill cliff and the transform can push it either way. Measured on a TITAN
/// RTX at N=2048, base -> rewritten:
///
///   HEAD_DIM=64,  32/128/128/32  43264 -> 41216 B shared, 1.67 -> 1.44 ms
///   HEAD_DIM=64,  64/128/128/64  53760 -> 49664 B shared, 1.54 -> 1.42 ms
///   HEAD_DIM=128, 32/64/64/32    51456 -> 53504 B shared, 3.90 -> 4.13 ms
///
/// and local (spill) traffic moves 40 -> 2.6 MB at HEAD_DIM=64 but 15 -> 688 MB
/// at HEAD_DIM=128. The scratch and barriers it removes are the same in both
/// cases -- what differs is whether the longer live range fits. The two head
/// dims are the only points measured, so the cutoff below separates the
/// observations rather than deriving a limit; widen it only with numbers.
constexpr int64_t kMaxTransposedOperandWidth = 64;

void reuseBufferForTransposedUses(Value src, LocalAllocOp buffer) {
  SmallVector<triton::TransOp> transposes;
  for (Operation *user : src.getUsers()) {
    auto cvt = dyn_cast<ConvertLayoutOp>(user);
    // The conversion must exist only to feed the transpose; if it has other
    // users it has to stay, and rewriting the transpose saves nothing.
    if (!cvt || !cvt->hasOneUse())
      continue;
    auto transOp = dyn_cast<triton::TransOp>(*cvt->getUsers().begin());
    if (!transOp || transOp.getOrder() != ArrayRef<int32_t>({1, 0}))
      continue;
    auto resultTy = dyn_cast<RankedTensorType>(transOp.getType());
    if (!resultTy)
      continue;
    auto dotEnc = dyn_cast<DotOperandEncodingAttr>(resultTy.getEncoding());
    if (!dotEnc)
      continue;
    auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(dotEnc.getParent());
    if (!mmaEnc || !mmaEnc.isTuring())
      continue;
    // Width here is the operand's non-contraction extent: the m of an A
    // operand, the n of a B operand. See kMaxTransposedOperandWidth.
    int64_t width = resultTy.getShape()[dotEnc.getOpIdx() == 0 ? 0 : 1];
    if (width > kMaxTransposedOperandWidth)
      continue;
    // `buffer` is inserted at the untransposed conversion, which may sit
    // either side of the transpose. Same block plus a position check is
    // enough dominance for the case this targets: both uses live in one loop
    // body.
    if (buffer->getBlock() != transOp->getBlock() ||
        !buffer->isBeforeInBlock(transOp))
      continue;
    transposes.push_back(transOp);
  }

  for (triton::TransOp transOp : transposes) {
    OpBuilder builder(transOp);
    auto view = MemDescTransOp::create(builder, transOp.getLoc(), buffer,
                                       ArrayRef<int32_t>({1, 0}));
    auto load =
        LocalLoadOp::create(builder, transOp.getLoc(), transOp.getType(), view);
    transOp.getResult().replaceAllUsesWith(load.getResult());
    transOp.erase();
    // The conversion that fed it is now dead. Leave it to the canonicalizer
    // rather than erasing a ConvertLayoutOp the caller may still be holding.
  }
}

} // namespace

class TritonGPUReduceDataDuplicationPass
    : public impl::TritonGPUReduceDataDuplicationBase<
          TritonGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Collect before rewriting: reuseBufferForTransposedUses() erases
    // transposes the walk has not reached yet, which a live walk does not
    // tolerate.
    SmallVector<triton::gpu::ConvertLayoutOp> cvtOps;
    mod.walk(
        [&](triton::gpu::ConvertLayoutOp cvtOp) { cvtOps.push_back(cvtOp); });
    for (triton::gpu::ConvertLayoutOp cvtOp : cvtOps) {
      // The predicate and the buffer type live in Utility.cpp so that the
      // sm75 pipeline-depth clamp, which has to reason about this allocation
      // long before this pass runs, asks the same question we answer here.
      std::optional<triton::gpu::MemDescType> tmpType =
          triton::getReduceDataDuplicationBufferType(cvtOp);
      if (!tmpType)
        continue;
      OpBuilder builder(cvtOp);
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   *tmpType, cvtOp.getSrc());
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      Value src = cvtOp.getSrc();
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
      reuseBufferForTransposedUses(src, tmp);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
