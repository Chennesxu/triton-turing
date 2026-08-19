#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu {
namespace {

//===----------------------------------------------------------------------===//
// assignLatencies
//===----------------------------------------------------------------------===//

// Return true if the preconditions for pipelining the loop are met.
bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (isOuterLoop(forOp))
    return false;
  return true;
}

bool hasLatenciesAssigned(scf::ForOp forOp) {
  auto helper = TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (helper.getAttr(&op))
      return true;
  }
  return false;
}

void assignUserProvidedLatencies(scf::ForOp forOp,
                                 DenseMap<Operation *, int> &opLatency) {
  auto helper = TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto latencyAttr = helper.getAttr(&op)) {
      opLatency[&op] = latencyAttr.getInt();
    }
  }
}

class AssignLoadLatencies {
public:
  AssignLoadLatencies(scf::ForOp forOp, int numStages,
                      DenseMap<Operation *, int> &opLatency,
                      int computeCapability = 0)
      : forOp(forOp), numStages(numStages), opLatency(opLatency),
        computeCapability(computeCapability) {};

  void run() {
    bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
    ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
        loadOpsToIndirectionLevel(forOp, pipelineWithoutDot, axisInfoAnalysis,
                                  numStages, /*filterSmall=*/true,
                                  computeCapability);
    if (loadOpToIndLevel.empty())
      return;

    // Calculate the stage distance between applicable loads.
    int maxIndirectionLevel = 0;
    for (auto &[loadOp, info] : loadOpToIndLevel)
      maxIndirectionLevel = std::max(maxIndirectionLevel, info.first);
    unsigned loadLatency = (numStages - 1) / (maxIndirectionLevel + 1);

    if (computeCapability == 75)
      loadLatency = clampLatencyToSharedMemory(loadOpToIndLevel, loadLatency);
    if (loadLatency == 0)
      return;

    for (auto [loadOp, dist] : loadOpToIndLevel) {
      opLatency[loadOp] = loadLatency;
    }
  }

private:
  scf::ForOp forOp;
  int numStages;
  DenseMap<Operation *, int> &opLatency;
  int computeCapability;

  // Turing kernels with fixed (non-autotuned) configs have no pruning safety
  // net: if the pipeline multibuffers don't fit shared memory, compilation
  // hard-fails with OutOfResources. Degrade gracefully instead. Each unit of
  // load latency costs one buffer slot per load, so lower the latency until
  // the estimated buffer footprint fits what is left of Turing's 64KB/CTA
  // hard limit after subtracting shared memory the loop already commits to
  // (e.g. loop-invariant dot operand staging), down to 0 (no pipelining),
  // which leaves the loop as it would be without the pipeliner.
  unsigned clampLatencyToSharedMemory(
      const llvm::MapVector<Operation *, std::pair<int, Operation *>>
          &loadOpToIndLevel,
      unsigned loadLatency) {
    constexpr int64_t kSm75SharedMemoryBudget = 64 * 1024;

    // Count scalar loads too. Skipping anything that is not a RankedTensorType
    // used to look safe, but the pipeliner buffers those loads all the same:
    // SageAttention's _attn_fwd does `k_scale = tl.load(K_scale_ptr)` off a
    // scalar pointer, and that one f32 still gets a slot per stage. Leaving it
    // out made the estimate 8 B per two slots too low, which was exactly enough
    // to push the kernel past 64 KB after this function had declared it fit.
    int64_t bytesPerSlot = 0;
    for (auto &[loadOp, info] : loadOpToIndLevel) {
      Type resultTy = loadOp->getResultTypes().front();
      if (auto tensorTy = dyn_cast<RankedTensorType>(resultTy)) {
        bytesPerSlot += tensorBytes(tensorTy);
      } else if (resultTy.isIntOrFloat()) {
        bytesPerSlot += llvm::divideCeil(resultTy.getIntOrFloatBitWidth(), 8);
      }
    }
    if (bytesPerSlot == 0)
      return loadLatency;

    // local_allocs live during the loop (defined inside, or defined outside
    // with users inside) occupy shared memory concurrently with the pipeline
    // buffers. Skip allocs that merely stage a pipelined load: the sync copy
    // lowering folds those into the pipeline buffers, counting them twice
    // would be wrong.
    int64_t staticBytes = 0;
    Operation *scope = forOp->getParentOfType<triton::FuncOp>();
    if (!scope)
      scope = forOp->getParentOfType<ModuleOp>();
    // A loop always sits inside one of the two, but this pass also runs on
    // hand-written IR fragments in tests. Without a scope there is nothing to
    // measure, and guessing a budget would be worse than not clamping.
    if (!scope)
      return loadLatency;
    scope->walk([&](ttg::LocalAllocOp alloc) {
      if (alloc.getSrc())
        if (Operation *def = alloc.getSrc().getDefiningOp())
          if (loadOpToIndLevel.count(def))
            return;
      bool liveInLoop =
          forOp->isAncestor(alloc) ||
          llvm::any_of(alloc->getUsers(),
                       [&](Operation *user) { return forOp->isAncestor(user); });
      if (!liveInLoop)
        return;
      auto ty = alloc.getType();
      int64_t elems = 1;
      for (int64_t d : ty.getShape())
        elems *= d;
      int64_t elemBits = ty.getElementType().getIntOrFloatBitWidth();
      staticBytes += elems * llvm::divideCeil(elemBits, 8);
    });

    // The walk above only finds local_allocs that already exist. The big one
    // usually does not: ReduceDataDuplication stages dot operands through
    // shared memory, and it runs long after this pass, so the buffer it will
    // create is invisible here. For SageAttention's _attn_fwd that is q --
    // 128x128 i8, 16 KB -- and missing it left the budget 16 KB too generous,
    // which is what let the pipeline overflow shared memory.
    //
    // Ask RDD's own predicate rather than guessing from the dots. RDD keys on
    // convert_layout, not on tt.dot, and the difference is not academic here:
    // a Turing bf16 or f32 dot falls back to FMA with no dot-operand
    // conversion at all, and an int4 dot never reaches shared memory, so
    // charging per dot operand would bill both of those for a buffer that is
    // never allocated.
    //
    // Only conversions defined outside this loop are counted, and only when a
    // user is directly inside it. Those stage once and stay live across every
    // iteration, so they compete with the pipeline for the budget.
    //
    // Two shapes are deliberately not modelled, both of which make this
    // estimate low rather than high: a candidate produced inside the loop, and
    // one that reaches the loop through an intervening op (a select, another
    // conversion) so that no direct user is in the loop. Telling those apart
    // from short-lived conversions the allocator will overlap needs provenance
    // and liveness analysis that does not belong in this pass; underestimating
    // only returns the loop to the pre-clamp behaviour, which is the same
    // trade-off documented for the rest of this function.
    //
    // RDD's transposed-operand reuse does not double-count here, and not
    // because of the loop check: its second conversion targets a linear
    // layout, which is exactly why RDD has to special-case it, so the helper
    // below rejects it and only the primary dot-operand conversion is billed.
    scope->walk([&](ttg::ConvertLayoutOp cvt) {
      if (forOp->isAncestor(cvt))
        return;
      std::optional<ttg::MemDescType> bufTy =
          getReduceDataDuplicationBufferType(cvt);
      if (!bufTy)
        return;
      if (!llvm::any_of(cvt->getUsers(), [&](Operation *user) {
            return forOp->isAncestor(user);
          }))
        return;
      staticBytes += memDescBytes(*bufTy);
    });

    int64_t budget = kSm75SharedMemoryBudget - staticBytes;
    unsigned clamped = loadLatency;
    // `>`, not `>=`: the allocator lets a kernel use the limit exactly, and
    // measured configurations do land on 65536 B legitimately. `>=` would
    // demote those for no gain -- it is not a safety margin, it only punishes
    // exact equality while an estimate that lands one byte short still passes.
    // Anything this estimate cannot see (alignment gaps between buffers,
    // conversion scratch) stays unmodelled rather than papered over here.
    while (clamped > 0 && bytesPerSlot * clamped > budget)
      --clamped;
    if (clamped != loadLatency)
      LDBG("sm75: clamping load latency " << loadLatency << " -> " << clamped
                                          << " (" << bytesPerSlot
                                          << "B/slot, " << staticBytes
                                          << "B static)");
    // Reducing the depth is routine -- it is what an autotuner would do. Losing
    // the pipeline outright is not: num_stages then has no effect at all, and
    // the only symptom is that raising it does not make the kernel faster. Say
    // so, or that stays invisible outside a debug build.
    if (clamped == 0 && loadLatency > 0)
      forOp->emitRemark()
          << "sm75: software pipelining disabled for this loop. One stage of "
             "multibuffering is estimated at "
          << bytesPerSlot
          << " B, and an estimated " << staticBytes
          << " B of Turing's 64KB/CTA shared memory is already committed to "
             "non-pipeline allocations, leaving no room for even a single "
             "stage. num_stages has no effect here. Both figures are "
             "estimates made before shared memory is allocated, so the real "
             "footprint may differ.";
    return clamped;
  }

  // Mirrors how Allocation.cpp sizes a shared buffer (the padded vs
  // getAllocationShapePerCTA split there), so the clamp's view of a
  // ReduceDataDuplication staging buffer matches the allocator's.
  //
  // Alignment is deliberately not modelled: the allocator aligns offsets and
  // reuses addresses across non-overlapping live ranges, so summing each
  // buffer rounded up to 16 B would over-count rather than add safety.
  static int64_t memDescBytes(ttg::MemDescType ty) {
    int64_t numElems;
    if (auto padded =
            dyn_cast<ttg::PaddedSharedEncodingAttr>(ty.getEncoding())) {
      numElems = padded.getPaddedSize(ttg::getShapePerCTA(ty));
    } else {
      numElems = 1;
      for (int64_t d : ttg::getAllocationShapePerCTA(ty))
        numElems *= d;
    }
    Type elemTy = ty.getElementType();
    int64_t elemBits =
        elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
    return numElems * elemBits / 8;
  }

  static int64_t tensorBytes(RankedTensorType ty) {
    Type elemTy = ty.getElementType();
    int64_t elemBits =
        elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
    return ty.getNumElements() * llvm::divideCeil(elemBits, 8);
  }

public:
  static bool canHaveSharedEncoding(tt::LoadOp op) {
    // If used by an user with DotOp encoding, all the uses must be compatible.
    bool incompatible = false;
    getSharedEncIfAllUsersAreDotEnc(op.getResult(), incompatible);
    return !incompatible;
  }

  static bool
  isPipeliningBeneficial(Operation *op, Operation *finalUser,
                         tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                         bool filterSmall, int computeCapability = 0) {
    // The >= 4-byte floor below is cp.async's minimum transfer size. The
    // sm75 synchronous copy path has no such hardware constraint, so on
    // Turing only a profitability floor of one 16-bit element remains:
    // pipelining strided fp16 loads (contiguity 1) is still worthwhile
    // because their long latency is exactly what overlapping hides.
    bool sm75SyncCopy = computeCapability == 75;
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (filterSmall &&
          (sm75SyncCopy
               ? getLoadContiguousBits(loadOp, axisInfoAnalysis) < 16
               : !canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis))) {
        LDBG("Load " << *loadOp << " is too small for pipelining");
        return false;
      }
    }
    if (isa<tt::DescriptorLoadLikeOpInterface>(op))
      return true;
    if (!canHaveSharedEncoding(cast<tt::LoadOp>(op))) {
      LDBG("Load " << *op << " cannot have shared encoding");
      return false;
    }

    ttg::SharedEncodingTrait localAllocEnc;
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return isa<ttg::LocalAllocOp>(user);
        })) {
      for (auto user : op->getUsers()) {
        auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
        if (!localAlloc)
          continue;
        auto enc = mlir::cast<ttg::SharedEncodingTrait>(
            localAlloc.getType().getEncoding());
        if (!localAllocEnc) {
          localAllocEnc = enc;
        }
        if (enc != localAllocEnc) {
          // If the load is used by a LocalAllocOp, all the users need to have
          // the same encoding.
          return false;
        }
      }
    }

    if (localAllocEnc) {
      auto registerTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      auto vecBytes = getCopyVecBytes(registerTy, localAllocEnc);
      // At least 4 consecutive bytes for cp.async; the sm75 sync copy only
      // needs a whole 16-bit element per st.shared.
      if (filterSmall && vecBytes < (sm75SyncCopy ? 2 : 4)) {
        return false;
      }
    }

    return true;
  }
};

class AssignMMALatencies {
public:
  AssignMMALatencies(scf::ForOp forOp, DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), opLatency(opLatency) {};

  void run() {
    DenseMap<Operation *, int> mmaSelfLatency;
    // Check if the load op (mma operand) is pipelineable.
    auto isLoadToBePipelined = [&](Operation *op) {
      return opLatency.count(op) && opLatency[op] > 0;
    };
    for (auto &op : forOp.getBody()->without_terminator()) {
      // If the acc can not be multibuffered, do not pipeline the uses of
      // the MMA to later stages.
      if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(&op)) {
        // Try to push out the wait by one stage even if the operands are not
        // pipelineable, but we know where the loads are scheduled, so we can
        // place the wait right before the loads.

        if (hasSyncDots(forOp)) {
          // Skip pipelining MMA in the loops where sync dots are used. This
          // is a dirty heuristic for performance drops in kernels where we
          // would rather want to have last iteration peeled instead of having a
          // full iteration of masked operations only to execute single wait.
          continue;
        }
        auto pipeHelper = ttng::MMAv5PipelineableOperandsHelper(
            mma, forOp, isLoadToBePipelined);
        if (pipeHelper.isPipelineable ||
            (pipeHelper.isOperandsStateDetermined &&
             !ttng::hasLoadsAfterMMA(mma, forOp))) {
          // MMA can be overlapped with itself
          mmaSelfLatency[mma] = 1;
          if (!ttng::requiresAccMultiBuffering(mma, forOp) ||
              (ttng::isAccMultibufferingPossible(mma, forOp) &&
               !getDisallowAccMultiBuffer(forOp))) {
            // MMA's users can be pushed to the next stage
            opLatency[&op] = 1;
          }
          // HACK: A pipelined MMA's latency should equal the number of buffers
          // for the accumulator, but when the user is in an `scf.if` in SWP,
          // the `scf.if` is pushed to the end of the loop rather than peeled
          // before the MMA op, requiring an extra buffer due to liverange
          // overlap. WS does not have this problem because the MMA is placed in
          // a different partition than the MMA, so we can correctly set the
          // latency.
          if (isWarpSpecialized(forOp)) {
            if (ttng::hasAccReadModifyWrite(mma, forOp))
              opLatency.erase(&op); // can't pipeline the MMA
            else
              opLatency[&op] += 1;
            // If all inputs to the MMA are warp specialized, set the self
            // latency to 0 since the MMA won't need to wait on itself.
            auto cantWarpSpec = [](Operation *op) { return isa<LoadOp>(op); };
            auto warpSpecHelper = ttng::MMAv5PipelineableOperandsHelper(
                mma, forOp, [&](Operation *op) {
                  return isLoadToBePipelined(op) && !cantWarpSpec(op);
                });
            if (warpSpecHelper.isPipelineable ||
                (warpSpecHelper.isOperandsStateDetermined &&
                 llvm::none_of(warpSpecHelper.unpipelineableOperandDefs,
                               cantWarpSpec)))
              mmaSelfLatency[mma] = 0;
          }
        }
      }
    }
    serializeSelfLatencies(forOp->getParentOfType<ModuleOp>(), mmaSelfLatency);
  }

private:
  scf::ForOp forOp;
  DenseMap<Operation *, int> &opLatency;

  bool hasSyncDots(scf::ForOp forOp) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isa<mlir::triton::DotOp>(op))
        return true;
    }
    return false;
  }

  bool isWarpSpecialized(scf::ForOp forOp) {
    scf::ForOp current = forOp;
    do {
      if (current->hasAttr(kWarpSpecializeAttrName)) {
        return true;
      }
      current = current->getParentOfType<scf::ForOp>();
    } while (current);
    return false;
  };
};

// Discover operations that should become async and assign latencies to them
// based on the numStages value provided by the user.
//
// Look for load ops that directly or indirectly feed into dot ops. Based on the
// requested number of stages assign the latencies in a way that cover all the
// stages with the sum of latencies in the chain from the first load to the
// final dot op.
void assignLatencies(ModuleOp moduleOp, int defaultNumStages) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) {
    // Bail out for loops with num_stage <= 1.
    if (preCondition(forOp) &&
        getNumStagesOrDefault(forOp, defaultNumStages) > 1)
      loops.push_back(forOp);
  });
  if (loops.empty())
    return;

  // The module may target another vendor or, in lit tests, carry no target
  // attribute; both would assert in getNVIDIAComputeCapability. Treat them
  // as "not sm75": the latency clamp below only applies to Turing.
  int computeCapability = 0;
  auto targetAttr = moduleOp->getAttrOfType<StringAttr>(AttrTargetName);
  if (targetAttr && targetAttr.getValue().starts_with("cuda:"))
    computeCapability = getNVIDIAComputeCapability(moduleOp);

  DenseMap<Operation *, int> opLatency;
  for (auto forOp : loops) {
    if (hasLatenciesAssigned(forOp)) {
      assignUserProvidedLatencies(forOp, opLatency);
      continue;
    }
    int numStages = getNumStagesOrDefault(forOp, defaultNumStages);
    AssignLoadLatencies(forOp, numStages, opLatency, computeCapability).run();
    AssignMMALatencies(forOp, opLatency).run();
  }
  serializeLatencies(moduleOp, opLatency);
}

} // namespace

// Create a map from load ops to their indirection level and the
// final use of the load op (another load op, or a dot op).
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
llvm::MapVector<Operation *, std::pair<int, Operation *>>
loadOpsToIndirectionLevel(scf::ForOp forOp, bool pipelineWithoutDot,
                          tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                          int numStages, bool filterSmall,
                          int computeCapability) {
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel;
  DenseSet<Operation *> seen;
  DenseSet<Operation *> excluded;

  std::function<void(Operation *, Operation *, int)> dfs =
      [&](Operation *op, Operation *finalUser, int distance) {
        if (!seen.insert(op).second || excluded.count(op))
          return;
        if (isa<tt::LoadOp, tt::DescriptorLoadLikeOpInterface>(op)) {
          if (!AssignLoadLatencies::isPipeliningBeneficial(
                  op, finalUser, axisInfoAnalysis, filterSmall,
                  computeCapability))
            return;
          if (loadOpToIndLevel.count(op)) {
            int level = loadOpToIndLevel[op].first;
            if (level != distance) {
              // If we have multiple uses at different distances, we don't
              // know which one to pick.
              LDBG("Load " << *op
                           << " has multiple uses at different distances:"
                           << level << " and " << distance);
              loadOpToIndLevel.erase(op);
              excluded.insert(op);
              return;
            }
          } else {
            LDBG("Load " << *op << " considered for pipelining with distance "
                         << distance);
            loadOpToIndLevel[op] = {distance, finalUser};
          }
          finalUser = op;
          distance++;
        }
        for (Value operand : getNestedOperands(op)) {
          if (isa<mlir::triton::DotOpInterface>(op)) {
            // Heuristic: only pipeline A and B operands of the dot op.
            if (operand == op->getOperand(2))
              continue;
          }
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, finalUser, distance);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // Arbitrary heuristic. TMEMStoreOp is included to keep logic consistent
    // with legacy code when we weren't hoisting tmem allocas.
    if (!isa<mlir::triton::DotOpInterface, ttng::TMEMStoreOp>(op))
      continue;
    seen.clear();
    dfs(&op, &op, 0);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (pipelineWithoutDot) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp, tt::DescriptorLoadLikeOpInterface>(op))
        dfs(&op, &op, 0);
    }
  }

  // We assume loads with different dist are assigned to different stages.
  // If numStages is 2, we will have no stage available for indirect loads
  // with dist >= 1. In general, when dist is equal to numStages - 1, we
  // should not pipeline it.
  for (auto iter = loadOpToIndLevel.begin(); iter != loadOpToIndLevel.end();) {
    if (iter->second.first >= numStages - 1)
      iter = loadOpToIndLevel.erase(iter);
    else
      ++iter;
  }

  return loadOpToIndLevel;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TRITONGPUASSIGNLATENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct AssignLatencies
    : public impl::TritonGPUAssignLatenciesBase<AssignLatencies> {
  using TritonGPUAssignLatenciesBase::TritonGPUAssignLatenciesBase;

  void runOnOperation() override { assignLatencies(getOperation(), numStages); }
};

} // namespace mlir::triton::gpu
