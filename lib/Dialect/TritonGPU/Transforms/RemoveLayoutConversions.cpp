#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUREMOVELAYOUTCONVERSIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-remove-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

static Attribute getTensorEncoding(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return tensorType ? tensorType.getEncoding() : Attribute();
}

static bool isBlockedOrBlockedSliceEncoding(Attribute encoding) {
  if (isa_and_nonnull<BlockedEncodingAttr>(encoding))
    return true;
  auto sliceEncoding = dyn_cast_or_null<SliceEncodingAttr>(encoding);
  return sliceEncoding && isa<BlockedEncodingAttr>(sliceEncoding.getParent());
}

static bool isSm75Target(FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return false;
  auto targetAttr = moduleOp->getAttrOfType<StringAttr>(AttrTargetName);
  return targetAttr && targetAttr.getValue().starts_with("cuda:") &&
         getNVIDIAComputeCapability(moduleOp) == 75;
}

// Synthesize the blocked parent encoding whose slice along `op.getAxis()`
// distributes the source tensor identically to `encoding` (inserting 1 element,
// 1 lane, 1 warp, and 1 CTA along the expanded axis as the slowest dimension).
static Attribute getExpandDimsParentEncoding(ExpandDimsOp op,
                                             Attribute encoding) {
  auto blocked = dyn_cast<BlockedEncodingAttr>(encoding);
  if (!blocked)
    return {};
  unsigned axis = op.getAxis();
  unsigned srcRank = blocked.getOrder().size();
  if (axis > srcRank)
    return {};
  auto insertOne = [&](ArrayRef<unsigned> values) {
    SmallVector<unsigned> result(values);
    result.insert(result.begin() + axis, 1);
    return result;
  };
  auto insertSlowest = [&](ArrayRef<unsigned> order) {
    SmallVector<unsigned> result;
    result.reserve(order.size() + 1);
    for (unsigned d : order)
      result.push_back(d >= axis ? d + 1 : d);
    result.push_back(axis);
    return result;
  };

  MLIRContext *ctx = op.getContext();
  LinearLayout cgaLL = blocked.getCGALayout().getLinearLayout();
  auto bases = cgaLL.getBases();
  for (auto &kv : bases) {
    for (auto &vec : kv.second)
      vec.insert(vec.begin() + axis, 0);
  }
  auto outDims = llvm::to_vector(cgaLL.getOutDims());
  outDims.insert(outDims.begin() + axis, {StringAttr(), 1});
  for (unsigned i = 0; i < outDims.size(); ++i)
    outDims[i].first = StringAttr::get(ctx, "dim" + std::to_string(i));
  auto parentCga = CGAEncodingAttr::get(
      ctx, LinearLayout(std::move(bases), outDims, /*requireSurjective=*/false));
  return BlockedEncodingAttr::get(ctx, insertOne(blocked.getSizePerThread()),
                                  insertOne(blocked.getThreadsPerWarp()),
                                  insertOne(blocked.getWarpsPerCTA()),
                                  insertSlowest(blocked.getOrder()), parentCga);
}

static bool isEquivalentEncoding(Value value, Attribute a, Attribute b) {
  if (a == b)
    return true;
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !a || !b || !isa<LayoutEncodingTrait>(a) ||
      !isa<LayoutEncodingTrait>(b))
    return false;
  return areLayoutsEquivalent(type.getShape(), cast<LayoutEncodingTrait>(a),
                              cast<LayoutEncodingTrait>(b));
}

static bool isExpensiveMathOp(Operation *op);
static int64_t getByteCount(Value result, int64_t minElementCount = 0,
                            int64_t minBitWidth = 0);

static int64_t getScopeLoopFrequencyWeight(Operation *enclosing) {
  constexpr int64_t kLoopIterationWeight = 8;
  constexpr int kMaxWeightedLoopDepth = 3;
  int depth = 0;
  for (Operation *op = enclosing; op && depth < kMaxWeightedLoopDepth;
       op = op->getParentOp())
    if (isa<scf::ForOp, scf::WhileOp>(op))
      ++depth;
  int64_t weight = 1;
  for (int i = 0; i < depth; ++i)
    weight *= kLoopIterationWeight;
  return weight;
}

static int64_t getOpLoopFrequencyWeight(Operation *op) {
  return getScopeLoopFrequencyWeight(op->getParentOp());
}

static int64_t getConvertUnitCost(Value value) {
  int64_t cost = 32 * getByteCount(value, 32, 32);
  return cost == 0 ? 1 : cost;
}

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// The current algorithm works by analyzing the IR and doing a one-shot rewrite
// based on the analysis. The algorithm is as follows.
//
// 1. Find all the anchor ops. These are ops that have a layout we want to
//    preserve.
//
// 2. For each anchor, propagate its layout to all its descendants.
//    An op can have multiple ancestors that are anchors, so at this stage an op
//    may have multiple layouts associated with it.
//
// 3. Resolve conflicts by deciding which of the multiple layouts the op should
//    keep, inserting convert-layout ops to resolve conflicts.  After this
//    stage, each value has only one layout associated with it.
//
// 4. Rewrite the IR by walking the function in dominance order. Since we
//    assume the IR is structured we just need to process the regions in the
//    correct order. For each op, rewrite it using the layout decided by the
//    analysis phase.
class LayoutPropagation {
public:
  // Structure to keep track of the layout associated to a value.
  struct LayoutInfo {
    LayoutInfo(Attribute encoding) { encodings.insert(encoding); }
    LayoutInfo() {}
    llvm::SmallSetVector<Attribute, 8> encodings;
    // Encodings that arrived only via synthesized expand_dims parents.
    llvm::SmallDenseSet<Attribute, 4> derivedEncodings;
  };
  LayoutPropagation(FuncOp F)
      : funcOp(F), enableSm75LoopComponent(isSm75Target(F)) {}
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
  // Add layouts given in `Info` to the uses of `value`.
  SmallVector<Value> propagateToUsers(Value value, LayoutInfo &info);
  // Set the encoding to all the values and fill out the values with new layout
  // in `changed`.
  void setEncoding(ValueRange values, LayoutInfo &info,
                   SmallVector<Value> &changed, Operation *op);
  // Resolve cases where a value has multiple layouts associated to it.
  void resolveConflicts();
  // Component-level resolution for scf.for-bound blocked components.
  void resolveConflictsByComponent(
      const llvm::MapVector<Value, SmallVector<Attribute>> &candidates);
  // Rewrite the IR for the full module.
  void rewrite();
  // Rewrite the IR for a region.
  void rewriteRegion(Region &R);
  // Rewrite an op based on the layout picked by the analysis.
  Operation *rewriteOp(Operation *op);
  // Rewrite a for op based on the layout picked by the analysis.
  Operation *rewriteForOp(scf::ForOp forOp);
  Operation *rewriteWhileOp(scf::WhileOp whileOp);
  Operation *rewriteIfOp(scf::IfOp ifOp);
  void rewriteYieldOp(scf::YieldOp yieldOp);
  void rewriteConditionOp(scf::ConditionOp conditionOp);
  void rewriteReduceToScalar(Operation *reduceOp);
  void rewriteAssertOp(AssertOp assertOp);
  Operation *cloneElementwise(OpBuilder &rewriter, Operation *op,
                              Attribute encoding);
  // Map the original value to the rewritten one.
  void map(Value old, Value newV);
  // Return the mapped value in the given encoding. This will insert a convert
  // if the encoding is different than the encoding decided at resolve time.
  Value getValueAs(Value value, Attribute encoding);
  // Return the original value mapped to the new desired encoding.
  Value getRewrittenValue(Value value);
  // Dump the current stage of layout information.
  void dump();

private:
  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  // map of the values rewrite based on their encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rewriteMapping;
  SetVector<Operation *> opToDelete;
  FuncOp funcOp;
  bool enableSm75LoopComponent = false;
};

class LayoutRematerialization {
public:
  LayoutRematerialization(FuncOp F) : funcOp(F) {}

  // Map the original value to the remat'ed one.
  void addRematValue(Value old, Attribute encoding, Value newV);
  // Get the remat'ed value in the given encoding, if one already exists and
  // is different then the layout conversion root.
  Value getRematValue(Value value, Attribute encoding) const {
    return rematMapping.lookup({value, encoding});
  }

  void cleanup();
  bool backwardRematerialization();
  void backwardRematerialization(ConvertLayoutOp convertOp);
  // TODO: Merge the three hoistConvert*(); functions as they are duplicate code
  void hoistConvertDotOperand();
  void hoistConvertDotOperand(ConvertLayoutOp convertOp);
  void hoistConvertOnTopOfExtOrBroadcast();
  void hoistConvertOnTopOfExtOrBroadcast(ConvertLayoutOp convertOp);
  void hoistConvertIntoConditionals();
  void hoistConvertIntoConditionals(ConvertLayoutOp convertOp);
  void rewriteSlice(SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
                    ConvertLayoutOp convertOp, IRMapping &mapping);
  void rewriteSlice(SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
                    ConvertLayoutOp convertOp);

  LogicalResult
  getConvertBackwardSlice(OpOperand &root, Attribute rootEncoding,
                          SetVector<Value> &slice,
                          DenseMap<Value, Attribute> &layout,
                          std::function<bool(Operation *)> stopPropagation);

  LogicalResult getRematerializableSlice(
      OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
      DenseMap<Value, Attribute> &layout,
      std::function<bool(Operation *)> stopPropagation = nullptr);

private:
  void updateRematMapping(SmallVector<std::tuple<Value, Value>> &values);
  // Existing tuples of (value, layout) that needs to be updated when recreating
  // scf ops. This prevents keeping track of Values that have been delete when
  // rewriting slices.
  DenseMap<Value, Attribute> mappedValues;
  // map of the values remat based on encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rematMapping;
  // DenseMap<std::pair<Operation*, Attribute>, Operation*>
  SetVector<Operation *> opToDelete;
  FuncOp funcOp;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
};

void LayoutRematerialization::addRematValue(Value old, Attribute encoding,
                                            Value newV) {
  LDBG("addRematValue " << old << " encoding " << encoding << " " << newV);
  rematMapping[{old, encoding}] = newV;
  mappedValues[old] = encoding;
}

// Remove unneeded values now that we are done with the rematMapping.
void LayoutRematerialization::cleanup() {
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
}

// Return true if the op is an op with a layout we don't want to change. We will
// propagate the layout starting from anchor ops.
bool isLayoutAnchor(Operation *op) {
  if (isa<DescriptorOpInterface>(op))
    return true;
  if (isa<LoadOp, StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<DotOp, DotScaledOp, nvidia_gpu::WarpGroupDotOp, AtomicRMWOp,
          AtomicCASOp, triton::nvidia_gpu::TMEMLoadOp>(op))
    return true;
  if (auto gatherOp = dyn_cast<GatherOp>(op))
    return gatherOp.getEfficientLayout();

  // Heuristic: Mark permuting reshape as a layout anchor.  Its dst can be
  // anything, so it stops forward-propagation of layouts.  We rely on the
  // backwards pass to fix it up if necessary.  (If we didn't do this, then
  // anything following the reshape won't be covered by the forward pass at
  // all.)
  if (auto reshape = dyn_cast<ReshapeOp>(op))
    return reshape.getAllowReorder();

  return false;
}

void LayoutPropagation::initAnchorLayout() {
  auto addAnchor = [&](Value v) {
    if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
      layouts.insert({v, LayoutInfo(tensorType.getEncoding())});
    }
  };

  // Consider function args as anchors.  This makes it easier to write tests --
  // you can pass a tensor with an encoding as an arg, instead of explicitly
  // calling tt.load.
  for (auto arg : funcOp.getArguments()) {
    addAnchor(arg);
  }

  funcOp.walk([&](Operation *op) {
    if (isLayoutAnchor(op)) {
      for (auto result : op->getResults()) {
        addAnchor(result);
      }
    }
  });
}

void LayoutPropagation::setEncoding(ValueRange values, LayoutInfo &info,
                                    SmallVector<Value> &changed,
                                    Operation *op) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool hasChanged = false;
    for (auto encoding : info.encodings) {
      Attribute dstEncoding;
      bool derived = info.derivedEncodings.contains(encoding);
      if (isa<ConvertLayoutOp>(op)) {
        // Try to remove the convert by making the dst encoding match the source
        // encoding.
        dstEncoding = encoding;
      } else {
        dstEncoding = inferDstEncoding(op, encoding);
        // inferDstEncoding cannot expand a blocked (non-slice) source. Directly
        // inside an scf.for body on SM75, where the convert this avoids would
        // run every iteration, synthesize the blocked parent the source is a
        // slice of.
        if (auto expandOp = dyn_cast<ExpandDimsOp>(op);
            enableSm75LoopComponent && expandOp && !dstEncoding &&
            isa<scf::ForOp>(op->getParentOp())) {
          dstEncoding = getExpandDimsParentEncoding(expandOp, encoding);
          derived = true;
        }
      }
      if (dstEncoding) {
        auto &layoutInfo = layouts[value];
        // An encoding is derived only while every path to it is derived.
        if (layoutInfo.encodings.insert(dstEncoding)) {
          hasChanged = true;
          if (derived)
            layoutInfo.derivedEncodings.insert(dstEncoding);
        } else if (!derived) {
          hasChanged |= layoutInfo.derivedEncodings.erase(dstEncoding);
        }
      }
    }
    if (hasChanged)
      changed.push_back(value);
  }
}

SmallVector<Value> LayoutPropagation::propagateToUsers(Value value,
                                                       LayoutInfo &info) {
  SmallVector<Value> changed;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setEncoding({arg, result}, info, changed, user);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
      setEncoding({arg}, info, changed, user);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto parent = yieldOp->getParentOp();
      SmallVector<Value> valuesToPropagate;
      if (isa<scf::ForOp, scf::IfOp>(parent))
        valuesToPropagate.push_back(parent->getResult(use.getOperandNumber()));
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        valuesToPropagate.push_back(
            forOp.getRegionIterArg(use.getOperandNumber()));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
        valuesToPropagate.push_back(
            whileOp.getBeforeArguments()[use.getOperandNumber()]);
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        setEncoding(valuesToPropagate, info, changed, user);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      // Skip arg 0 as it is the condition.
      if (use.getOperandNumber() == 0)
        continue;
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      unsigned argIndex = use.getOperandNumber() - 1;
      Value afterArg = whileOp.getAfterArguments()[argIndex];
      Value result = whileOp->getResult(argIndex);
      setEncoding({afterArg, result}, info, changed, user);
      continue;
    }
    if (auto dotWaitOp = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(user)) {
      unsigned opIndex = use.getOperandNumber();
      Value result = dotWaitOp->getResult(opIndex);
      setEncoding(result, info, changed, user);
      continue;
    }
    if (auto gatherOp = dyn_cast<GatherOp>(user)) {
      // Propagate the layout through the indices only, and if the layout does
      // not have an efficient layout set.
      if (!gatherOp.getEfficientLayout() &&
          &use == &gatherOp.getIndicesMutable()) {
        setEncoding(gatherOp.getResult(), info, changed, user);
        continue;
      }
    }
    if (isa<AtomicRMWOp, AtomicCASOp>(user)) {
      // Atomic ops are layout anchors whose layout must not change, even though
      // they carry SameOperandsAndResultEncoding. Stop propagating operand
      // layouts across them so stale candidates cannot leak to downstream ops.
      continue;
    }
    if (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
        user->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            ConvertLayoutOp>(user)) {
      setEncoding(user->getResults(), info, changed, user);
      continue;
    }
  }
  return changed;
}

void LayoutPropagation::propagateLayout() {
  SmallVector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed = propagateToUsers(currentValue, info);

    LLVM_DEBUG({
      DBGS() << "propagateLayout considering " << currentValue << ", which has "
             << info.encodings.size() << " candidate encoding(s):\n";
      for (Attribute encoding : info.encodings)
        DBGS() << "  " << encoding << "\n";
      DBGS() << "changed: " << changed.size() << "\n";
    });

    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

static void
collectUseTargets(OpOperand &use, Attribute enc,
                  SmallVectorImpl<std::pair<Value, Attribute>> &targets) {
  Operation *user = use.getOwner();

  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    targets.push_back({forOp.getTiedLoopRegionIterArg(&use), enc});
    targets.push_back({forOp.getTiedLoopResult(&use), enc});
    return;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
    targets.push_back(
        {whileOp.getBeforeArguments()[use.getOperandNumber()], enc});
    return;
  }
  if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
    auto parent = yieldOp->getParentOp();
    if (isa<scf::ForOp, scf::IfOp>(parent))
      targets.push_back({parent->getResult(use.getOperandNumber()), enc});
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      targets.push_back({forOp.getRegionIterArg(use.getOperandNumber()), enc});
    if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
      targets.push_back(
          {whileOp.getBeforeArguments()[use.getOperandNumber()], enc});
    return;
  }
  if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
    if (use.getOperandNumber() == 0)
      return;
    auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
    unsigned argIndex = use.getOperandNumber() - 1;
    targets.push_back({whileOp.getAfterArguments()[argIndex], enc});
    targets.push_back({whileOp->getResult(argIndex), enc});
    return;
  }
  if (isa<nvidia_gpu::WarpGroupDotWaitOp>(user)) {
    Value result = user->getResult(use.getOperandNumber());
    if (getTensorEncoding(result))
      targets.push_back({result, enc});
    return;
  }
  if (auto gatherOp = dyn_cast<GatherOp>(user)) {
    if (!gatherOp.getEfficientLayout() &&
        &use == &gatherOp.getIndicesMutable()) {
      Attribute dstEnc = inferDstEncoding(user, enc);
      if (dstEnc)
        targets.push_back({gatherOp.getResult(), dstEnc});
      return;
    }
  }
  if (isa<AtomicRMWOp, AtomicCASOp>(user))
    return;
  if (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      user->hasTrait<OpTrait::Elementwise>() ||
      isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
          ConvertLayoutOp>(user)) {
    for (Value result : user->getResults()) {
      if (!getTensorEncoding(result))
        continue;
      Attribute dstEnc =
          isa<ConvertLayoutOp>(user) ? enc : inferDstEncoding(user, enc);
      if (dstEnc)
        targets.push_back({result, dstEnc});
    }
  }
}

static bool isLayoutTransparentProducer(Operation *op) {
  if (isLayoutAnchor(op) || op->getNumRegions() != 0 || isExpensiveMathOp(op))
    return false;
  if (isa<BroadcastOp, ExpandDimsOp, ConvertLayoutOp>(op))
    return true;
  return (op->hasTrait<OpTrait::Elementwise>() ||
          op->hasTrait<OpTrait::SameOperandsAndResultEncoding>()) &&
         isMemoryEffectFree(op);
}

static constexpr int kMaxRematDepth = 6;

static int64_t
getRematConversionCost(Value value, Attribute required, int depth,
                       function_ref<Attribute(Value)> encodingOf) {
  if (depth == kMaxRematDepth
          ? isEquivalentEncoding(value, encodingOf(value), required)
          : encodingOf(value) == required)
    return 0;
  int64_t full = getConvertUnitCost(value);
  Operation *def = value.getDefiningOp();
  if (!def || depth <= 0)
    return full;
  if (isa<arith::ConstantOp, SplatOp, MakeRangeOp>(def))
    return 0;
  if (!isLayoutTransparentProducer(def))
    return full;
  Attribute srcRequired =
      isa<ConvertLayoutOp>(def) ? required : inferSrcEncoding(def, required);
  if (!srcRequired)
    return full;
  int64_t sum = 0;
  for (Value operand : def->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    sum += getRematConversionCost(operand, srcRequired, depth - 1, encodingOf);
    if (sum >= full)
      return full;
  }
  return sum;
}

void LayoutPropagation::resolveConflictsByComponent(
    const llvm::MapVector<Value, SmallVector<Attribute>> &candidates) {
  llvm::EquivalenceClasses<Value> tied;
  auto unite = [&](Value a, Value b) {
    if (candidates.count(a) && candidates.count(b))
      tied.unionSets(a, b);
  };
  for (auto &it : candidates)
    tied.insert(it.first);
  for (auto &it : candidates) {
    Value v = it.first;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!forOp || arg.getArgNumber() < forOp.getNumInductionVars())
        continue;
      unsigned i = arg.getArgNumber() - forOp.getNumInductionVars();
      unite(v, forOp.getResult(i));
      unite(v, forOp.getInitArgs()[i]);
      unite(v, forOp.getYieldedValues()[i]);
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      unsigned i = cast<OpResult>(v).getResultNumber();
      unite(v, forOp.getRegionIterArg(i));
    } else if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      unsigned i = cast<OpResult>(v).getResultNumber();
      unite(v, ifOp.thenYield().getOperand(i));
      if (!ifOp.getElseRegion().empty())
        unite(v, ifOp.elseYield().getOperand(i));
    } else if ((def->hasTrait<OpTrait::Elementwise>() ||
                def->hasTrait<OpTrait::SameOperandsAndResultEncoding>()) &&
               def->getNumRegions() == 0 && !isLayoutAnchor(def)) {
      for (Value operand : def->getOperands())
        if (isa<RankedTensorType>(operand.getType()))
          unite(v, operand);
    }
  }

  llvm::MapVector<Value, SmallVector<Value>> components;
  for (auto &it : candidates)
    components[tied.getLeaderValue(it.first)].push_back(it.first);

  auto resolvedEncoding = [&](Value v) -> Attribute {
    auto it = layouts.find(v);
    if (it != layouts.end() && !it->second.encodings.empty())
      return *it->second.encodings.begin();
    return getTensorEncoding(v);
  };

  struct AssignmentCost {
    int64_t total = 0;
    int64_t inLoop = 0;
  };
  // Stage 0 moves scf.for-bound components when in-loop convert cost strictly
  // drops without raising total cost. Stage 1 lets direct external neighbors of
  // Stage-0 moved components (e.g., a post-loop tt.reduce consuming the moved
  // accumulator) follow when that strictly lowers total cost; Stage-0 moved
  // components are not reconsidered in Stage 1.
  DenseSet<Value> movedInStage0;
  DenseSet<Value> movedRoots;
  for (int stage = 0; stage < 2; ++stage) {
    if (stage == 1 && movedInStage0.empty())
      break;
    for (auto &[root, members] : components) {
      if (stage == 1 && movedRoots.contains(root))
        continue;
      DenseSet<Value> inComponent(members.begin(), members.end());
      SmallVector<Attribute> shared(candidates.lookup(members.front()));
      bool skip = false;
      bool hasForCarriedValue = false;
      for (Value m : members) {
        const SmallVector<Attribute> &cands = candidates.lookup(m);
        llvm::erase_if(shared, [&](Attribute enc) {
          return !llvm::is_contained(cands, enc);
        });
        if (Operation *def = m.getDefiningOp()) {
          skip |= isLayoutAnchor(def) || isa<LoadOp, scf::WhileOp>(def);
          hasForCarriedValue |= isa<scf::ForOp>(def);
        } else {
          auto arg = cast<BlockArgument>(m);
          bool isForArg = isa<scf::ForOp>(arg.getOwner()->getParentOp());
          skip |= !isForArg;
          hasForCarriedValue |= isForArg;
        }
      }
      if (skip || shared.size() < 2)
        continue;

      if (stage == 0 && !hasForCarriedValue)
        continue;

      DenseMap<Value, Attribute> resolved;
      bool adjacentToStage0 = false;
      for (Value m : members) {
        resolved[m] = resolvedEncoding(m);
        if (stage == 1) {
          if (Operation *def = m.getDefiningOp())
            for (Value operand : def->getOperands())
              adjacentToStage0 |= !inComponent.contains(operand) &&
                                  movedInStage0.contains(operand);
          for (Operation *user : m.getUsers())
            for (Value result : user->getResults())
              adjacentToStage0 |= !inComponent.contains(result) &&
                                  movedInStage0.contains(result);
        }
      }
      if (stage == 1 && !adjacentToStage0)
        continue;

      auto priceAssignment = [&](function_ref<Attribute(Value)> assignment) {
        AssignmentCost result;
        auto charge = [&](int64_t weight, int64_t c) {
          result.total += weight * c;
          if (weight > 1)
            result.inLoop += weight * c;
        };
        auto assignedEncoding = [&](Value v) {
          return inComponent.contains(v) ? assignment(v) : resolvedEncoding(v);
        };
        auto operandCost = [&](Value operand, Attribute req, Operation *at) {
          charge(getOpLoopFrequencyWeight(at),
                 getRematConversionCost(operand, req, kMaxRematDepth,
                                        assignedEncoding));
        };
        for (Value m : members) {
          Attribute enc = assignment(m);
          if (auto arg = dyn_cast<BlockArgument>(m)) {
            auto forOp = cast<scf::ForOp>(arg.getOwner()->getParentOp());
            unsigned i = arg.getArgNumber() - forOp.getNumInductionVars();
            operandCost(forOp.getInitArgs()[i], enc, forOp);
            Value yielded = forOp.getYieldedValues()[i];
            operandCost(yielded, enc,
                        yielded.getParentBlock()->getTerminator());
          } else if (Operation *def = m.getDefiningOp()) {
            if (auto forOp = dyn_cast<scf::ForOp>(def)) {
              unsigned i = cast<OpResult>(m).getResultNumber();
              Value arg = forOp.getRegionIterArg(i);
              if (!isEquivalentEncoding(m, assignedEncoding(arg), enc))
                charge(getOpLoopFrequencyWeight(forOp), getConvertUnitCost(m));
            } else if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
              unsigned i = cast<OpResult>(m).getResultNumber();
              operandCost(ifOp.thenYield().getOperand(i), enc,
                          ifOp.thenYield());
              if (!ifOp.getElseRegion().empty())
                operandCost(ifOp.elseYield().getOperand(i), enc,
                            ifOp.elseYield());
            } else {
              Attribute srcReq =
                  isa<ConvertLayoutOp>(def) ? enc : inferSrcEncoding(def, enc);
              for (Value operand : def->getOperands()) {
                if (!isa<RankedTensorType>(operand.getType()))
                  continue;
                if (!srcReq)
                  charge(getOpLoopFrequencyWeight(def),
                         getConvertUnitCost(operand));
                else
                  operandCost(operand, srcReq, def);
              }
            }
          }
          for (OpOperand &use : m.getUses()) {
            Operation *user = use.getOwner();
            int64_t weight = getOpLoopFrequencyWeight(user);
            SmallVector<std::pair<Value, Attribute>> targets;
            collectUseTargets(use, enc, targets);
            if (targets.empty()) {
              charge(weight,
                     getRematConversionCost(m, getTensorEncoding(m),
                                            kMaxRematDepth, assignedEncoding));
              continue;
            }
            for (auto &[target, dstEnc] : targets) {
              if (inComponent.contains(target) ||
                  isEquivalentEncoding(target, resolvedEncoding(target),
                                       dstEnc))
                continue;
              int64_t c = getConvertUnitCost(m);
              auto it = candidates.find(target);
              if (it != candidates.end() &&
                  llvm::is_contained(it->second, dstEnc))
                c = std::min(c, getConvertUnitCost(target));
              charge(weight, c);
            }
          }
        }
        return result;
      };

      auto bestCost =
          priceAssignment([&](Value v) { return resolved.lookup(v); });
      Attribute best;
      for (Attribute enc : shared) {
        auto cost = priceAssignment([&](Value) { return enc; });
        bool better = stage == 1 ? cost.total < bestCost.total
                                 : cost.inLoop < bestCost.inLoop &&
                                       cost.total <= bestCost.total;
        if (better) {
          bestCost = cost;
          best = enc;
        }
      }
      if (!best)
        continue;
      LDBG("stage " << stage << " component of " << members.size()
                    << " values moved to " << best << ", convert cost "
                    << bestCost.total << " (inLoop " << bestCost.inLoop << ")");
      for (Value m : members) {
        LayoutInfo &info = layouts[m];
        info.encodings.clear();
        info.encodings.insert(best);
        if (stage == 0)
          movedInStage0.insert(m);
      }
      if (stage == 0)
        movedRoots.insert(root);
    }
  }
}

void LayoutPropagation::resolveConflicts() {
  if (!enableSm75LoopComponent) {
    for (auto &it : layouts) {
      Operation *op = it.first.getDefiningOp();
      LayoutInfo &info = it.second;
      if (info.encodings.size() <= 1)
        continue;
      // Hacky resolve, prefer block encoding.
      // TODO: add a proper heuristic.
      Attribute encoding = *info.encodings.begin();
      bool isLoadOrStore =
          op && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op);
      for (auto &e : info.encodings) {
        if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
            (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
          encoding = e;
          break;
        }
      }
      info.encodings.clear();
      info.encodings.insert(encoding);
    }
    return;
  }
  // A value reached only through synthesized expand_dims parents keeps its
  // original encoding as an option so per-value resolution stays baseline
  // unless component resolution proves a synthesized parent is cheaper.
  for (auto &it : layouts) {
    LayoutInfo &info = it.second;
    if (info.derivedEncodings.empty() ||
        !llvm::all_of(info.encodings, [&](Attribute enc) {
          return info.derivedEncodings.contains(enc);
        }))
      continue;
    if (Attribute original = getTensorEncoding(it.first))
      info.encodings.insert(original);
  }

  // Capture blocked/blocked-slice candidate sets before per-value resolution
  // collapses them.
  llvm::MapVector<Value, SmallVector<Attribute>> componentCandidates;
  for (auto &it : layouts) {
    const LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1 ||
        !isa<RankedTensorType>(it.first.getType()))
      continue;
    bool blockedOnly = llvm::all_of(info.encodings, [](Attribute enc) {
      return isBlockedOrBlockedSliceEncoding(enc);
    });
    if (blockedOnly)
      componentCandidates[it.first] = SmallVector<Attribute>(
          info.encodings.begin(), info.encodings.end());
  }

  for (auto &it : layouts) {
    Operation *op = it.first.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1)
      continue;
    // Leave derived (synthesized-parent) encodings to component resolution.
    SmallVector<Attribute> legacy;
    for (Attribute e : info.encodings)
      if (!info.derivedEncodings.contains(e))
        legacy.push_back(e);
    if (legacy.empty())
      legacy.assign(info.encodings.begin(), info.encodings.end());
    // Hacky resolve, prefer block encoding.
    // TODO: add a proper heuristic.
    Attribute encoding = legacy.front();
    bool isLoadOrStore =
        op && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op);
    for (Attribute e : legacy) {
      if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
          (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
        encoding = e;
        break;
      }
    }
    info.encodings.clear();
    info.encodings.insert(encoding);
  }

  if (!componentCandidates.empty())
    resolveConflictsByComponent(componentCandidates);
}

void LayoutPropagation::dump() {
  for (auto it : layouts) {
    llvm::errs() << "Value: ";
    OpPrintingFlags flags;
    flags.skipRegions();
    it.first.print(llvm::errs(), flags);
    llvm::errs() << " \n encoding:\n";
    for (auto encoding : it.second.encodings) {
      encoding.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "--\n";
  }
}

void LayoutPropagation::rewrite() { rewriteRegion(funcOp->getRegion(0)); }

bool reduceToScalar(Operation *op) {
  // For reductions returning a scalar we can change the src encoding without
  // affecting the output.
  return isa<ReduceOp>(op) && !isa<RankedTensorType>(op->getResultTypes()[0]);
}

void LayoutPropagation::rewriteRegion(Region &region) {
  std::deque<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.front();
    queue.pop_front();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      SmallVector<Value> results = op.getResults();
      for (Value result : results) {
        auto it = layouts.find(result);
        // If we haven't mapped this value skip.
        if (it == layouts.end())
          continue;
        LayoutInfo &info = it->second;
        assert(info.encodings.size() == 1 &&
               "we should have resolved to a single encoding");
        auto encoding = cast<RankedTensorType>(result.getType()).getEncoding();
        // If the encoding is already what we want skip.
        if (encoding == *info.encodings.begin())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        Operation *newOp = rewriteOp(&op);
        for (Region &R : newOp->getRegions())
          queue.push_back(&R);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        rewriteYieldOp(yieldOp);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        rewriteConditionOp(conditionOp);
      } else if (reduceToScalar(&op)) {
        rewriteReduceToScalar(&op);
      } else if (auto assertOp = dyn_cast<AssertOp>(&op)) {
        rewriteAssertOp(assertOp);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          auto it = layouts.find(operand.get());
          if (it == layouts.end())
            continue;
          Attribute encoding =
              cast<RankedTensorType>(operand.get().getType()).getEncoding();
          Value newOperand = getValueAs(operand.get(), encoding);
          op.setOperand(operand.getOperandNumber(), newOperand);
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
}

void LayoutPropagation::map(Value old, Value newV) {
  rewriteMapping[{old, cast<RankedTensorType>(newV.getType()).getEncoding()}] =
      newV;
}

Value LayoutPropagation::getRewrittenValue(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return value;
  auto layoutIt = layouts.find(value);
  if (layoutIt == layouts.end()) {
    return value;
  }
  assert(layoutIt->second.encodings.size() == 1 &&
         "we should have resolved to a single encoding");
  Attribute encodingPicked = *(layoutIt->second.encodings.begin());
  if (encodingPicked == tensorType.getEncoding())
    return value;
  return rewriteMapping.at({value, encodingPicked});
}

Value LayoutPropagation::getValueAs(Value value, Attribute encoding) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    Value rewrittenValue = getRewrittenValue(value);
    if (cast<RankedTensorType>(rewrittenValue.getType()).getEncoding() ==
        encoding)
      return rewrittenValue;
    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPointAfterValue(rewrittenValue);
    auto tmpType = tensorType.cloneWithEncoding(encoding);
    Value converted = ConvertLayoutOp::create(rewriter, value.getLoc(), tmpType,
                                              rewrittenValue);
    // TODO: we could cache the conversion.
    return converted;
  }
  return value;
}

Operation *LayoutPropagation::cloneElementwise(OpBuilder &rewriter,
                                               Operation *op,
                                               Attribute encoding) {
  Operation *newOp = rewriter.clone(*op);

  Attribute operandEnc;
  if (op->getNumOperands() > 0) {
    for (auto operand : op->getOperands()) {
      auto ty =
          dyn_cast<RankedTensorType>(getRewrittenValue(operand).getType());
      if (!ty)
        continue;
      auto enc = ty.getEncoding();
      if (inferDstEncoding(op, enc) == encoding) {
        operandEnc = enc;
        break;
      }
    }
    if (!operandEnc)
      operandEnc = inferSrcEncoding(op, encoding);
    assert(operandEnc);
  }

  for (OpOperand &operand : op->getOpOperands()) {
    newOp->setOperand(operand.getOperandNumber(),
                      getValueAs(operand.get(), operandEnc));
  }

  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
    auto origType = dyn_cast<RankedTensorType>(op->getResult(i).getType());
    if (!origType)
      continue;
    auto newType = origType.cloneWithEncoding(encoding);
    newOp->getResult(i).setType(newType);
  }
  return newOp;
}

Operation *LayoutPropagation::rewriteForOp(scf::ForOp forOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(forOp);
  for (auto [operand, result] :
       llvm::zip(forOp.getInitArgs(), forOp.getResults())) {
    Value convertedOperand = operand;
    if (layouts.count(result))
      convertedOperand =
          getValueAs(operand, *layouts[result].encodings.begin());
    operands.push_back(convertedOperand);
  }
  auto newForOp =
      scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), operands);
  newForOp->setAttrs(forOp->getAttrs());
  newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->getOperations().begin(),
      forOp.getBody()->getOperations());

  for (auto [oldResult, newResult] :
       llvm::zip(forOp.getResults(), newForOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }

  for (auto [oldArg, newArg] : llvm::zip(forOp.getBody()->getArguments(),
                                         newForOp.getBody()->getArguments())) {
    if (oldArg.getType() == newArg.getType()) {
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }
    map(oldArg, newArg);
  }
  return newForOp.getOperation();
}

Operation *LayoutPropagation::rewriteWhileOp(scf::WhileOp whileOp) {
  SmallVector<Value> operands;
  SmallVector<Type> returnTypes;
  OpBuilder rewriter(whileOp);
  for (auto [operand, arg] :
       llvm::zip(whileOp->getOperands(), whileOp.getBeforeArguments())) {
    Value convertedOperand = operand;
    if (layouts.count(arg))
      convertedOperand = getValueAs(operand, *layouts[arg].encodings.begin());
    operands.push_back(convertedOperand);
  }
  for (Value ret : whileOp.getResults()) {
    auto it = layouts.find(ret);
    if (it == layouts.end()) {
      returnTypes.push_back(ret.getType());
      continue;
    }
    auto origType = dyn_cast<RankedTensorType>(ret.getType());
    auto newType = origType.cloneWithEncoding(it->second.encodings[0]);
    returnTypes.push_back(newType);
  }

  auto newWhileOp =
      scf::WhileOp::create(rewriter, whileOp.getLoc(), returnTypes, operands);
  SmallVector<Type> argsTypesBefore;
  for (Value operand : operands)
    argsTypesBefore.push_back(operand.getType());
  SmallVector<Location> bbArgLocsBefore(argsTypesBefore.size(),
                                        whileOp.getLoc());
  SmallVector<Location> bbArgLocsAfter(returnTypes.size(), whileOp.getLoc());
  rewriter.createBlock(&newWhileOp.getBefore(), {}, argsTypesBefore,
                       bbArgLocsBefore);
  rewriter.createBlock(&newWhileOp.getAfter(), {}, returnTypes, bbArgLocsAfter);

  for (int i = 0; i < whileOp.getNumRegions(); ++i) {
    newWhileOp->getRegion(i).front().getOperations().splice(
        newWhileOp->getRegion(i).front().getOperations().begin(),
        whileOp->getRegion(i).front().getOperations());
  }

  auto remapArg = [&](Value oldVal, Value newVal) {
    if (oldVal.getType() == newVal.getType())
      oldVal.replaceAllUsesWith(newVal);
    else
      map(oldVal, newVal);
  };
  for (auto [oldResult, newResult] :
       llvm::zip(whileOp.getResults(), newWhileOp.getResults()))
    remapArg(oldResult, newResult);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getBeforeArguments(), newWhileOp.getBeforeArguments()))
    remapArg(oldArg, newArg);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getAfterArguments(), newWhileOp.getAfterArguments()))
    remapArg(oldArg, newArg);
  return newWhileOp.getOperation();
}

Operation *LayoutPropagation::rewriteIfOp(scf::IfOp ifOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(ifOp);
  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  for (unsigned i = 0, e = ifOp->getNumResults(); i < e; ++i) {
    auto it = layouts.find(ifOp->getResult(i));
    if (it == layouts.end())
      continue;
    auto origType = cast<RankedTensorType>(ifOp->getResult(i).getType());
    Attribute encoding = *(it->second.encodings.begin());
    newResultTypes[i] = origType.cloneWithEncoding(encoding);
  }
  auto newIfOp = scf::IfOp::create(rewriter, ifOp.getLoc(), newResultTypes,
                                   ifOp.getCondition(), true, true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  for (auto [oldResult, newResult] :
       llvm::zip(ifOp.getResults(), newIfOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }
  return newIfOp.getOperation();
}

void LayoutPropagation::rewriteYieldOp(scf::YieldOp yieldOp) {
  Operation *parentOp = yieldOp->getParentOp();
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    Type yieldType = operand.get().getType();
    if (isa<scf::ForOp, scf::IfOp>(parentOp))
      yieldType = parentOp->getResult(operand.getOperandNumber()).getType();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      yieldType =
          whileOp.getBeforeArguments()[operand.getOperandNumber()].getType();
    auto tensorType = dyn_cast<RankedTensorType>(yieldType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    yieldOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteConditionOp(scf::ConditionOp conditionOp) {
  scf::WhileOp whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
  for (unsigned i = 1; i < conditionOp->getNumOperands(); ++i) {
    OpOperand &operand = conditionOp->getOpOperand(i);
    Type argType = whileOp->getResult(operand.getOperandNumber() - 1).getType();
    auto tensorType = dyn_cast<RankedTensorType>(argType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    conditionOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteReduceToScalar(Operation *reduceOp) {
  OpBuilder rewriter(reduceOp);
  Attribute srcEncoding;
  // Since all the operands need to have the same encoding pick the first one
  // and use it for all the operands.
  for (Value operand : reduceOp->getOperands()) {
    auto it = layouts.find(operand);
    if (it != layouts.end()) {
      srcEncoding = it->second.encodings[0];
      break;
    }
  }
  if (!srcEncoding)
    return;
  for (OpOperand &operand : reduceOp->getOpOperands()) {
    Value newOperand = getValueAs(operand.get(), srcEncoding);
    reduceOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteAssertOp(AssertOp assertOp) {
  Attribute srcEncoding;
  // Only need to deal with the first operand which is the condition tensor.
  Value operand = assertOp->getOperand(0);
  auto it = layouts.find(operand);
  if (it == layouts.end())
    return;
  srcEncoding = it->second.encodings[0];
  Value newOperand = getValueAs(operand, srcEncoding);
  assertOp->setOperand(0, newOperand);
}

Operation *LayoutPropagation::rewriteOp(Operation *op) {
  opToDelete.insert(op);
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp);
  OpBuilder rewriter(op);
  Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
  if (auto convertOp = dyn_cast<ConvertLayoutOp>(op)) {
    Attribute srcEncoding = convertOp.getSrc().getType().getEncoding();
    auto it = layouts.find(convertOp.getSrc());
    if (it != layouts.end())
      srcEncoding = *(it->second.encodings.begin());
    Value src = getValueAs(convertOp.getSrc(), srcEncoding);
    auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    auto newType = tensorType.cloneWithEncoding(encoding);
    auto cvt = ConvertLayoutOp::create(rewriter, op->getLoc(), newType, src);
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
  if (canFoldIntoConversion(op, encoding)) {
    Operation *newOp = rewriter.clone(*op);
    auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    auto newType = tensorType.cloneWithEncoding(encoding);
    auto cvt = ConvertLayoutOp::create(rewriter, op->getLoc(), newType,
                                       newOp->getResult(0));
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
  if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp, GatherOp,
          ConvertLayoutOp, nvidia_gpu::WarpGroupDotWaitOp>(op)) {
    Operation *newOp = cloneElementwise(rewriter, op, encoding);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      if (oldResult.getType() == newResult.getType()) {
        oldResult.replaceAllUsesWith(newResult);
        continue;
      }
      map(oldResult, newResult);
    }
    return newOp;
  }
  llvm::report_fatal_error("unexpected op in rewrite");
  return nullptr;
}

bool canBeRemat(Operation *op) {
  if (isa<LoadOp, StoreOp>(op))
    return !isExpensiveLoadOrStore(op);
  if (isa<AtomicRMWOp, AtomicCASOp, DotOp>(op))
    return false;
  if (auto gather = dyn_cast<GatherOp>(op))
    return !gather.getEfficientLayout();

  if (isa<scf::WhileOp, scf::ConditionOp>(op))
    return false;

  return true;
}

void LayoutRematerialization::updateRematMapping(
    SmallVector<std::tuple<Value, Value>> &values) {
  for (auto [old, newV] : values) {
    auto it = mappedValues.find(old);
    if (it != mappedValues.end()) {
      Attribute encoding = it->second;
      auto rematIt = rematMapping.find({old, it->second});
      assert(rematIt != rematMapping.end());
      Value replacedValue = rematIt->second;
      rematMapping.erase(rematIt);
      mappedValues.erase(it);
      // Loop through the replacement value to find the new version of remat
      // value. This should be okay as the number of values should be small.
      for (auto [before, after] : values) {
        if (before == replacedValue) {
          replacedValue = after;
          break;
        }
      }
      rematMapping[{newV, encoding}] = replacedValue;
      mappedValues[newV] = encoding;
    }
  }
}

void LayoutRematerialization::rewriteSlice(SetVector<Value> &slice,
                                           DenseMap<Value, Attribute> &layout,
                                           ConvertLayoutOp convertOp,
                                           IRMapping &mapping) {
  SetVector<Operation *> opsToRewrite;
  // Keep track of yield operands that need to be duplicated.
  DenseMap<Operation *, SmallVector<int>> yieldOperandsMap;
  // Keep these around to remove them from the slice after our collection pass
  // This ensures we don't duplicate them during an for rewrite or causing the
  // for/yield to fall out of sync
  SetVector<Value> valuesWithExistingRemat;
  for (Value v : slice) {
    auto layoutIt = layout.find(v);
    assert(layoutIt != layout.end());
    // If we already have a remat value for this value, use it.
    if (Value remat = getRematValue(v, layoutIt->second)) {
      mapping.map(v, remat);
      valuesWithExistingRemat.insert(v);
      continue;
    }
    if (v.getDefiningOp()) {
      opsToRewrite.insert(v.getDefiningOp());
      if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
        unsigned operandIdx = cast<OpResult>(v).getResultNumber();
        opsToRewrite.insert(ifOp.thenYield().getOperation());
        yieldOperandsMap[ifOp.thenYield()].push_back(operandIdx);
        opsToRewrite.insert(ifOp.elseYield().getOperation());
        yieldOperandsMap[ifOp.elseYield()].push_back(operandIdx);
      }
    } else {
      BlockArgument blockArg = cast<BlockArgument>(v);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (auto loopOp = cast<LoopLikeOpInterface>(parentOp)) {
        opsToRewrite.insert(loopOp.getOperation());
        OpOperand *operand = loopOp.getTiedLoopYieldedValue(blockArg);
        auto yieldOp = blockArg.getOwner()->getTerminator();
        yieldOperandsMap[yieldOp].push_back(operand->getOperandNumber());
        opsToRewrite.insert(yieldOp);
      }
    }
  }
  slice.set_subtract(valuesWithExistingRemat);
  opsToRewrite = mlir::topologicalSort(opsToRewrite);

  // replaceAllUsesWith calls delayed until after initial rewrite.
  // This is required for slice.count(value) to work mid rewrite.
  SmallVector<std::tuple<Value, Value>> replacements;

  SmallVector<Operation *> deadOps;
  IRRewriter builder(slice.begin()->getContext());
  for (Operation *op : opsToRewrite) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Keep a mapping of the operands index to the new operands index.
      SmallVector<std::pair<size_t, size_t>> argMapping;
      SmallVector<Value> newOperands;
      for (auto arg : forOp.getRegionIterArgs()) {
        if (slice.count(arg)) {
          OpOperand &initVal = *forOp.getTiedLoopInit(arg);
          argMapping.push_back(std::make_pair(
              forOp.getTiedLoopResult(&initVal).getResultNumber(),
              forOp.getInitArgs().size() + newOperands.size()));
          newOperands.push_back(mapping.lookup(initVal.get()));
        }
      }
      // Create a new for loop with the new operands.
      scf::ForOp newForOp = replaceForOpWithNewSignature(
          builder, forOp, newOperands, replacements);
      deadOps.push_back(forOp.getOperation());
      Block &loopBody = *newForOp.getBody();
      for (auto m : argMapping) {
        mapping.map(forOp.getResult(m.first), newForOp.getResult(m.second));
        int numIndVars = newForOp.getNumInductionVars();
        mapping.map(loopBody.getArgument(m.first + numIndVars),
                    loopBody.getArgument(m.second + numIndVars));
        LLVM_DEBUG({
          DBGS() << "mapping forOp "
                 << loopBody.getArgument(m.first + numIndVars) << " to "
                 << loopBody.getArgument(m.second + numIndVars) << '\n';
        });
        // The result is not in the layout/slice, the argument is.
        Value oldArg = loopBody.getArgument(m.first + numIndVars);
        addRematValue(newForOp.getResult(m.first), layout[oldArg],
                      newForOp.getResult(m.second));
        addRematValue(oldArg, layout[oldArg],
                      loopBody.getArgument(m.second + numIndVars));
      }
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Type> newTypes;
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          auto it = layout.find(res);
          assert(it != layout.end());

          auto oldType = cast<RankedTensorType>(res.getType());
          auto newType = oldType.cloneWithEncoding(it->second);
          newTypes.push_back(newType);
        }
      }
      scf::IfOp newIfOp =
          replaceIfOpWithNewSignature(builder, ifOp, newTypes, replacements);
      unsigned oldIdx = 0;
      unsigned newIdx = ifOp.getNumResults();
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          // Why can't we use res instead of ifOp.getResult(oldIdx)?
          mapping.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
          addRematValue(ifOp.getResult(oldIdx), layout[res],
                        newIfOp.getResult(newIdx));
          ++newIdx;
        }
        ++oldIdx;
      }
      deadOps.push_back(ifOp.getOperation());
      continue;
    }
    builder.setInsertionPoint(op);
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
      SmallVector<int> operandsToRewrite = yieldOperandsMap[op];
      // Sort so that operands are added in the same order as the new scf
      // results/arguments.
      std::sort(operandsToRewrite.begin(), operandsToRewrite.end());
      for (int operandIdx : operandsToRewrite) {
        yieldOperands.push_back(mapping.lookup(yieldOp.getOperand(operandIdx)));
      }
      scf::YieldOp::create(builder, op->getLoc(), yieldOperands);
      op->erase();
      continue;
    }
    if (isa<arith::ConstantOp>(op)) {
      Operation *newOp = builder.clone(*op);
      auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
      auto newType = tensorType.cloneWithEncoding(layout[op->getResult(0)]);
      auto cvt = ConvertLayoutOp::create(builder, op->getLoc(), newType,
                                         newOp->getResult(0));
      mapping.map(op->getResult(0), cvt.getResult());
      addRematValue(op->getResult(0), layout[op->getResult(0)],
                    cvt.getResult());
      continue;
    }
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [old, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layout.find(old);
      if (it == layout.end())
        continue;
      auto newType =
          cast<RankedTensorType>(old.getType()).cloneWithEncoding(it->second);
      newV.setType(newType);
      addRematValue(old, it->second, newV);
    }
  }
  // Check mapping and see if there are existing convertOps on the old Argument
  convertOp.replaceAllUsesWith(mapping.lookup(convertOp.getSrc()));
  opToDelete.insert(convertOp);

  updateRematMapping(replacements);
  for (auto &kv : replacements) {
    builder.replaceAllUsesWith(std::get<0>(kv), std::get<1>(kv));
  }

  for (Operation *op : deadOps)
    opToDelete.insert(op);
}

void LayoutRematerialization::rewriteSlice(SetVector<Value> &slice,
                                           DenseMap<Value, Attribute> &layout,
                                           ConvertLayoutOp convertOp) {
  IRMapping mapping;
  rewriteSlice(slice, layout, convertOp, mapping);
}

LogicalResult LayoutRematerialization::getConvertBackwardSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation) {
  // Allow re-using existing conversions for a value. Check dominance of any
  // reusable materializations against the root value. This is sufficient
  // because the conversions are processed in post-order.
  auto getExistingConversion = [&](OpOperand &value, Attribute encoding) {
    Value remat = getRematValue(value.get(), encoding);
    if (!remat)
      return Value();
    // `value` can be replaced with an existing rematerialization if it
    // dominates the current use of value.
    Operation *user = value.getOwner();
    if (domInfo.properlyDominates(remat, user)) {
      return remat;
    }
    // FIXME: If the current user is a conversion, then we know it will become
    // a no-op when its operand is replaced with `remat`, but we need to check
    // that its users are all dominated by `remat` so the IR is valid.
    // if (isa<ConvertLayoutOp>(user) && remat.getDefiningOp() &&
    //     domInfo.properlyDominates(user, remat.getDefiningOp())) {
    //   for (Operation *op : user->getUsers()) {
    //     if (!domInfo.dominates(remat, op))
    //       return Value();
    //   }
    //   return remat;
    // }
    return Value();
  };

  return mlir::getConvertBackwardSlice(root, slice, rootEncoding, layout,
                                       stopPropagation, getExistingConversion);
}

LogicalResult LayoutRematerialization::getRematerializableSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &sliceArg,
    DenseMap<Value, Attribute> &layoutArg,
    std::function<bool(Operation *)> stopPropagation) {
  // Operate on copies of the input, we do not want to modify them unless we
  // have succeeded.
  auto slice = sliceArg;
  auto layout = layoutArg;
  LogicalResult result = getConvertBackwardSlice(root, rootEncoding, slice,
                                                 layout, stopPropagation);
  if (result.failed() || slice.empty())
    return failure();

  // Check if all the operations in the slice can be rematerialized.
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      if (!canBeRemat(op))
        return failure();
    }
  }
  sliceArg = std::move(slice);
  layoutArg = std::move(layout);
  return success();
}

bool LayoutRematerialization::backwardRematerialization() {
  bool changed = false;
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    backwardRematerialization(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    } else {
      changed = true;
    }
  }
  return changed;
}

void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertOnTopOfExtOrBroadcast(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertIntoConditionals() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertIntoConditionals(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

static bool isExpensiveMathOp(Operation *op) {
  // These operations are either multiple instructions or have throughput
  // lower than 16 according to the arithmetic instructions table in:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
  return isa<arith::DivFOp, math::ErfcOp, math::SinhOp, math::CoshOp,
             math::TanhOp, math::AsinhOp, math::AcoshOp, math::AtanhOp,
             math::CtPopOp, math::CountLeadingZerosOp,
             math::CountTrailingZerosOp, math::ExpOp, math::Exp2Op,
             math::ExpM1Op, math::LogOp, math::Log2Op, math::Log10Op,
             math::Log1pOp, math::SinOp, math::CosOp, math::TanOp, math::AsinOp,
             math::AcosOp, math::AtanOp, math::Atan2Op, math::PowFOp,
             math::SqrtOp, math::RsqrtOp, math::ErfOp, math::CbrtOp>(op);
}

static int64_t getByteCount(Value result, int64_t minElementCount,
                            int64_t minBitWidth) {
  int64_t elementCount = 0;
  int64_t dtypeBitWidth = 0;
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    elementCount = tensorTy.getNumElements();
    auto elemType = tensorTy.getElementType();
    if (elemType.isIntOrFloat()) {
      dtypeBitWidth = elemType.getIntOrFloatBitWidth();
    }
  }
  if (elementCount < minElementCount) {
    elementCount = minElementCount;
  }
  if (dtypeBitWidth < minBitWidth) {
    dtypeBitWidth = minBitWidth;
  }
  return (elementCount * dtypeBitWidth) >> 3;
}

/// Compute the cost of a ConvertLayoutOp with source \p convertSrc.
int64_t getConvertCost(Value convertSrc) {
  // Measure the number of bytes that we're manipulating with the
  // ConvertLayoutOp. We pessimistically assume that we round-trip
  // through shared memory and that we cannot vectorise sub-register
  // loads/stores, so we set a minimum element count of 32 (the warp
  // size and number of shared memory banks) and minimum bitwidth of
  // 32 (the width per bank of the shared memory load/store unit).
  auto convertLayoutBytes = getByteCount(convertSrc, 32, 32);
  // We measure costs in standardised milli-SM-cycles. The smem load
  // and store each cost 8 * convertLayoutBytes, and then we double
  // it to account for extra cost due to synchronisation.
  return 32 * convertLayoutBytes;
}

/// Determine whether rematerializing \p slice is beneficial given that it will
/// eliminate \p convertOp and require creating new convert ops with cost \p
/// newCvtCost.
bool isRematBeneficial(ConvertLayoutOp convertOp, const SetVector<Value> &slice,
                       int64_t newCvtCost) {
  // Identify all operations in the slice
  SetVector<Operation *> sliceOps;
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      sliceOps.insert(op);
    }
  }

  // Compute single-use operations
  DenseMap<Operation *, bool> isSingleUse;
  std::function<bool(Operation *)> isOpSingleUse;
  isOpSingleUse = [&](Operation *op) -> bool {
    // lookup in memoization array:
    auto it = isSingleUse.find(op);
    if (it != isSingleUse.end()) {
      return it->second;
    }

    bool singleUse = true;

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (user == convertOp) {
          continue;
        }
        if (sliceOps.contains(user)) {
          if (!isOpSingleUse(user)) {
            singleUse = false;
            break;
          }
        } else {
          singleUse = false;
          break;
        }
      }
      if (!singleUse) {
        break;
      }
    }

    // insert into memoization array:
    isSingleUse[op] = singleUse;
    return singleUse;
  };

  int64_t convertLayoutCost = getConvertCost(convertOp.getSrc());
  int64_t rematerialisationCost = newCvtCost;

  // Evaluate single-use status for every operation in slice
  for (Operation *op : sliceOps) {
    auto dialect = op->getDialect();
    if (isOpSingleUse(op)) {
      // when we rematerialise, this operation does not get duplicated
      // so it does not contribute to our cost model:
      continue;
    } else if (isa<arith::ConstantOp>(op)) {
      // special-case: arith.constant has zero cost
      continue;
    } else if (isa<LoadOp>(op) || isa<LocalLoadOp>(op)) {
      // optimistically assume L1-cached:
      for (Value result : op->getResults()) {
        rematerialisationCost += 8 * getByteCount(result);
      }
    } else if (isa<arith::ArithDialect, math::MathDialect>(dialect)) {
      // this is an arithmetic operation; we distinguish between cheap
      // operations (such as floating point add/mul which can be fused
      // as halves of a single-cycle FMA instruction) and expensive
      // operations which use the special function unit and/or involve
      // multiple instructions.
      int64_t multiplier = isExpensiveMathOp(op) ? 8 : 1;
      for (Value result : op->getResults()) {
        rematerialisationCost += multiplier * getByteCount(result);
      }
    } else if (isa<ReduceOp>(op)) {
      // Reduce op introduce much cost.
      auto reduceOp = dyn_cast<ReduceOp>(op);
      ReduceOpHelper helper(reduceOp);
      if (!helper.isAssociative()) {
        // We shouldn't rematerize a no associative reduce op if it has multiple
        // use chain.
        LDBG("  skipped rematerialization due to non-associative reduce in the "
             "slice");
        return false;
      }
      rematerialisationCost += helper.getIntraWarpSizeWithUniqueData();
      rematerialisationCost += 8 * helper.getInterWarpSizeWithUniqueData();
    }
  }

  LLVM_DEBUG({
    DBGS() << "  convert layout cost: " << convertLayoutCost << "\n";
    DBGS() << "  rematerialisation cost: " << rematerialisationCost << "\n";
  });

  return convertLayoutCost >= rematerialisationCost;
}

void LayoutRematerialization::backwardRematerialization(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return;
  Value oldV = convertOp.getSrc();
  LDBG("check backward remat with source " << oldV << " encoding "
                                           << targetType.getEncoding());
  // Check to see if there are existing remat'ed values for the pair of oldValue
  // and encoding. Make sure it dominates the current conversion.
  Value newV = getRematValue(oldV, targetType.getEncoding());
  if (newV && domInfo.properlyDominates(newV, convertOp)) {
    // Replace it with the remat'ed value.
    convertOp.replaceAllUsesWith(newV);
    opToDelete.insert(convertOp);
    LDBG("found remat'ed value" << newV);
    return;
  }

  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout);
  if (result.failed()) {
    LDBG("  getRematerializableSlice failed");
    return;
  }

  // 2. Determine whether rematerialisation is beneficial.
  if (!isRematBeneficial(convertOp, slice, /*newCvtCost=*/0)) {
    LDBG("  skipped rematerialization due to higher cost");
    return;
  }

  LLVM_DEBUG({
    DBGS() << "  remat convert op " << convertOp << '\n';
    for (Value v : slice)
      DBGS() << "    " << v << '\n';
  });

  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp);
}

void LayoutRematerialization::hoistConvertDotOperand() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertDotOperand(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertDotOperand(
    ConvertLayoutOp convertOp) {
  auto targetType = convertOp.getType();
  // The pass is targeted to MMA dot operands

  auto canBePipelined = [&](ConvertLayoutOp convertOp) {
    // FIXME: Check that the parent is a for loop
    auto parent = convertOp->getParentOp();
    if (!parent)
      return false;

    // Find all the dot-like ops in the for loop that have a dot operand
    // encoding on the lhs and check if any of them post-dominates the load +
    // cvt
    SmallVector<Operation *> dotLikeOps;
    parent->walk([&](Operation *op) {
      if (!isa<mlir::triton::DotOpInterface>(op))
        return;
      auto opType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!opType)
        return;
      auto dotEnc = dyn_cast<DotOperandEncodingAttr>(opType.getEncoding());
      if (!dotEnc)
        return;
      if (isa<MmaEncodingTrait>(dotEnc.getParent()))
        dotLikeOps.push_back(op);
    });
    if (dotLikeOps.empty())
      return false;
    return llvm::any_of(dotLikeOps, [&](Operation *dot) {
      return postDomInfo.postDominates(dot, convertOp);
    });
  };

  // We move convert #dot_operand next to their loads. This is done
  // so that it's then easy to pipeline these loads
  if (!canBePipelined(convertOp))
    return;

  // We hoist over any operation that can be done without data movement between
  // threads We do views and elementwise pure ops for now
  auto noDataMovement = [](Operation *op) {
    return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
           isa<BroadcastOp, Fp4ToFpOp, ConvertLayoutOp, UpcastFpOpInterface>(
               op) ||
           isView(op);
  };
  // Stop the slice as soon as we find an operation that cannot be done without
  // data movement between threads
  auto stop = std::not_fn(noDataMovement);

  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  // Set-up the conversion "cache"
  LogicalResult result = getConvertBackwardSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout, stop);
  if (result.failed())
    return;

  IRMapping mapping;
  OpBuilder builder(convertOp.getContext());
  SetVector<Value> innerSlice;
  for (Value v : slice) {
    if (!v.getDefiningOp()) {
      LLVM_DEBUG(
          { DBGS() << "  Block arguments not supported. Got " << v << "\n"; });
      return;
    }

    // We expect the leaves of the slice to be Load, DescriptorLoad or
    // arith::Constant This could be generalised if necessary
    if (!isa<LoadOp, DescriptorLoadOp>(v.getDefiningOp())) {
      auto op = v.getDefiningOp();
      if (isa<arith::ConstantOp>(op) || noDataMovement(op)) {
        innerSlice.insert(v);
        continue;
      } else {
        LLVM_DEBUG({
          DBGS() << "  Leaves must be Load, DescriptorLoad or Constant. Got "
                 << v << "\n";
        });
        return;
      }
    }
    Operation *loadOp = v.getDefiningOp();
    builder.setInsertionPointAfter(loadOp);
    auto type = dyn_cast<RankedTensorType>(loadOp->getResult(0).getType());
    if (!type)
      continue;
    auto newType = type.cloneWithEncoding(layout[loadOp->getResult(0)]);
    auto newConvertOp = ConvertLayoutOp::create(builder, convertOp.getLoc(),
                                                newType, loadOp->getResult(0));
    mapping.map(loadOp->getResult(0), newConvertOp.getResult());
  }

  if (innerSlice.empty()) {
    return;
  }

  LLVM_DEBUG({
    DBGS() << "  Hoisting " << convertOp << '\n';
    for (Value v : innerSlice)
      DBGS() << "    " << v << '\n';
  });

  rewriteSlice(innerSlice, layout, convertOp, mapping);
}

// For convert left we try to hoist them above type extension to reduce the cost
// of the convert.
void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return;

  auto isExtOrBroadcastOp = [](Operation *op) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, BroadcastOp,
            ExpandDimsOp>(op)) {
      return true;
    }
    if (auto fpToFpOp = dyn_cast<FpToFpOp>(op)) {
      auto srcType = cast<RankedTensorType>(fpToFpOp.getOperand().getType());
      return getElementBitWidth(srcType) <
             getElementBitWidth(cast<RankedTensorType>(fpToFpOp.getType()));
    }
    return false;
  };
  // 1. Take a backward slice of all the tensor dependencies.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      isExtOrBroadcastOp);
  if (result.failed())
    return;

  Operation *extOrBroadcastOp = nullptr;
  unsigned sliceSize = slice.size();
  for (unsigned i = 0; i < sliceSize; i++) {
    Value v = slice[i];
    Operation *op = v.getDefiningOp();
    if (!op || !isExtOrBroadcastOp(op))
      continue;

    Attribute srcEncoding = inferSrcEncoding(op, layout[v]);
    if (!srcEncoding)
      return;

    // If we can rematerialize the rest of the ext slice we can ignore this ext
    // as it won't need a convert.
    if (succeeded(getRematerializableSlice(op->getOpOperand(0), srcEncoding,
                                           slice, layout)))
      continue;

    // Only apply it if there is a single ext op otherwise we would have to
    // duplicate the convert.
    if (extOrBroadcastOp != nullptr)
      return;
    extOrBroadcastOp = op;
  }

  if (extOrBroadcastOp == nullptr)
    return;
  Attribute dstEncoding = layout[extOrBroadcastOp->getResult(0)];
  Attribute srcEncoding = inferSrcEncoding(extOrBroadcastOp, dstEncoding);
  if (!srcEncoding)
    return;
  int64_t newCvtCost = getConvertCost(extOrBroadcastOp->getOperand(0));
  if (!isRematBeneficial(convertOp, slice, newCvtCost))
    return;
  // Move the convert before the ext op and rewrite the slice.
  OpBuilder builder(extOrBroadcastOp);
  auto tensorType =
      cast<RankedTensorType>(extOrBroadcastOp->getOperand(0).getType());
  auto newType = tensorType.cloneWithEncoding(srcEncoding);
  auto newConvertOp = ConvertLayoutOp::create(
      builder, convertOp.getLoc(), newType, extOrBroadcastOp->getOperand(0));
  Operation *newExtOrBroadcast = builder.clone(*extOrBroadcastOp);
  newExtOrBroadcast->setOperand(0, newConvertOp.getResult());
  auto oldExtOrBroadcastType =
      cast<RankedTensorType>(extOrBroadcastOp->getResult(0).getType());
  Type newExtOrBroadcastType =
      oldExtOrBroadcastType.cloneWithEncoding(dstEncoding);
  newExtOrBroadcast->getResult(0).setType(newExtOrBroadcastType);
  IRMapping mapping;
  mapping.map(extOrBroadcastOp->getResult(0), newExtOrBroadcast->getResult(0));
  slice.remove(extOrBroadcastOp->getResult(0));
  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp, mapping);
}

void LayoutRematerialization::hoistConvertIntoConditionals(
    ConvertLayoutOp convertOp) {
  // Take the backward slice of tensor dependencies rooted at the conversion,
  // stopping at conditionals. This subslice is used to initialize the analysis.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  auto isIfOp = [](Operation *op) { return isa<scf::IfOp>(op); };
  if (failed(getRematerializableSlice(convertOp.getSrcMutable(),
                                      convertOp.getType().getEncoding(), slice,
                                      layout, isIfOp)))
    return;

  // These are the conditional edges above which conversions should be hoisted.
  // The value represents the `scf.if` op result and the operand represents the
  // edge into one of the branches.
  SmallVector<std::pair<Value, OpOperand *>> hoistAbove;

  // The list of `scf.if` op results in the slice that are not rematerializable.
  // Hoisting is terminated at these values.
  SmallVector<OpResult> terminals;

  // This loop recurses through the subslices of the backwards dependencies, so
  // re-query the size of `slice`.
  for (unsigned i = 0; i != slice.size(); ++i) {
    Value v = slice[i];
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      continue;

    Attribute rootLayout = layout.at(v);
    unsigned resIdx = cast<OpResult>(v).getResultNumber();

    // Take the backward slice along each branch.
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    OpOperand &thenRes = thenYield.getResultsMutable()[resIdx];
    OpOperand &elseRes = elseYield.getResultsMutable()[resIdx];

    auto newSlice = slice;
    auto newLayout = layout;

    LogicalResult thenResult = getRematerializableSlice(
        thenRes, rootLayout, newSlice, newLayout, isIfOp);
    LogicalResult elseResult = getRematerializableSlice(
        elseRes, rootLayout, newSlice, newLayout, isIfOp);

    // If propagation across both edges of this conditional succeeded, then we
    // don't need to hoist across it. Merge into the current slice.
    if (succeeded(thenResult) && succeeded(elseResult)) {
      slice = std::move(newSlice);
      layout = std::move(newLayout);
      continue;
    }

    // If propagation across both edges failed, then this conditional
    // terminates backwards rematerialization.
    if (failed(thenResult) && failed(elseResult)) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    // Only hoist into conditionals inside loops. The assumption is that an if
    // inside a loop executes fewer than the total number of loop iterations,
    // making this hoist profitable.
    if (!isa<scf::ForOp>(ifOp->getParentOp())) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    slice = std::move(newSlice);
    layout = std::move(newLayout);
    // The layout conversion can be rematerialized along one edge but not the
    // other. We can hoist the conversion into the other branch. Push this
    // into the subslice list for analysis.
    if (succeeded(thenResult)) {
      hoistAbove.emplace_back(v, &elseRes);
    } else {
      hoistAbove.emplace_back(v, &thenRes);
    }
  }

  // Exit early if there is nothing to do.
  if (hoistAbove.empty())
    return;

  // Rematerialize failed hoists right before the condtional, and hoist those
  // that succeeded into the branch and then rewrite the slice.
  IRMapping mapping;
  auto hoistRemat = [&](OpBuilder &b, Value v, Attribute encoding) {
    auto tensorType = cast<RankedTensorType>(v.getType());
    auto newType = tensorType.cloneWithEncoding(encoding);
    Value newCvt = ConvertLayoutOp::create(b, convertOp.getLoc(), newType, v);

    mapping.map(v, newCvt);
    slice.remove(v);
  };
  for (Value v : terminals) {
    OpBuilder b(v.getContext());
    b.setInsertionPointAfter(v.getDefiningOp());
    hoistRemat(b, v, layout.at(v));
  }
  for (auto [result, edge] : hoistAbove) {
    OpBuilder b(edge->getOwner());
    hoistRemat(b, edge->get(), layout.at(result));
  }
  rewriteSlice(slice, layout, convertOp, mapping);
}

bool backwardRematerialization(ModuleOp module) {
  bool changed = false;
  module.walk([&](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp);
    changed |= layoutRemat.backwardRematerialization();
    layoutRemat.cleanup();
  });
  return changed;
}

void hoistConvert(ModuleOp module) {
  SmallVector<ConvertLayoutOp> convertOps;
  module.walk([](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp);
    layoutRemat.hoistConvertOnTopOfExtOrBroadcast();
    layoutRemat.cleanup();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertIntoConditionals();
    layoutRemat.cleanup();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertDotOperand();
    layoutRemat.cleanup();
  });
}
} // namespace

class TritonGPURemoveLayoutConversionsPass
    : public impl::TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TritonGPURemoveLayoutConversionsPass)
  // Cleanup convert ops.
  void cleanupConvertOps() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet cleanUpPatterns(context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns, context);
    if (applyPatternsGreedily(m, std::move(cleanUpPatterns)).failed()) {
      signalPassFailure();
    }

    LLVM_DEBUG({
      DBGS() << "Module after canonicalizing:\n";
      m.dump();
    });
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // 1. Propagate layout forward starting from "anchor" ops.
    m.walk([](FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp);
      layoutPropagation.initAnchorLayout();
      layoutPropagation.propagateLayout();
      layoutPropagation.resolveConflicts();
      layoutPropagation.rewrite();
    });

    LLVM_DEBUG({
      DBGS() << "Module after propagating layouts forward:\n";
      m.dump();
    });

    cleanupConvertOps();

    bool changed = false;
    do {
      changed = false;
      // 2. For remaining convert ops, try to rematerialize the slice of
      // producer operation to avoid having to convert.
      changed = backwardRematerialization(m);
      LLVM_DEBUG({
        DBGS() << "Module after backward remat:\n";
        m.dump();
      });

      // Cleanup dummy converts created during backward remat.
      cleanupConvertOps();
    } while (changed);
    // 3. For remaining converts, try to hoist them above cast generating larger
    // size types in order to reduce the cost of the convert op.
    hoistConvert(m);
    LLVM_DEBUG({
      DBGS() << "Module after hoisting converts:\n";
      m.dump();
    });

    // 4. Apply clean up patterns to remove dead convert and dead code generated
    // by the previous transformations.
    // scf canonicalization is best effort and doesn't need to converge
    RewritePatternSet convertCleanup(context);
    ConvertLayoutOp::getCanonicalizationPatterns(convertCleanup, context);
    if (applyPatternsGreedily(m, std::move(convertCleanup)).failed()) {
      signalPassFailure();
    }

    RewritePatternSet scfCleanup(context);
    populateForOpDeadArgumentElimination(scfCleanup);
    scf::ForOp::getCanonicalizationPatterns(scfCleanup, context);
    scf::IfOp::getCanonicalizationPatterns(scfCleanup, context);
    if (applyPatternsGreedily(m, std::move(scfCleanup)).failed()) {
      LLVM_DEBUG(DBGS() << "scf cleanup did not converge\n");
    }

    LLVM_DEBUG({
      DBGS() << "Module after final cleanups:\n";
      m.dump();
    });
  }
};

} // namespace mlir::triton::gpu
