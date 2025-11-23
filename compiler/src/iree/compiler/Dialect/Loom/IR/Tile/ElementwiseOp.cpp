// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.h"
#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tile.elementwise - ShapeAwareOpInterface
//===----------------------------------------------------------------------===//

ValueRange TileElementwiseOp::getOperandDynamicDims(unsigned idx) {
  // Find the offset into input_dims for the idx-th input.
  // Each input contributes its number of dynamic dimensions.
  size_t offset = 0;
  for (unsigned i = 0; i < idx; ++i) {
    auto tileType = cast<TileType>(getInputs()[i].getType());
    offset += tileType.getNumDynamicDims();
  }

  auto tileType = cast<TileType>(getInputs()[idx].getType());
  size_t numDynDims = tileType.getNumDynamicDims();

  return getInputDims().slice(offset, numDynDims);
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Phase 1: Propagation Patterns
//===----------------------------------------------------------------------===//

// Propagate splat constants into the region as scalar constants.
// elementwise(%const_splat, %tile) { use %arg0 } ->
// elementwise(%tile) { %c = arith.constant; use %c }
struct PropagateConstantIntoRegion
    : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    Block& bodyBlock = op.getBody().front();
    auto inputs = op.getInputs();

    // Find splat constant inputs.
    SmallVector<std::pair<unsigned, TileConstantOp>> splatInputs;
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (auto constantOp = input.getDefiningOp<TileConstantOp>()) {
        if (constantOp.isSplat()) {
          splatInputs.emplace_back(idx, constantOp);
        }
      }
    }

    if (splatInputs.empty()) {
      return failure();
    }

    // Collect non-constant inputs and their dims.
    // Note: newInputs can be empty if all inputs are splat constants.
    // The op supports zero inputs - shape comes from result type + result_dims.
    SmallVector<Value> newInputs;
    SmallVector<Value> newInputDims;
    SmallVector<unsigned> oldToNewArgIdx(inputs.size(),
                                         std::numeric_limits<unsigned>::max());

    unsigned newIdx = 0;
    for (unsigned i = 0; i < inputs.size(); ++i) {
      bool isSplatConst =
          llvm::any_of(splatInputs, [i](auto& p) { return p.first == i; });
      if (!isSplatConst) {
        oldToNewArgIdx[i] = newIdx++;
        newInputs.push_back(inputs[i]);
        // Add dynamic dims for this input.
        for (Value dim : op.getOperandDynamicDims(i)) {
          newInputDims.push_back(dim);
        }
      }
    }

    // Build the new op.
    Location loc = op.getLoc();
    auto newOp =
        TileElementwiseOp::create(rewriter, loc, op.getResult().getType(),
                                  newInputs, newInputDims, op.getResultDims());

    // Set up the new region.
    Block& newBlock = newOp.getBody().emplaceBlock();
    for (Value input : newInputs) {
      auto tileType = cast<TileType>(input.getType());
      newBlock.addArgument(tileType.getElementType(), loc);
    }

    // Clone the body, replacing constant args with scalar constants.
    rewriter.setInsertionPointToStart(&newBlock);
    IRMapping mapping;

    // Map non-constant args to new block args.
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (oldToNewArgIdx[i] != std::numeric_limits<unsigned>::max()) {
        mapping.map(bodyBlock.getArgument(i),
                    newBlock.getArgument(oldToNewArgIdx[i]));
      }
    }

    // For constant args, create scalar constants.
    for (auto [idx, constantOp] : splatInputs) {
      auto splatValue = cast<TypedAttr>(constantOp.getSplatValue());
      Value scalarConst =
          arith::ConstantOp::create(rewriter, loc, splatValue).getResult();
      mapping.map(bodyBlock.getArgument(idx), scalarConst);
    }

    // Clone body ops.
    for (Operation& bodyOp : bodyBlock.getOperations()) {
      rewriter.clone(bodyOp, mapping);
    }

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Propagate poison inputs into the region as scalar poison.
// Instead of immediately folding to poison, let poison propagate through
// the scalar ops so that partial computations might still be valid.
struct PropagatePoisonIntoRegion : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    Block& bodyBlock = op.getBody().front();
    auto inputs = op.getInputs();

    // Find poison inputs.
    SmallVector<unsigned> poisonInputIdxs;
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (input.getDefiningOp<ub::PoisonOp>()) {
        poisonInputIdxs.push_back(idx);
      }
    }

    if (poisonInputIdxs.empty()) {
      return failure();
    }

    // Collect non-poison inputs.
    // Note: newInputs can be empty if all inputs are poison.
    // The op supports zero inputs - shape comes from result type + result_dims.
    SmallVector<Value> newInputs;
    SmallVector<Value> newInputDims;
    SmallVector<unsigned> oldToNewArgIdx(inputs.size(),
                                         std::numeric_limits<unsigned>::max());

    unsigned newIdx = 0;
    for (unsigned i = 0; i < inputs.size(); ++i) {
      bool isPoison = llvm::is_contained(poisonInputIdxs, i);
      if (!isPoison) {
        oldToNewArgIdx[i] = newIdx++;
        newInputs.push_back(inputs[i]);
        for (Value dim : op.getOperandDynamicDims(i)) {
          newInputDims.push_back(dim);
        }
      }
    }

    // Build the new op.
    Location loc = op.getLoc();
    auto newOp =
        TileElementwiseOp::create(rewriter, loc, op.getResult().getType(),
                                  newInputs, newInputDims, op.getResultDims());

    Block& newBlock = newOp.getBody().emplaceBlock();
    for (Value input : newInputs) {
      auto tileType = cast<TileType>(input.getType());
      newBlock.addArgument(tileType.getElementType(), loc);
    }

    // Clone the body, replacing poison args with scalar poison.
    rewriter.setInsertionPointToStart(&newBlock);
    IRMapping mapping;

    // Map non-poison args.
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (oldToNewArgIdx[i] != std::numeric_limits<unsigned>::max()) {
        mapping.map(bodyBlock.getArgument(i),
                    newBlock.getArgument(oldToNewArgIdx[i]));
      }
    }

    // For poison args, create scalar poison.
    for (unsigned idx : poisonInputIdxs) {
      Type elemType = bodyBlock.getArgument(idx).getType();
      Value scalarPoison =
          ub::PoisonOp::create(rewriter, loc, elemType, nullptr).getResult();
      mapping.map(bodyBlock.getArgument(idx), scalarPoison);
    }

    // Clone body ops.
    for (Operation& bodyOp : bodyBlock.getOperations()) {
      rewriter.clone(bodyOp, mapping);
    }

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 2: Cleanup Patterns
//===----------------------------------------------------------------------===//

// Eliminate unused inputs from elementwise.
// After constant/poison propagation, some inputs may become unused.
struct EliminateUnusedElementwiseInputs
    : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    Block& bodyBlock = op.getBody().front();
    auto inputs = op.getInputs();

    // Find which block args are actually used.
    SmallVector<bool> argUsed(inputs.size(), false);
    for (auto [idx, arg] : llvm::enumerate(bodyBlock.getArguments())) {
      if (!arg.use_empty()) {
        argUsed[idx] = true;
      }
    }

    // Count unused args.
    unsigned numUnused = llvm::count(argUsed, false);
    if (numUnused == 0) {
      return failure();
    }
    // Note: Can produce zero inputs - shape comes from result type +
    // result_dims. FoldElementwiseYieldsCapturedValue will then fold to fill.

    // Build new inputs list.
    SmallVector<Value> newInputs;
    SmallVector<Value> newInputDims;
    SmallVector<unsigned> oldToNewArgIdx(inputs.size(),
                                         std::numeric_limits<unsigned>::max());

    unsigned newIdx = 0;
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (argUsed[i]) {
        oldToNewArgIdx[i] = newIdx++;
        newInputs.push_back(inputs[i]);
        for (Value dim : op.getOperandDynamicDims(i)) {
          newInputDims.push_back(dim);
        }
      }
    }

    // Build the new op.
    Location loc = op.getLoc();
    auto newOp =
        TileElementwiseOp::create(rewriter, loc, op.getResult().getType(),
                                  newInputs, newInputDims, op.getResultDims());

    Block& newBlock = newOp.getBody().emplaceBlock();
    for (Value input : newInputs) {
      auto tileType = cast<TileType>(input.getType());
      newBlock.addArgument(tileType.getElementType(), loc);
    }

    // Clone body with remapped args.
    rewriter.setInsertionPointToStart(&newBlock);
    IRMapping mapping;
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (oldToNewArgIdx[i] != std::numeric_limits<unsigned>::max()) {
        mapping.map(bodyBlock.getArgument(i),
                    newBlock.getArgument(oldToNewArgIdx[i]));
      }
    }

    for (Operation& bodyOp : bodyBlock.getOperations()) {
      rewriter.clone(bodyOp, mapping);
    }

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Fold elementwise that just passes through one of its inputs.
// elementwise(%a, %b) { yield %a } -> %a
struct FoldElementwisePassthrough : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    Block& bodyBlock = op.getBody().front();
    auto yieldOp = cast<TileYieldOp>(bodyBlock.getTerminator());
    Value yieldValue = yieldOp.getValue();

    // Check if yield value is a block argument.
    auto blockArg = dyn_cast<BlockArgument>(yieldValue);
    if (!blockArg || blockArg.getOwner() != &bodyBlock) {
      return failure();
    }

    // Get the corresponding input tile.
    unsigned argIdx = blockArg.getArgNumber();
    Value inputTile = op.getInputs()[argIdx];

    // Replace the elementwise with its input.
    rewriter.replaceOp(op, inputTile);
    return success();
  }
};

// Fold elementwise that yields a captured (non-block-arg) value.
// elementwise(%a, %b) { yield %captured } -> fill(%captured, alloca)
struct FoldElementwiseYieldsCapturedValue
    : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    Block& bodyBlock = op.getBody().front();

    // Check if any block arguments are used in the region.
    for (BlockArgument arg : bodyBlock.getArguments()) {
      if (!arg.use_empty()) {
        return failure();
      }
    }

    // All block args are unused - this yields a captured value.
    auto yieldOp = cast<TileYieldOp>(bodyBlock.getTerminator());
    Value yieldValue = yieldOp.getValue();

    // Clone the body ops that compute the yield value (they use captures only).
    Location loc = op.getLoc();
    IRMapping mapping;
    for (Operation& bodyOp : bodyBlock.without_terminator()) {
      rewriter.clone(bodyOp, mapping);
    }

    Value scalarResult = mapping.lookupOrDefault(yieldValue);

    // Create alloca + fill.
    auto allocaOp = TileAllocaOp::create(
        rewriter, loc, op.getResult().getType(), op.getResultDims());
    rewriter.replaceOpWithNewOp<TileFillOp>(op, op.getResult().getType(),
                                            allocaOp.getResult(),
                                            op.getResultDims(), scalarResult);
    return success();
  }
};

// Fold elementwise where all inputs are fills to a single fill.
// This is a specialized optimization that computes the scalar result directly.
struct FoldElementwiseOfFills : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    // Collect fill values from all inputs.
    SmallVector<TileFillOp> fillOps;
    for (Value input : op.getInputs()) {
      auto fillOp = input.getDefiningOp<TileFillOp>();
      if (!fillOp) {
        return failure();
      }
      // Only fold if the fill has no other uses.
      if (!fillOp->hasOneUse()) {
        return failure();
      }
      fillOps.push_back(fillOp);
    }

    // All inputs are fills. Clone the body to compute the scalar result.
    Location loc = op.getLoc();
    Region& body = op.getBody();
    Block& bodyBlock = body.front();

    // Map block arguments to fill values.
    IRMapping mapping;
    for (auto [arg, fillOp] : llvm::zip(bodyBlock.getArguments(), fillOps)) {
      mapping.map(arg, fillOp.getValue());
    }

    // Clone the body operations (except the yield) to compute scalar result.
    for (Operation& bodyOp : bodyBlock.without_terminator()) {
      rewriter.clone(bodyOp, mapping);
    }

    // Get the scalar result from the yield.
    auto yieldOp = cast<TileYieldOp>(bodyBlock.getTerminator());
    Value scalarResult = mapping.lookupOrDefault(yieldOp.getValue());

    // Create a new alloca and fill with the computed scalar.
    auto allocaOp = TileAllocaOp::create(
        rewriter, loc, op.getResult().getType(), op.getResultDims());
    rewriter.replaceOpWithNewOp<TileFillOp>(op, op.getResult().getType(),
                                            allocaOp.getResult(),
                                            op.getResultDims(), scalarResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 3: Fusion
//===----------------------------------------------------------------------===//

// Fuse chained elementwise ops when the intermediate has a single use.
// elementwise(elementwise(%a, %b) { ... }, %c) { ... } ->
// elementwise(%a, %b, %c) { fused body }
struct FuseElementwiseChain : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp outerOp,
                                PatternRewriter& rewriter) const override {
    // Look for an input that is itself an elementwise with single use.
    for (auto [outerArgIdx, input] : llvm::enumerate(outerOp.getInputs())) {
      auto innerOp = input.getDefiningOp<TileElementwiseOp>();
      if (!innerOp || !innerOp->hasOneUse()) {
        continue;
      }

      // Found a fusable inner op. Build the fused op.
      Location loc = outerOp.getLoc();

      // Collect all inputs: inner inputs + outer inputs (except the fused one).
      SmallVector<Value> newInputs;
      SmallVector<Value> newInputDims;
      SmallVector<unsigned> innerArgToNewArg;
      SmallVector<unsigned> outerArgToNewArg(
          outerOp.getInputs().size(), std::numeric_limits<unsigned>::max());

      // Add inner op inputs.
      for (auto [idx, innerInput] : llvm::enumerate(innerOp.getInputs())) {
        innerArgToNewArg.push_back(newInputs.size());
        newInputs.push_back(innerInput);
        for (Value dim : innerOp.getOperandDynamicDims(idx)) {
          newInputDims.push_back(dim);
        }
      }

      // Add outer op inputs (except the one being fused).
      for (auto [idx, outerInput] : llvm::enumerate(outerOp.getInputs())) {
        if (idx == outerArgIdx) {
          continue;  // This is the inner op result, skip.
        }
        outerArgToNewArg[idx] = newInputs.size();
        newInputs.push_back(outerInput);
        for (Value dim : outerOp.getOperandDynamicDims(idx)) {
          newInputDims.push_back(dim);
        }
      }

      // Create the fused elementwise op.
      auto fusedOp = TileElementwiseOp::create(
          rewriter, loc, outerOp.getResult().getType(), newInputs, newInputDims,
          outerOp.getResultDims());

      Block& fusedBlock = fusedOp.getBody().emplaceBlock();
      for (Value input : newInputs) {
        auto tileType = cast<TileType>(input.getType());
        fusedBlock.addArgument(tileType.getElementType(), loc);
      }

      // Clone inner body first.
      rewriter.setInsertionPointToStart(&fusedBlock);
      IRMapping innerMapping;
      Block& innerBlock = innerOp.getBody().front();
      for (auto [idx, arg] : llvm::enumerate(innerBlock.getArguments())) {
        innerMapping.map(arg, fusedBlock.getArgument(innerArgToNewArg[idx]));
      }
      for (Operation& bodyOp : innerBlock.without_terminator()) {
        rewriter.clone(bodyOp, innerMapping);
      }

      // Get the inner yield value (this replaces the outer's block arg).
      auto innerYield = cast<TileYieldOp>(innerBlock.getTerminator());
      Value innerResult = innerMapping.lookupOrDefault(innerYield.getValue());

      // Clone outer body.
      IRMapping outerMapping;
      Block& outerBlock = outerOp.getBody().front();
      for (auto [idx, arg] : llvm::enumerate(outerBlock.getArguments())) {
        if (idx == outerArgIdx) {
          // Map the fused input's block arg to the inner result.
          outerMapping.map(arg, innerResult);
        } else {
          outerMapping.map(arg, fusedBlock.getArgument(outerArgToNewArg[idx]));
        }
      }
      for (Operation& bodyOp : outerBlock.getOperations()) {
        rewriter.clone(bodyOp, outerMapping);
      }

      rewriter.replaceOp(outerOp, fusedOp.getResult());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Phase 4: Shape Optimizations
//===----------------------------------------------------------------------===//

// Sink broadcasts over elementwise when all inputs are broadcast from the
// same source shape with the same alignment. This reduces computation by
// performing elementwise on smaller tiles.
///
// elementwise(broadcast(%a), broadcast(%b)) { body } ->
// broadcast(elementwise(%a, %b) { body })
struct SinkBroadcastsOverElementwise
    : public OpRewritePattern<TileElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    auto inputs = op.getInputs();
    if (inputs.empty()) {
      return failure();
    }

    // Collect broadcast info for each input.
    SmallVector<TileBroadcastOp> broadcasts;
    TileType commonSourceType;
    TileBroadcastAlign commonAlign;

    for (auto [idx, input] : llvm::enumerate(inputs)) {
      auto broadcastOp = input.getDefiningOp<TileBroadcastOp>();
      if (!broadcastOp) {
        return failure();  // All inputs must be broadcasts.
      }

      auto sourceType = cast<TileType>(broadcastOp.getOperand().getType());

      if (idx == 0) {
        commonSourceType = sourceType;
        commonAlign = broadcastOp.getAlign();
      } else {
        // Check source shapes match.
        if (sourceType.getShapeDims() != commonSourceType.getShapeDims()) {
          return failure();
        }
        // Check alignments match.
        if (broadcastOp.getAlign() != commonAlign) {
          return failure();
        }
      }

      broadcasts.push_back(broadcastOp);
    }

    // All broadcasts have the same source shape and alignment.
    // Create the smaller elementwise op.
    Location loc = op.getLoc();

    // Collect source operands and their dims.
    SmallVector<Value> newInputs;
    SmallVector<Value> newInputDims;
    for (TileBroadcastOp bc : broadcasts) {
      newInputs.push_back(bc.getOperand());
      for (Value dim : bc.getOperandDims()) {
        newInputDims.push_back(dim);
      }
    }

    // Result dims come from the first broadcast's operand dims.
    SmallVector<Value> newResultDims(broadcasts[0].getOperandDims());

    // Create new elementwise with smaller shape.
    auto newElementwise =
        TileElementwiseOp::create(rewriter, loc, commonSourceType, newInputs,
                                  newInputDims, newResultDims);

    // Clone the region using inlineRegionBefore which handles block args.
    rewriter.inlineRegionBefore(op.getBody(), newElementwise.getBody(),
                                newElementwise.getBody().end());

    // Broadcast the result.
    rewriter.replaceOpWithNewOp<TileBroadcastOp>(
        op, op.getResult().getType(), newElementwise.getResult(), newResultDims,
        op.getResultDims(), commonAlign);
    return success();
  }
};

}  // namespace

void TileElementwiseOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                    MLIRContext* context) {
  // Patterns are designed to compose - propagation patterns create
  // opportunities for cleanup patterns. All patterns are safe to run in any
  // order.
  results.add<PropagateConstantIntoRegion, PropagatePoisonIntoRegion,
              EliminateUnusedElementwiseInputs, FoldElementwisePassthrough,
              FoldElementwiseYieldsCapturedValue, FoldElementwiseOfFills,
              FuseElementwiseChain, SinkBroadcastsOverElementwise>(context);
}

}  // namespace mlir::iree_compiler::IREE::Loom
