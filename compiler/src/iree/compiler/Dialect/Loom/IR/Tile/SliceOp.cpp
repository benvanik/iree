// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilErrors.h"
#include "iree/compiler/Utils/Diagnostics.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tile.slice
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mixed Offset/Size/Stride Accessors
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> TileSliceOp::getMixedOffsets() {
  SmallVector<OpFoldResult> offsets;
  Builder b(getContext());
  ArrayRef<int64_t> staticOffsets = getStaticOffsets();
  ValueRange dynamicOffsets = getOffsets();
  unsigned dynamicIndex = 0;

  for (int64_t staticVal : staticOffsets) {
    if (ShapedType::isDynamic(staticVal)) {
      offsets.push_back(dynamicOffsets[dynamicIndex++]);
    } else {
      offsets.push_back(b.getIndexAttr(staticVal));
    }
  }
  return offsets;
}

SmallVector<OpFoldResult> TileSliceOp::getMixedSizes() {
  // Sizes are derived from the result tile type shape + result_dims.
  SmallVector<OpFoldResult> sizes;
  Builder b(getContext());
  TileType resultType = getResultType();
  ValueRange dynamicDims = getResultDims();
  unsigned dynamicIndex = 0;

  for (int64_t dim : resultType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      // Dynamic dimension - use the corresponding result_dims operand.
      sizes.push_back(dynamicDims[dynamicIndex++]);
    } else {
      // Static dimension - use an IntegerAttr.
      sizes.push_back(b.getIndexAttr(dim));
    }
  }
  return sizes;
}

SmallVector<OpFoldResult> TileSliceOp::getMixedStrides() {
  // loom.tile.slice always has unit strides.
  SmallVector<OpFoldResult> strides;
  Builder b(getContext());
  int64_t rank = getRank();
  for (int64_t i = 0; i < rank; ++i) {
    strides.push_back(b.getIndexAttr(1));
  }
  return strides;
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

namespace {

// Compose a slice of a slice into a single slice.
// slice(slice(x)[off1])[off2] -> slice(x)[off1+off2]
struct ComposeSliceOfSlice : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the source is itself a slice.
    auto sourceSlice = op.getSource().getDefiningOp<TileSliceOp>();
    if (!sourceSlice) {
      return failure();
    }

    // Get the offsets from both slices.
    SmallVector<OpFoldResult> innerOffsets = sourceSlice.getMixedOffsets();
    SmallVector<OpFoldResult> outerOffsets = op.getMixedOffsets();

    // Compose offsets: result[i] = inner[i] + outer[i]
    Location loc = op.getLoc();
    SmallVector<OpFoldResult> composedOffsets;
    SmallVector<Value> dynamicOffsets;
    SmallVector<int64_t> staticOffsets;

    for (size_t i = 0; i < innerOffsets.size(); ++i) {
      OpFoldResult inner = innerOffsets[i];
      OpFoldResult outer = outerOffsets[i];

      // Try to fold statically if both are constant.
      auto innerAttr = dyn_cast<Attribute>(inner);
      auto outerAttr = dyn_cast<Attribute>(outer);

      if (innerAttr && outerAttr) {
        // Both static - compute sum.
        int64_t innerVal = cast<IntegerAttr>(innerAttr).getInt();
        int64_t outerVal = cast<IntegerAttr>(outerAttr).getInt();
        staticOffsets.push_back(innerVal + outerVal);
      } else {
        // At least one dynamic - create arith.addi.
        staticOffsets.push_back(ShapedType::kDynamic);
        Value innerVal = innerAttr ? arith::ConstantIndexOp::create(
                                         rewriter, loc,
                                         cast<IntegerAttr>(innerAttr).getInt())
                                   : cast<Value>(inner);
        Value outerVal = outerAttr ? arith::ConstantIndexOp::create(
                                         rewriter, loc,
                                         cast<IntegerAttr>(outerAttr).getInt())
                                   : cast<Value>(outer);
        dynamicOffsets.push_back(
            arith::AddIOp::create(rewriter, loc, innerVal, outerVal));
      }
    }

    // Create the composed slice from the original source.
    rewriter.replaceOpWithNewOp<TileSliceOp>(
        op, op.getType(), sourceSlice.getSource(), sourceSlice.getSourceDims(),
        dynamicOffsets, op.getResultDims(),
        rewriter.getDenseI64ArrayAttr(staticOffsets));
    return success();
  }
};

// Fold slice of poison to poison.
// slice(ub.poison) -> ub.poison
struct FoldSliceOfPoison : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto sourcePoison = op.getSource().getDefiningOp<ub::PoisonOp>();
    if (!sourcePoison) {
      return failure();
    }

    return replaceWithPoisonAndRemark(rewriter, op, "slice source is poison",
                                      sourcePoison);
  }
};

// Fold impossible slices to ub.poison.
// A slice with same shape as source but provably non-zero offset is UB.
struct FoldImpossibleSliceToPoison : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    TileType sourceType = op.getSourceType();
    TileType resultType = op.getResultType();

    // Only applies when shapes are the same STATIC shape.
    // For dynamic shapes, we can't prove the dimensions are equal.
    if (sourceType.getShape() != resultType.getShape()) {
      return failure();
    }

    // Must be fully static to prove UB.
    if (!sourceType.hasStaticShape()) {
      return failure();
    }

    // Check if any offset is provably non-zero.
    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    bool hasNonZeroOffset = false;
    for (OpFoldResult foldResult : offsets) {
      if (auto attr = dyn_cast<Attribute>(foldResult)) {
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (intAttr && intAttr.getInt() != 0) {
          hasNonZeroOffset = true;
          break;
        }
      }
      // Dynamic offset - can't prove it's non-zero.
    }

    if (!hasNonZeroOffset) {
      return failure();
    }

    return replaceWithPoisonAndRemark(
        rewriter, op,
        "same-shape slice with non-zero offset is undefined behavior");
  }
};

// Fold out-of-bounds slices to ub.poison.
// When offset + result_size > source_size (statically provable), it's UB.
struct FoldOutOfBoundsSliceToPoison : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    TileType sourceType = op.getSourceType();
    TileType resultType = op.getResultType();

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();
    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();

    // Check each dimension for out-of-bounds access.
    for (size_t i = 0; i < sourceShape.size(); ++i) {
      int64_t sourceDim = sourceShape[i];
      int64_t resultDim = resultShape[i];

      // Need static dimensions to prove OOB.
      if (ShapedType::isDynamic(sourceDim) ||
          ShapedType::isDynamic(resultDim)) {
        continue;
      }

      // Check if offset is static.
      auto offsetAttr = dyn_cast<Attribute>(offsets[i]);
      if (!offsetAttr) {
        continue;
      }

      int64_t offset = cast<IntegerAttr>(offsetAttr).getInt();

      // Check: offset + resultDim > sourceDim
      if (offset + resultDim > sourceDim) {
        return replaceWithPoisonAndRemark(rewriter, op,
                                          "slice extends beyond source bounds");
      }
    }

    return failure();
  }
};

// Fold slice of fill to a smaller fill.
// slice(fill(val, target)) -> fill(val, alloca(slice_shape))
// This eliminates the larger fill and allocation when only a slice is needed.
struct FoldSliceOfFill : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the source is a fill.
    auto fillOp = op.getSource().getDefiningOp<TileFillOp>();
    if (!fillOp) {
      return failure();
    }

    // Only fold if the fill has no other uses - otherwise we'd be duplicating
    // the fill work (since the original fill would still need to happen).
    if (!fillOp->hasOneUse()) {
      return failure();
    }

    // Create a new alloca for the slice shape.
    Location loc = op.getLoc();
    auto allocaOp = TileAllocaOp::create(rewriter, loc, op.getResultType(),
                                         op.getResultDims());

    // Create a fill of the smaller tile with the same value.
    rewriter.replaceOpWithNewOp<TileFillOp>(
        op, op.getResultType(), allocaOp.getResult(), op.getResultDims(),
        fillOp.getValue());
    return success();
  }
};

// Fold constant offset operands into the static_offsets attribute.
struct FoldTileSliceConstantOffsets : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> newStaticOffsets(op.getStaticOffsets());
    SmallVector<Value> newDynamicOffsets;
    bool changed = false;

    unsigned dynamicIndex = 0;
    for (size_t i = 0; i < newStaticOffsets.size(); ++i) {
      if (ShapedType::isDynamic(newStaticOffsets[i])) {
        Value dynValue = op.getOffsets()[dynamicIndex++];
        if (auto constOp = dynValue.getDefiningOp<arith::ConstantIndexOp>()) {
          // Convert dynamic to static.
          newStaticOffsets[i] = constOp.value();
          changed = true;
        } else {
          newDynamicOffsets.push_back(dynValue);
        }
      }
    }

    if (!changed) {
      return failure();
    }

    // Create new op with updated attributes.
    rewriter.replaceOpWithNewOp<TileSliceOp>(
        op, op.getType(), op.getSource(), op.getSourceDims(), newDynamicOffsets,
        op.getResultDims(), rewriter.getDenseI64ArrayAttr(newStaticOffsets));
    return success();
  }
};

// Fold slice of update when the slice extracts exactly what was updated.
// slice(update(%data, %target[off])[off]) -> %data
// This pattern applies when the slice offset matches the update offset and
// the slice size matches the update size exactly.
struct FoldSliceOfUpdate : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Check if source is an update operation.
    auto updateOp = op.getSource().getDefiningOp<TileUpdateOp>();
    if (!updateOp) {
      return failure();
    }

    // Check if slice offset matches update offset.
    SmallVector<OpFoldResult> sliceOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> updateOffsets = updateOp.getMixedOffsets();
    if (!offsetsAreEqual(sliceOffsets, updateOffsets)) {
      return failure();
    }

    // Check if slice size matches update size (the update value's shape).
    // Slice result shape == update value's shape.
    TileType sliceResultType = op.getResultType();
    TileType updateType = updateOp.getUpdateType();
    if (sliceResultType.getShape() != updateType.getShape()) {
      return failure();
    }

    // For dynamic dimensions, also need to match the dynamic dims.
    // If shapes match statically, dynamic dims should be compatible.
    // TODO: Add dynamic dim comparison if needed.

    // The slice extracts exactly what was updated - return the update value.
    rewriter.replaceOp(op, updateOp.getUpdate());
    return success();
  }
};

// Fold slice of update when the slice region is disjoint from the update.
// slice(update(%data, %target[off1])[off2]) -> slice(%target[off2])
// When the slice region doesn't overlap with the updated region, we can
// slice directly from the original target.
struct FoldSliceOfUpdateDisjoint : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Check if source is an update operation.
    auto updateOp = op.getSource().getDefiningOp<TileUpdateOp>();
    if (!updateOp) {
      return failure();
    }

    // Get slice region: offset and size.
    SmallVector<OpFoldResult> sliceOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> sliceSizes = op.getMixedSizes();

    // Get update region: offset and size.
    SmallVector<OpFoldResult> updateOffsets = updateOp.getMixedOffsets();
    SmallVector<OpFoldResult> updateSizes = updateOp.getMixedSizes();

    // Check if regions are disjoint.
    if (!regionsAreDisjoint(sliceOffsets, sliceSizes, updateOffsets,
                            updateSizes)) {
      return failure();
    }

    // The slice is from an unmodified region - slice from the original target.
    rewriter.replaceOpWithNewOp<TileSliceOp>(
        op, op.getType(), updateOp.getTarget(), updateOp.getTargetDims(),
        op.getOffsets(), op.getResultDims(), op.getStaticOffsetsAttr());
    return success();
  }
};

// Fold slice of broadcast when slicing only along broadcast dimensions.
// slice(broadcast<right>(x)[off]) -> broadcast<right>(x) with smaller result
// When the slice only affects the broadcast-added dimensions (not the original
// source dimensions), we can broadcast directly to the smaller result shape.
struct FoldSliceOfBroadcast : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the source is a broadcast.
    auto broadcastOp = op.getSource().getDefiningOp<TileBroadcastOp>();
    if (!broadcastOp) {
      return failure();
    }

    TileType broadcastSrcType =
        cast<TileType>(broadcastOp.getOperand().getType());
    TileType broadcastResType =
        cast<TileType>(broadcastOp.getResult().getType());
    TileType sliceResType = op.getResultType();

    ArrayRef<int64_t> srcShape = broadcastSrcType.getShape();
    ArrayRef<int64_t> broadcastShape = broadcastResType.getShape();
    ArrayRef<int64_t> sliceShape = sliceResType.getShape();

    int64_t srcRank = srcShape.size();
    int64_t broadcastRank = broadcastShape.size();
    int64_t sliceRank = sliceShape.size();

    // Slice must preserve rank (we don't handle rank-reducing slices here).
    if (sliceRank != broadcastRank) {
      return failure();
    }

    // Determine which dimensions are from the original source vs
    // broadcast-added. For right alignment: source dims are at indices
    // [broadcastRank - srcRank, broadcastRank) For left alignment: source dims
    // are at indices [0, srcRank)
    int64_t srcStartIdx = (broadcastOp.getAlign() == TileBroadcastAlign::Right)
                              ? (broadcastRank - srcRank)
                              : 0;

    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = op.getMixedSizes();

    // Check constraints:
    // 1. For source dimensions: offset must be 0 and size must match source dim
    // 2. For broadcast-added dimensions: any offset/size is fine (we're slicing
    // replicated data)
    for (int64_t i = 0; i < broadcastRank; ++i) {
      bool isSourceDim = (i >= srcStartIdx && i < srcStartIdx + srcRank);

      if (isSourceDim) {
        int64_t srcDimIdx = i - srcStartIdx;
        int64_t srcDim = srcShape[srcDimIdx];

        // For source dimensions, we need offset=0 and size=srcDim.
        // (We could be more sophisticated and slice the source too, but that's
        // more complex.)
        auto offsetAttr = dyn_cast<Attribute>(offsets[i]);
        if (!offsetAttr) {
          return failure();  // Dynamic offset in source dim - can't prove it's
                             // 0.
        }
        if (cast<IntegerAttr>(offsetAttr).getInt() != 0) {
          return failure();  // Non-zero offset in source dim.
        }

        auto sizeAttr = dyn_cast<Attribute>(sizes[i]);
        if (!sizeAttr) {
          return failure();  // Dynamic size in source dim.
        }
        if (cast<IntegerAttr>(sizeAttr).getInt() != srcDim) {
          return failure();  // Size doesn't match source dim.
        }
      }
      // For broadcast-added dimensions, any slice is fine.
    }

    // All source dimensions pass through unchanged - we can broadcast directly
    // to the slice result shape.
    rewriter.replaceOpWithNewOp<TileBroadcastOp>(
        op, sliceResType, broadcastOp.getOperand(),
        broadcastOp.getOperandDims(), op.getResultDims(),
        broadcastOp.getAlignAttr());
    return success();
  }
};

// Push slice through elementwise to reduce computation size.
// slice(elementwise(%a, %b) { body })[off] ->
// elementwise(slice(%a)[off], slice(%b)[off]) { body }
// This reduces the elementwise computation to only the needed region.
struct FoldSliceOfElementwise : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the source is an elementwise operation.
    auto elementwiseOp = op.getSource().getDefiningOp<TileElementwiseOp>();
    if (!elementwiseOp) {
      return failure();
    }

    // Only fold if the elementwise has a single use - otherwise we'd be
    // duplicating slices and potentially increasing code size.
    if (!elementwiseOp->hasOneUse()) {
      return failure();
    }

    Location loc = op.getLoc();

    // The result type of the slice becomes the type of all sliced inputs
    // and the result of the new elementwise.
    TileType sliceResultType = op.getResultType();

    // Create slices of all elementwise inputs with the same offsets.
    SmallVector<Value> slicedInputs;
    SmallVector<Value> slicedInputDims;
    for (auto [idx, input] : llvm::enumerate(elementwiseOp.getInputs())) {
      // Get the source dims for this input (needed by the slice op).
      // All inputs have the same shape, so we can use the elementwise's
      // getOperandDynamicDims.
      auto inputSourceDims = elementwiseOp.getOperandDynamicDims(idx);

      // Create a slice of this input with the same offsets as the outer slice.
      auto sliceOp = TileSliceOp::create(
          rewriter, loc, sliceResultType, input, inputSourceDims,
          op.getOffsets(), op.getResultDims(), op.getStaticOffsetsAttr());
      slicedInputs.push_back(sliceOp.getResult());

      // The sliced inputs have the same dynamic dims as the slice result.
      for (Value dim : op.getResultDims()) {
        slicedInputDims.push_back(dim);
      }
    }

    // Create a new elementwise with the sliced inputs.
    // Result dims are the slice's result dims.
    auto newElementwise =
        TileElementwiseOp::create(rewriter, loc, sliceResultType, slicedInputs,
                                  slicedInputDims, op.getResultDims());

    // Move the region from the old elementwise to the new one.
    rewriter.inlineRegionBefore(elementwiseOp.getBody(),
                                newElementwise.getBody(),
                                newElementwise.getBody().end());

    rewriter.replaceOp(op, newElementwise.getResult());
    return success();
  }
};

// Compose tile.slice of tensor.slice into a single tensor.slice.
// tile.slice(tensor.slice(%tensor[off1])[off2]) ->
// tensor.slice(%tensor[off1+off2]) This crosses the tensor/tile boundary to
// eliminate the intermediate tile.
struct ComposeTileSliceOfTensorSlice : public OpRewritePattern<TileSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Check if source is a tensor.slice.
    auto tensorSlice = op.getSource().getDefiningOp<TensorSliceOp>();
    if (!tensorSlice) {
      return failure();
    }

    // Get offsets from both slices.
    SmallVector<OpFoldResult> tensorOffsets = tensorSlice.getMixedOffsets();
    SmallVector<OpFoldResult> tileOffsets = op.getMixedOffsets();

    // Compose offsets: result[i] = tensor[i] + tile[i]
    SmallVector<int64_t> newStaticOffsets;
    SmallVector<Value> newDynamicOffsets;
    Location loc = op.getLoc();

    for (size_t i = 0; i < tensorOffsets.size(); ++i) {
      auto tensorOffAttr = dyn_cast<Attribute>(tensorOffsets[i]);
      auto tileOffAttr = dyn_cast<Attribute>(tileOffsets[i]);

      if (tensorOffAttr && tileOffAttr) {
        // Both static - compute sum statically.
        int64_t sum = cast<IntegerAttr>(tensorOffAttr).getInt() +
                      cast<IntegerAttr>(tileOffAttr).getInt();
        newStaticOffsets.push_back(sum);
      } else {
        // At least one dynamic - compute at runtime.
        newStaticOffsets.push_back(ShapedType::kDynamic);
        Value tensorOff =
            tensorOffAttr
                ? arith::ConstantIndexOp::create(
                      rewriter, loc, cast<IntegerAttr>(tensorOffAttr).getInt())
                : cast<Value>(tensorOffsets[i]);
        Value tileOff =
            tileOffAttr
                ? arith::ConstantIndexOp::create(
                      rewriter, loc, cast<IntegerAttr>(tileOffAttr).getInt())
                : cast<Value>(tileOffsets[i]);
        Value sum = arith::AddIOp::create(rewriter, loc, tensorOff, tileOff);
        newDynamicOffsets.push_back(sum);
      }
    }

    // Create a single tensor.slice with composed offsets and tile.slice result
    // type.
    rewriter.replaceOpWithNewOp<TensorSliceOp>(
        op, op.getType(), tensorSlice.getSource(), tensorSlice.getSourceDims(),
        newDynamicOffsets, op.getResultDims(),
        rewriter.getDenseI64ArrayAttr(newStaticOffsets));
    return success();
  }
};

}  // namespace

void TileSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<ComposeSliceOfSlice, FoldSliceOfPoison, FoldSliceOfFill,
              FoldImpossibleSliceToPoison, FoldOutOfBoundsSliceToPoison,
              FoldTileSliceConstantOffsets, FoldSliceOfUpdate,
              FoldSliceOfUpdateDisjoint, FoldSliceOfBroadcast,
              FoldSliceOfElementwise, ComposeTileSliceOfTensorSlice>(context);
}

//===----------------------------------------------------------------------===//
// Folding
//===----------------------------------------------------------------------===//

OpFoldResult TileSliceOp::fold(FoldAdaptor adaptor) {
  TileType sourceType = getSourceType();
  TileType resultType = getResultType();

  // Identity slice: same shape and all offsets are zero.
  if (sourceType.getShape() == resultType.getShape()) {
    if (allOffsetsAreZero(getMixedOffsets())) {
      // Identity - return source directly.
      return getSource();
    }
  }

  return {};
}

}  // namespace mlir::iree_compiler::IREE::Loom
