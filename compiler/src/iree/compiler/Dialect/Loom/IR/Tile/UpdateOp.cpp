// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilErrors.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/Diagnostics.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tile.update
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TiedOpInterface
//===----------------------------------------------------------------------===//

Value TileUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

std::optional<unsigned> TileUpdateOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target is operand 0
}

SmallVector<int64_t> TileUpdateOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// Mixed Offset/Size/Stride Accessors
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> TileUpdateOp::getMixedOffsets() {
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

SmallVector<OpFoldResult> TileUpdateOp::getMixedSizes() {
  // Sizes are derived from the update tile type shape + update_dims.
  SmallVector<OpFoldResult> sizes;
  Builder b(getContext());
  TileType updateType = getUpdateType();
  ValueRange dynamicDims = getUpdateDims();
  unsigned dynamicIndex = 0;

  for (int64_t dim : updateType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      // Dynamic dimension - use the corresponding update_dims operand.
      sizes.push_back(dynamicDims[dynamicIndex++]);
    } else {
      // Static dimension - use an IntegerAttr.
      sizes.push_back(b.getIndexAttr(dim));
    }
  }
  return sizes;
}

SmallVector<OpFoldResult> TileUpdateOp::getMixedStrides() {
  // loom.tile.update always has unit strides.
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

// Fold update with poison target to poison.
// update(%sub, ub.poison) -> ub.poison
struct FoldUpdateOfPoisonTarget : public OpRewritePattern<TileUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileUpdateOp op,
                                PatternRewriter& rewriter) const override {
    auto targetPoison = op.getTarget().getDefiningOp<ub::PoisonOp>();
    if (!targetPoison) {
      return failure();
    }

    return replaceWithPoisonAndRemark(rewriter, op, "update target is poison",
                                      targetPoison);
  }
};

// Fold impossible updates to ub.poison.
// An update with same shape as target but provably non-zero offset is UB.
struct FoldImpossibleUpdateToPoison : public OpRewritePattern<TileUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileUpdateOp op,
                                PatternRewriter& rewriter) const override {
    TileType updateType = op.getUpdateType();
    TileType targetType = op.getTargetType();

    // Only applies when shapes are the same STATIC shape.
    // For dynamic shapes, we can't prove the dimensions are equal.
    if (updateType.getShape() != targetType.getShape()) {
      return failure();
    }

    // Must be fully static to prove UB.
    if (!updateType.hasStaticShape()) {
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
        "same-shape update with non-zero offset is undefined behavior");
  }
};

// Fold out-of-bounds updates to ub.poison.
// When offset + update_size > target_size (statically provable), it's UB.
struct FoldOutOfBoundsUpdateToPoison : public OpRewritePattern<TileUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileUpdateOp op,
                                PatternRewriter& rewriter) const override {
    TileType targetType = op.getTargetType();
    TileType updateType = op.getUpdateType();

    ArrayRef<int64_t> targetShape = targetType.getShape();
    ArrayRef<int64_t> updateShape = updateType.getShape();
    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();

    // Check each dimension for out-of-bounds access.
    for (size_t i = 0; i < targetShape.size(); ++i) {
      int64_t targetDim = targetShape[i];
      int64_t updateDim = updateShape[i];

      // Need static dimensions to prove OOB.
      if (ShapedType::isDynamic(targetDim) ||
          ShapedType::isDynamic(updateDim)) {
        continue;
      }

      // Check if offset is static.
      auto offsetAttr = dyn_cast<Attribute>(offsets[i]);
      if (!offsetAttr) {
        continue;
      }

      int64_t offset = cast<IntegerAttr>(offsetAttr).getInt();

      // Check: offset + updateDim > targetDim
      if (offset + updateDim > targetDim) {
        return replaceWithPoisonAndRemark(
            rewriter, op, "update extends beyond target bounds");
      }
    }

    return failure();
  }
};

// Fold idempotent update: update(fill(v), fill(v)) -> fill(v) (keep target).
// When updating a fill with a fill of the same value, the update is a no-op.
struct FoldIdempotentFillUpdate : public OpRewritePattern<TileUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileUpdateOp op,
                                PatternRewriter& rewriter) const override {
    // Check if both target and update are fills.
    auto targetFill = op.getTarget().getDefiningOp<TileFillOp>();
    auto updateFill = op.getUpdate().getDefiningOp<TileFillOp>();
    if (!targetFill || !updateFill) {
      return failure();
    }

    // Check if they fill with the same value.
    if (targetFill.getValue() != updateFill.getValue()) {
      return failure();
    }

    // The update is idempotent - it fills with the same value.
    // Replace with just the target fill.
    rewriter.replaceOp(op, targetFill.getResult());
    return success();
  }
};

// Fold constant offset operands into the static_offsets attribute.
struct FoldTileUpdateConstantOffsets : public OpRewritePattern<TileUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileUpdateOp op,
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
    rewriter.replaceOpWithNewOp<TileUpdateOp>(
        op, op.getType(), op.getTarget(), op.getTargetDims(), op.getUpdate(),
        op.getUpdateDims(), newDynamicOffsets,
        rewriter.getDenseI64ArrayAttr(newStaticOffsets));
    return success();
  }
};

// Fold update over update when the second update completely overwrites first.
// update(%data2, update(%data1, %target[off])[off]) -> update(%data2,
// %target[off]) When both updates are at the same offset with the same size,
// the first update is overwritten and can be skipped.
struct FoldUpdateOverUpdate : public OpRewritePattern<TileUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileUpdateOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the target is itself an update operation.
    auto innerUpdate = op.getTarget().getDefiningOp<TileUpdateOp>();
    if (!innerUpdate) {
      return failure();
    }

    // Only fold if the inner update has a single use (this op).
    // Otherwise, the inner update's result is still needed elsewhere.
    if (!innerUpdate->hasOneUse()) {
      return failure();
    }

    // Check if both updates have the same offset.
    SmallVector<OpFoldResult> outerOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> innerOffsets = innerUpdate.getMixedOffsets();
    if (!offsetsAreEqual(outerOffsets, innerOffsets)) {
      return failure();
    }

    // Check if both updates have the same size.
    // (Both update the same region, so outer overwrites inner.)
    TileType outerUpdateType = op.getUpdateType();
    TileType innerUpdateType = innerUpdate.getUpdateType();
    if (outerUpdateType.getShape() != innerUpdateType.getShape()) {
      return failure();
    }

    // The outer update completely overwrites the inner - skip the inner.
    rewriter.replaceOpWithNewOp<TileUpdateOp>(
        op, op.getType(), innerUpdate.getTarget(), innerUpdate.getTargetDims(),
        op.getUpdate(), op.getUpdateDims(), op.getOffsets(),
        op.getStaticOffsetsAttr());
    return success();
  }
};

}  // namespace

void TileUpdateOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                               MLIRContext* context) {
  results.add<FoldUpdateOfPoisonTarget, FoldImpossibleUpdateToPoison,
              FoldOutOfBoundsUpdateToPoison, FoldIdempotentFillUpdate,
              FoldTileUpdateConstantOffsets, FoldUpdateOverUpdate>(context);
}

//===----------------------------------------------------------------------===//
// Folding
//===----------------------------------------------------------------------===//

OpFoldResult TileUpdateOp::fold(FoldAdaptor adaptor) {
  TileType updateType = getUpdateType();
  TileType targetType = getTargetType();

  // Identity update: update covers entire target (same shape, offset=0).
  if (updateType.getShape() == targetType.getShape()) {
    if (allOffsetsAreZero(getMixedOffsets())) {
      // Update replaces entire target - just return the update value.
      return getUpdate();
    }
  }

  return {};
}

}  // namespace mlir::iree_compiler::IREE::Loom
