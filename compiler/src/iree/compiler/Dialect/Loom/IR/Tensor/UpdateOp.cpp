// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.h"
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
// loom.tensor.update
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TiedOpInterface
//===----------------------------------------------------------------------===//

Value TensorUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

std::optional<unsigned> TensorUpdateOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target is operand 0
}

SmallVector<int64_t> TensorUpdateOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// Mixed Offset/Size/Stride Accessors
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> TensorUpdateOp::getMixedOffsets() {
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

SmallVector<OpFoldResult> TensorUpdateOp::getMixedSizes() {
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

SmallVector<OpFoldResult> TensorUpdateOp::getMixedStrides() {
  // loom.tensor.update always has unit strides.
  SmallVector<OpFoldResult> strides;
  Builder b(getContext());
  int64_t rank = getRank();
  for (int64_t i = 0; i < rank; ++i) {
    strides.push_back(b.getIndexAttr(1));
  }
  return strides;
}

//===----------------------------------------------------------------------===//
// CopySubrangeOpInterface
//===----------------------------------------------------------------------===//

Value TensorUpdateOp::getSubrangeSource() {
  // For update ops, the "source" being copied is the update tile.
  return getUpdate();
}

ValueRange TensorUpdateOp::getSourceDynamicDims() { return getUpdateDims(); }

ArrayRef<int64_t> TensorUpdateOp::getSourceShape() {
  return getUpdateType().getShape();
}

SmallVector<OpFoldResult> TensorUpdateOp::getSourceMixedOffsets() {
  // We read the entire update tile (offsets are all zero).
  SmallVector<OpFoldResult> offsets;
  Builder b(getContext());
  for (int64_t i = 0; i < getRank(); ++i) {
    offsets.push_back(b.getIndexAttr(0));
  }
  return offsets;
}

SmallVector<OpFoldResult> TensorUpdateOp::getSourceMixedSizes() {
  return getMixedSizes();
}

Value TensorUpdateOp::getSubrangeTarget() { return getTarget(); }

ValueRange TensorUpdateOp::getTargetDynamicDims() { return getTargetDims(); }

ArrayRef<int64_t> TensorUpdateOp::getTargetShape() {
  return getTargetType().getShape();
}

SmallVector<OpFoldResult> TensorUpdateOp::getTargetMixedOffsets() {
  return getMixedOffsets();
}

SmallVector<OpFoldResult> TensorUpdateOp::getTargetMixedSizes() {
  return getMixedSizes();
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

namespace {

// Fold constant offset operands into the static_offsets attribute.
struct FoldTensorUpdateConstantOffsets
    : public OpRewritePattern<TensorUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorUpdateOp op,
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
    rewriter.replaceOpWithNewOp<TensorUpdateOp>(
        op, op.getType(), op.getTarget(), op.getTargetDims(), op.getUpdate(),
        op.getUpdateDims(), newDynamicOffsets,
        rewriter.getDenseI64ArrayAttr(newStaticOffsets));
    return success();
  }
};

// Round-trip identity: update(slice(%tensor[off]), %tensor[off]) -> %tensor
// When we slice data and immediately write it back to the same position.
struct FoldTensorUpdateOfSlice : public OpRewritePattern<TensorUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorUpdateOp op,
                                PatternRewriter& rewriter) const override {
    // Check if update tile comes from a slice.
    auto sliceOp = op.getUpdate().getDefiningOp<TensorSliceOp>();
    if (!sliceOp) {
      return failure();
    }

    // Check if the slice source is the same tensor as the update target.
    if (sliceOp.getSource() != op.getTarget()) {
      return failure();
    }

    // Check if offsets match.
    SmallVector<OpFoldResult> sliceOffsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> updateOffsets = op.getMixedOffsets();
    if (!offsetsAreEqual(sliceOffsets, updateOffsets)) {
      return failure();
    }

    // Check if sizes match (slice result shape == update tile shape).
    // They should match by type since update tile is the slice result.
    // This is a no-op round-trip: slice then update at same position.
    rewriter.replaceOp(op, op.getTarget());
    return success();
  }
};

// Dead store elimination: update over update at same position.
// update(%tile2, update(%tile1, %tensor[off])[off]) -> update(%tile2,
// %tensor[off])
struct FoldTensorUpdateOverUpdate : public OpRewritePattern<TensorUpdateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorUpdateOp op,
                                PatternRewriter& rewriter) const override {
    // Check if target is itself an update.
    auto innerUpdate = op.getTarget().getDefiningOp<TensorUpdateOp>();
    if (!innerUpdate) {
      return failure();
    }

    // Check if offsets match.
    SmallVector<OpFoldResult> outerOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> innerOffsets = innerUpdate.getMixedOffsets();
    if (!offsetsAreEqual(outerOffsets, innerOffsets)) {
      return failure();
    }

    // Check if sizes match (both update tiles have same shape).
    TileType outerUpdateType = op.getUpdateType();
    TileType innerUpdateType = innerUpdate.getUpdateType();
    if (outerUpdateType.getShape() != innerUpdateType.getShape()) {
      return failure();
    }

    // The inner update is completely overwritten - skip it.
    // Only do this if the inner update has a single use (this op).
    if (!innerUpdate->hasOneUse()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TensorUpdateOp>(
        op, op.getType(), innerUpdate.getTarget(), innerUpdate.getTargetDims(),
        op.getUpdate(), op.getUpdateDims(), op.getOffsets(),
        op.getStaticOffsetsAttr());
    return success();
  }
};

}  // namespace

void TensorUpdateOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<FoldTensorUpdateConstantOffsets, FoldTensorUpdateOfSlice,
              FoldTensorUpdateOverUpdate>(context);
}

}  // namespace mlir::iree_compiler::IREE::Loom
