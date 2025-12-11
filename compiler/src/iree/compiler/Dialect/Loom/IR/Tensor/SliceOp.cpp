// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.h"
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
// loom.tensor.slice
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mixed Offset/Size/Stride Accessors
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> TensorSliceOp::getMixedOffsets() {
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

SmallVector<OpFoldResult> TensorSliceOp::getMixedSizes() {
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

SmallVector<OpFoldResult> TensorSliceOp::getMixedStrides() {
  // loom.tensor.slice always has unit strides.
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

Value TensorSliceOp::getSubrangeSource() { return getSource(); }

ValueRange TensorSliceOp::getSourceDynamicDims() { return getSourceDims(); }

ArrayRef<int64_t> TensorSliceOp::getSourceShape() {
  return getSourceType().getShape();
}

SmallVector<OpFoldResult> TensorSliceOp::getSourceMixedOffsets() {
  return getMixedOffsets();
}

SmallVector<OpFoldResult> TensorSliceOp::getSourceMixedSizes() {
  return getMixedSizes();
}

ArrayRef<int64_t> TensorSliceOp::getTargetShape() {
  // For slice ops, target shape is the result shape (where data is written).
  return getResultType().getShape();
}

SmallVector<OpFoldResult> TensorSliceOp::getTargetMixedOffsets() {
  // Slice writes to result at offset [0,0,...].
  SmallVector<OpFoldResult> offsets;
  Builder b(getContext());
  for (int64_t i = 0; i < getRank(); ++i) {
    offsets.push_back(b.getIndexAttr(0));
  }
  return offsets;
}

SmallVector<OpFoldResult> TensorSliceOp::getTargetMixedSizes() {
  return getMixedSizes();
}

Value TensorSliceOp::getSubrangeTarget() {
  // Slice ops have no target operand - data is read from source, written to
  // result.
  return Value();
}

ValueRange TensorSliceOp::getTargetDynamicDims() {
  // No target operand for slice ops.
  return ValueRange();
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

namespace {

// Fold constant offset operands into the static_offsets attribute.
struct FoldTensorSliceConstantOffsets : public OpRewritePattern<TensorSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorSliceOp op,
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
    rewriter.replaceOpWithNewOp<TensorSliceOp>(
        op, op.getType(), op.getSource(), op.getSourceDims(), newDynamicOffsets,
        op.getResultDims(), rewriter.getDenseI64ArrayAttr(newStaticOffsets));
    return success();
  }
};

// Store-to-load forwarding: slice of update at same position.
// slice(update(%tile, %tensor[off])[off]) -> %tile
struct FoldTensorSliceOfUpdate : public OpRewritePattern<TensorSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto updateOp = op.getSource().getDefiningOp<TensorUpdateOp>();
    if (!updateOp) {
      return failure();
    }

    // Check if slice offset matches update offset.
    SmallVector<OpFoldResult> sliceOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> updateOffsets = updateOp.getMixedOffsets();
    if (!offsetsAreEqual(sliceOffsets, updateOffsets)) {
      return failure();
    }

    // Check if slice result shape matches update tile shape.
    TileType sliceResultType = op.getResultType();
    TileType updateType = updateOp.getUpdateType();
    if (sliceResultType.getShape() != updateType.getShape()) {
      return failure();
    }

    // The slice extracts exactly what was updated - return the update tile.
    rewriter.replaceOp(op, updateOp.getUpdate());
    return success();
  }
};

// Disjoint slice of update - bypass the update.
// slice(update(%tile, %tensor[off1])[off2]) -> slice(%tensor[off2])
// When the slice region doesn't overlap with the updated region.
struct FoldTensorSliceOfUpdateDisjoint
    : public OpRewritePattern<TensorSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto updateOp = op.getSource().getDefiningOp<TensorUpdateOp>();
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

    // The slice is from an unmodified region - slice from the original tensor.
    rewriter.replaceOpWithNewOp<TensorSliceOp>(
        op, op.getType(), updateOp.getTarget(), updateOp.getTargetDims(),
        op.getOffsets(), op.getResultDims(), op.getStaticOffsetsAttr());
    return success();
  }
};

}  // namespace

void TensorSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                MLIRContext* context) {
  results.add<FoldTensorSliceConstantOffsets, FoldTensorSliceOfUpdate,
              FoldTensorSliceOfUpdateDisjoint>(context);
}

}  // namespace mlir::iree_compiler::IREE::Loom
