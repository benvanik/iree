// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tile.copy
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InferTypeOpInterface
//===----------------------------------------------------------------------===//

// Manually implement type inference because AllTypesMatch<["target", "result"]>
// auto-generates code that uses fixed operand indices, which breaks with
// variadic operands that shift positions.
LogicalResult TileCopyOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  // Use the Adaptor to correctly locate the target operand by name,
  // which accounts for the variable-length operand segments.
  TileCopyOp::Adaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes.push_back(adaptor.getTarget().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// TiedOpInterface
//===----------------------------------------------------------------------===//

Value TileCopyOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

std::optional<unsigned> TileCopyOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  // target is operand 3 (after source, source_dims, source_offsets)
  return {3};
}

SmallVector<int64_t> TileCopyOp::getTiedResultOperandIndices() {
  return {3};  // target
}

//===----------------------------------------------------------------------===//
// Mixed Offset/Size/Stride Accessors
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> TileCopyOp::getMixedSourceOffsets() {
  SmallVector<OpFoldResult> offsets;
  Builder b(getContext());
  ArrayRef<int64_t> staticOffsets = getStaticSourceOffsets();
  ValueRange dynamicOffsets = getSourceOffsets();
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

SmallVector<OpFoldResult> TileCopyOp::getMixedTargetOffsets() {
  SmallVector<OpFoldResult> offsets;
  Builder b(getContext());
  ArrayRef<int64_t> staticOffsets = getStaticTargetOffsets();
  ValueRange dynamicOffsets = getTargetOffsets();
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

SmallVector<OpFoldResult> TileCopyOp::getMixedSizes() {
  SmallVector<OpFoldResult> result;
  Builder b(getContext());
  ArrayRef<int64_t> staticSizes = getStaticSizes();
  ValueRange dynamicSizes = getSizes();
  unsigned dynamicIndex = 0;

  for (int64_t staticVal : staticSizes) {
    if (ShapedType::isDynamic(staticVal)) {
      result.push_back(dynamicSizes[dynamicIndex++]);
    } else {
      result.push_back(b.getIndexAttr(staticVal));
    }
  }
  return result;
}

SmallVector<OpFoldResult> TileCopyOp::getMixedStrides() {
  // loom.tile.copy always has unit strides.
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

Value TileCopyOp::getSubrangeSource() { return getSource(); }

ValueRange TileCopyOp::getSourceDynamicDims() { return getSourceDims(); }

ArrayRef<int64_t> TileCopyOp::getSourceShape() {
  return getSourceType().getShape();
}

SmallVector<OpFoldResult> TileCopyOp::getSourceMixedOffsets() {
  return getMixedSourceOffsets();
}

SmallVector<OpFoldResult> TileCopyOp::getSourceMixedSizes() {
  return getMixedSizes();
}

Value TileCopyOp::getSubrangeTarget() { return getTarget(); }

ValueRange TileCopyOp::getTargetDynamicDims() { return getTargetDims(); }

ArrayRef<int64_t> TileCopyOp::getTargetShape() {
  return getTargetType().getShape();
}

SmallVector<OpFoldResult> TileCopyOp::getTargetMixedOffsets() {
  return getMixedTargetOffsets();
}

SmallVector<OpFoldResult> TileCopyOp::getTargetMixedSizes() {
  return getMixedSizes();
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

namespace {

// Fold constant source offset operands into the static_source_offsets
// attribute.
struct FoldTileCopyConstantSourceOffsets : public OpRewritePattern<TileCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileCopyOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> newStaticOffsets(op.getStaticSourceOffsets());
    SmallVector<Value> newDynamicOffsets;
    bool changed = false;

    unsigned dynamicIndex = 0;
    for (size_t i = 0; i < newStaticOffsets.size(); ++i) {
      if (ShapedType::isDynamic(newStaticOffsets[i])) {
        Value dynValue = op.getSourceOffsets()[dynamicIndex++];
        if (auto constOp = dynValue.getDefiningOp<arith::ConstantIndexOp>()) {
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

    rewriter.replaceOpWithNewOp<TileCopyOp>(
        op, op.getType(), op.getSource(), op.getSourceDims(), newDynamicOffsets,
        op.getTarget(), op.getTargetDims(), op.getTargetOffsets(),
        op.getSizes(), rewriter.getDenseI64ArrayAttr(newStaticOffsets),
        op.getStaticTargetOffsetsAttr(), op.getStaticSizesAttr());
    return success();
  }
};

// Fold constant target offset operands into the static_target_offsets
// attribute.
struct FoldTileCopyConstantTargetOffsets : public OpRewritePattern<TileCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileCopyOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> newStaticOffsets(op.getStaticTargetOffsets());
    SmallVector<Value> newDynamicOffsets;
    bool changed = false;

    unsigned dynamicIndex = 0;
    for (size_t i = 0; i < newStaticOffsets.size(); ++i) {
      if (ShapedType::isDynamic(newStaticOffsets[i])) {
        Value dynValue = op.getTargetOffsets()[dynamicIndex++];
        if (auto constOp = dynValue.getDefiningOp<arith::ConstantIndexOp>()) {
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

    rewriter.replaceOpWithNewOp<TileCopyOp>(
        op, op.getType(), op.getSource(), op.getSourceDims(),
        op.getSourceOffsets(), op.getTarget(), op.getTargetDims(),
        newDynamicOffsets, op.getSizes(), op.getStaticSourceOffsetsAttr(),
        rewriter.getDenseI64ArrayAttr(newStaticOffsets),
        op.getStaticSizesAttr());
    return success();
  }
};

// Fold constant size operands into the static_sizes attribute.
struct FoldTileCopyConstantSizes : public OpRewritePattern<TileCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileCopyOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> newStaticSizes(op.getStaticSizes());
    SmallVector<Value> newDynamicSizes;
    bool changed = false;

    unsigned dynamicIndex = 0;
    for (size_t i = 0; i < newStaticSizes.size(); ++i) {
      if (ShapedType::isDynamic(newStaticSizes[i])) {
        Value dynValue = op.getSizes()[dynamicIndex++];
        if (auto constOp = dynValue.getDefiningOp<arith::ConstantIndexOp>()) {
          newStaticSizes[i] = constOp.value();
          changed = true;
        } else {
          newDynamicSizes.push_back(dynValue);
        }
      }
    }

    if (!changed) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TileCopyOp>(
        op, op.getType(), op.getSource(), op.getSourceDims(),
        op.getSourceOffsets(), op.getTarget(), op.getTargetDims(),
        op.getTargetOffsets(), newDynamicSizes, op.getStaticSourceOffsetsAttr(),
        op.getStaticTargetOffsetsAttr(),
        rewriter.getDenseI64ArrayAttr(newStaticSizes));
    return success();
  }
};

// Convert copy to slice when target offsets are all zero and sizes match
// target. copy(%src[src_off], %tgt[0,0,...], sizes) where sizes == target.shape
// -> update(slice(%src[src_off]) -> result_shape, %tgt[0,0,...])
// This simplifies to just: slice when target is uninitialized.
struct SimplifyCopyToSlice : public OpRewritePattern<TileCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileCopyOp op,
                                PatternRewriter& rewriter) const override {
    // Check if target offsets are all zero.
    if (!allOffsetsAreZero(op.getMixedTargetOffsets())) {
      return failure();
    }

    // Check if sizes match the target shape (full overwrite).
    TileType targetType = op.getTargetType();
    ArrayRef<int64_t> targetShape = targetType.getShape();
    SmallVector<OpFoldResult> copySizes = op.getMixedSizes();

    for (size_t i = 0; i < targetShape.size(); ++i) {
      if (ShapedType::isDynamic(targetShape[i])) {
        // Can't statically verify - would need runtime check.
        return failure();
      }
      auto sizeAttr = dyn_cast<Attribute>(copySizes[i]);
      if (!sizeAttr) {
        return failure();  // Dynamic size.
      }
      if (cast<IntegerAttr>(sizeAttr).getInt() != targetShape[i]) {
        return failure();  // Size doesn't match target dim.
      }
    }

    // The copy completely overwrites the target - it's equivalent to a slice.
    // Create a slice from source and return it (the target is irrelevant).
    rewriter.replaceOpWithNewOp<TileSliceOp>(
        op, op.getType(), op.getSource(), op.getSourceDims(),
        op.getSourceOffsets(), op.getTargetDims(),
        op.getStaticSourceOffsetsAttr());
    return success();
  }
};

// Convert copy to update when source offsets are all zero and sizes match
// source. copy(%src[0,0,...], %tgt[tgt_off], sizes) where sizes == source.shape
// -> update(%src, %tgt[tgt_off])
struct SimplifyCopyToUpdate : public OpRewritePattern<TileCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileCopyOp op,
                                PatternRewriter& rewriter) const override {
    // Check if source offsets are all zero.
    if (!allOffsetsAreZero(op.getMixedSourceOffsets())) {
      return failure();
    }

    // Check if sizes match the source shape (full read).
    TileType sourceType = op.getSourceType();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    SmallVector<OpFoldResult> copySizes = op.getMixedSizes();

    for (size_t i = 0; i < sourceShape.size(); ++i) {
      if (ShapedType::isDynamic(sourceShape[i])) {
        return failure();  // Can't statically verify.
      }
      auto sizeAttr = dyn_cast<Attribute>(copySizes[i]);
      if (!sizeAttr) {
        return failure();  // Dynamic size.
      }
      if (cast<IntegerAttr>(sizeAttr).getInt() != sourceShape[i]) {
        return failure();  // Size doesn't match source dim.
      }
    }

    // The copy reads the entire source - it's equivalent to an update.
    rewriter.replaceOpWithNewOp<TileUpdateOp>(
        op, op.getType(), op.getTarget(), op.getTargetDims(), op.getSource(),
        op.getSourceDims(), op.getTargetOffsets(),
        op.getStaticTargetOffsetsAttr());
    return success();
  }
};

}  // namespace

void TileCopyOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<FoldTileCopyConstantSourceOffsets,
              FoldTileCopyConstantTargetOffsets, FoldTileCopyConstantSizes,
              SimplifyCopyToSlice, SimplifyCopyToUpdate>(context);
}

//===----------------------------------------------------------------------===//
// Folding
//===----------------------------------------------------------------------===//

OpFoldResult TileCopyOp::fold(FoldAdaptor adaptor) {
  // Identity copy: source == target, all offsets equal, sizes match full shape.
  // This is a rare case but worth handling.
  if (getSource() == getTarget()) {
    if (offsetsAreEqual(getMixedSourceOffsets(), getMixedTargetOffsets())) {
      // Same source/target, same offsets - the copy is a no-op.
      return getTarget();
    }
  }

  return {};
}

}  // namespace mlir::iree_compiler::IREE::Loom
