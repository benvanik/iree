// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomOpPatterns.h"

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// Poison Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Creates a ub.poison value and emits a structured ERR_LOOM_FOLD_0002 remark.
/// Used when slice ops read from a poison source.
Value createPoisonForSourcePoison(PatternRewriter &rewriter, Operation *op,
                                  Type resultType, Operation *sourcePoison) {
  Errors::ERR_LOOM_FOLD_0002::emit(
      op, {op->getName().getStringRef(), sourcePoison->getResult(0).getType()});

  // Create fused location for poison chain tracking.
  SmallVector<Location> locs = {op->getLoc(), sourcePoison->getLoc()};
  Location fusedLoc = FusedLoc::get(op->getContext(), locs);

  return ub::PoisonOp::create(rewriter, fusedLoc, resultType,
                              /*value=*/nullptr);
}

/// Creates a ub.poison value and emits a structured ERR_LOOM_FOLD_0003 remark.
/// Used when update ops write into a poison target.
Value createPoisonForTargetPoison(PatternRewriter &rewriter, Operation *op,
                                  Type resultType, Operation *targetPoison) {
  Errors::ERR_LOOM_FOLD_0003::emit(
      op, {op->getName().getStringRef(), targetPoison->getResult(0).getType()});

  // Create fused location for poison chain tracking.
  SmallVector<Location> locs = {op->getLoc(), targetPoison->getLoc()};
  Location fusedLoc = FusedLoc::get(op->getContext(), locs);

  return ub::PoisonOp::create(rewriter, fusedLoc, resultType,
                              /*value=*/nullptr);
}

/// Creates a ub.poison value and emits a structured ERR_LOOM_FOLD_0004 remark.
/// Used when offset + size > bound for some dimension.
Value createPoisonForOutOfBounds(PatternRewriter &rewriter, Operation *op,
                                 Type resultType, int64_t dimIndex,
                                 int64_t offset, int64_t size, int64_t bound) {
  Errors::ERR_LOOM_FOLD_0004::emit(op, {op->getName().getStringRef(), dimIndex,
                                        offset, size, offset + size, bound});

  return ub::PoisonOp::create(rewriter, op->getLoc(), resultType,
                              /*value=*/nullptr);
}

/// Creates a ub.poison value and emits a structured ERR_LOOM_FOLD_0005 remark.
/// Used when offset < 0 for some dimension.
Value createPoisonForNegativeOffset(PatternRewriter &rewriter, Operation *op,
                                    Type resultType, int64_t dimIndex,
                                    int64_t offset) {
  Errors::ERR_LOOM_FOLD_0005::emit(
      op, {op->getName().getStringRef(), dimIndex, offset});

  return ub::PoisonOp::create(rewriter, op->getLoc(), resultType,
                              /*value=*/nullptr);
}

//===----------------------------------------------------------------------===//
// Interface-Based Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Fold slice-like ops (ops with no target) where source is poison.
/// slice(poison) -> poison
///
/// CRITICAL: This pattern ONLY applies to slice-like ops (no target).
/// For update ops, source poison does NOT poison the entire result.
struct FoldSliceSourcePoison
    : public OpInterfaceRewritePattern<CopySubrangeOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CopySubrangeOpInterface op,
                                PatternRewriter &rewriter) const override {
    // Only applies to slice-like ops (no target operand).
    // For update ops, source poison does NOT poison the entire result.
    if (op.getSubrangeTarget()) {
      return failure();
    }

    Value source = op.getSubrangeSource();
    if (!source) {
      return failure();
    }

    auto sourcePoison = source.getDefiningOp<ub::PoisonOp>();
    if (!sourcePoison) {
      return failure();
    }

    Value poison = createPoisonForSourcePoison(
        rewriter, op, op->getResult(0).getType(), sourcePoison);
    rewriter.replaceOp(op, poison);
    return success();
  }
};

/// Fold update-like ops where target is poison.
/// update(tile, poison) -> poison
///
/// Writing INTO poison storage produces a poison result.
struct FoldUpdateTargetPoison
    : public OpInterfaceRewritePattern<CopySubrangeOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CopySubrangeOpInterface op,
                                PatternRewriter &rewriter) const override {
    // Only applies to ops that write to a target.
    Value target = op.getSubrangeTarget();
    if (!target) {
      return failure();
    }

    auto targetPoison = target.getDefiningOp<ub::PoisonOp>();
    if (!targetPoison) {
      return failure();
    }

    Value poison = createPoisonForTargetPoison(
        rewriter, op, op->getResult(0).getType(), targetPoison);
    rewriter.replaceOp(op, poison);
    return success();
  }
};

/// Fold ops where offset + size > bound (out-of-bounds access).
/// Works for both slice (checks source bounds) and update (checks target
/// bounds).
struct FoldCopySubrangeOutOfBounds
    : public OpInterfaceRewritePattern<CopySubrangeOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CopySubrangeOpInterface op,
                                PatternRewriter &rewriter) const override {
    // Determine which bounds to check based on op type.
    // For slice ops: check source bounds (reading from source at offset).
    // For update ops: check target bounds (writing to target at offset).
    ArrayRef<int64_t> bounds;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;

    Value target = op.getSubrangeTarget();
    if (target) {
      // Update-like op: check target bounds.
      bounds = op.getTargetShape();
      offsets = op.getTargetMixedOffsets();
      sizes = op.getTargetMixedSizes();
    } else {
      // Slice-like op: check source bounds.
      bounds = op.getSourceShape();
      offsets = op.getSourceMixedOffsets();
      sizes = op.getSourceMixedSizes();
    }

    // Check each dimension for out-of-bounds or negative offsets.
    for (size_t i = 0; i < bounds.size(); ++i) {
      int64_t bound = bounds[i];
      if (ShapedType::isDynamic(bound)) {
        continue;  // Can't prove bounds violation with dynamic dim.
      }

      // Check offset.
      auto offsetAttr = dyn_cast<Attribute>(offsets[i]);
      if (!offsetAttr) {
        continue;  // Dynamic offset, can't prove violation.
      }
      int64_t offset = cast<IntegerAttr>(offsetAttr).getInt();

      // Check for negative offset first.
      if (offset < 0) {
        Value poison = createPoisonForNegativeOffset(
            rewriter, op, op->getResult(0).getType(), i, offset);
        rewriter.replaceOp(op, poison);
        return success();
      }

      // Check size.
      auto sizeAttr = dyn_cast<Attribute>(sizes[i]);
      if (!sizeAttr) {
        continue;  // Dynamic size, can't prove violation.
      }
      int64_t size = cast<IntegerAttr>(sizeAttr).getInt();

      // Check: offset + size > bound
      if (offset + size > bound) {
        Value poison = createPoisonForOutOfBounds(
            rewriter, op, op->getResult(0).getType(), i, offset, size, bound);
        rewriter.replaceOp(op, poison);
        return success();
      }
    }

    return failure();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateCopySubrangeOpInterfacePatterns(RewritePatternSet &patterns) {
  patterns.add<FoldSliceSourcePoison, FoldUpdateTargetPoison,
               FoldCopySubrangeOutOfBounds>(patterns.getContext());
}

}  // namespace mlir::iree_compiler::IREE::Loom
