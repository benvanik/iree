// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.h"
#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilErrors.h"
#include "iree/compiler/Utils/Diagnostics.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tile.broadcast
//===----------------------------------------------------------------------===//

LogicalResult TileBroadcastOp::verify() {
  auto srcTy = cast<TileType>(getOperand().getType());
  auto resTy = cast<TileType>(getResult().getType());

  auto srcShape = srcTy.getShapeDims();
  auto resShape = resTy.getShapeDims();

  // Check shape compatibility based on alignment.
  int64_t offset = (getAlign() == IREE::Loom::TileBroadcastAlign::Left)
                       ? 0
                       : (resShape.size() - srcShape.size());
  for (unsigned i = 0; i < srcShape.size(); ++i) {
    int64_t srcDim = srcShape[i];
    int64_t resDim = resShape[i + offset];
    if (!ShapedType::isDynamic(srcDim) && !ShapedType::isDynamic(resDim) &&
        srcDim != resDim && srcDim != 1) {
      return emitErrorCode<Errors::ERR_LOOM_BROADCAST_0001>(
          getOperation(), i, srcDim, static_cast<unsigned>(i + offset), resDim);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

namespace {

// Fold broadcast of poison to poison.
// broadcast(ub.poison) -> ub.poison
struct FoldBroadcastOfPoison : public OpRewritePattern<TileBroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileBroadcastOp op,
                                PatternRewriter& rewriter) const override {
    auto sourcePoison = op.getOperand().getDefiningOp<ub::PoisonOp>();
    if (!sourcePoison) {
      return failure();
    }

    return replaceWithPoisonAndRemark(
        rewriter, op, "broadcast operand is poison", sourcePoison);
  }
};

// Fold broadcast of fill to a larger fill.
// broadcast(fill(val, target)) -> fill(val, larger_alloca)
// Broadcasting a filled tile is the same as filling a larger tile.
struct FoldBroadcastOfFill : public OpRewritePattern<TileBroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileBroadcastOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the operand is a fill.
    auto fillOp = op.getOperand().getDefiningOp<TileFillOp>();
    if (!fillOp) {
      return failure();
    }

    // Only fold if the fill has no other uses.
    if (!fillOp->hasOneUse()) {
      return failure();
    }

    // Create a new alloca for the result shape.
    Location loc = op.getLoc();
    auto allocaOp = TileAllocaOp::create(
        rewriter, loc, op.getResult().getType(), op.getResultDims());

    // Create a fill of the larger tile with the same value.
    rewriter.replaceOpWithNewOp<TileFillOp>(
        op, op.getResult().getType(), allocaOp.getResult(), op.getResultDims(),
        fillOp.getValue());
    return success();
  }
};

// Fold broadcast of splat constant to a larger constant.
// broadcast(constant<splat>) -> constant<splat> with larger shape
struct FoldBroadcastOfSplatConstant : public OpRewritePattern<TileBroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TileBroadcastOp op,
                                PatternRewriter& rewriter) const override {
    auto constantOp = op.getOperand().getDefiningOp<TileConstantOp>();
    if (!constantOp || !constantOp.isSplat()) {
      return failure();
    }

    // Create a new constant with the result shape and same splat value.
    auto resultTileType = cast<TileType>(op.getResult().getType());
    rewriter.replaceOp(
        op, TileConstantOp::createSplat(rewriter, op.getLoc(), resultTileType,
                                        constantOp.getSplatValue()));
    return success();
  }
};

}  // namespace

void TileBroadcastOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                  MLIRContext* context) {
  results.add<FoldBroadcastOfPoison, FoldBroadcastOfFill,
              FoldBroadcastOfSplatConstant>(context);
}

}  // namespace mlir::iree_compiler::IREE::Loom
