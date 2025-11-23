// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tensor.dim
//===----------------------------------------------------------------------===//

void TensorDimOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // Try to give meaningful names based on the dimension index.
  APInt constIndex;
  if (matchPattern(getIndex(), m_ConstantInt(&constIndex))) {
    setNameFn(getResult(), "dim" + std::to_string(constIndex.getZExtValue()));
  }
}

OpFoldResult TensorDimOp::fold(FoldAdaptor adaptor) {
  auto tensorType = cast<IREE::Loom::TensorType>(getSource().getType());

  // Only fold when index is a constant.
  auto indexAttr = dyn_cast_if_present<IntegerAttr>(adaptor.getIndex());
  if (!indexAttr) {
    return {};
  }

  int64_t dimensionIndex = indexAttr.getInt();

  // Bounds check (constraint handles errors, but fold should be defensive).
  if (dimensionIndex < 0 || dimensionIndex >= tensorType.getRank()) {
    return {};
  }

  // If dimension is static, fold to constant.
  int64_t dimSize = tensorType.getDimSize(dimensionIndex);
  if (!ShapedType::isDynamic(dimSize)) {
    Builder builder(getContext());
    return builder.getIndexAttr(dimSize);
  }

  // Dynamic dimension - cannot fold directly.
  // The canonicalization pattern will handle resolution through
  // ShapeAwareOpInterface on the producing op.
  return {};
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {

// Resolves dynamic dimension queries through ShapeAwareOpInterface.
// When the source tensor is produced by an op that implements ShapeAwareOp,
// we can replace the dim query with the actual dynamic dimension value.
struct FoldTensorDimThroughShapeAware : public OpRewritePattern<TensorDimOp> {
  using OpRewritePattern<TensorDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorDimOp op,
                                PatternRewriter& rewriter) const override {
    Value source = op.getSource();
    auto shapeAwareOp = dyn_cast_if_present<Util::ShapeAwareOpInterface>(
        source.getDefiningOp());
    if (!shapeAwareOp) {
      return failure();
    }

    // We only support constant dimension indices.
    APInt index;
    if (!matchPattern(op.getIndex(), m_ConstantInt(&index))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-constant dim index unsupported");
    }

    auto tensorType = cast<IREE::Loom::TensorType>(source.getType());
    int64_t dimensionIndex = index.getZExtValue();

    // Static dimensions are handled by fold(), so we only reach here for
    // dynamic dimensions.
    assert(ShapedType::isDynamic(tensorType.getDimSize(dimensionIndex)) &&
           "static dims should be folded");

    // Get dynamic dimension through ShapeAwareOpInterface.
    unsigned dynamicDimensionIndex =
        tensorType.getDynamicDimIndex(dimensionIndex);
    auto dynamicDims = shapeAwareOp.getResultDynamicDimsFromValue(source);
    rewriter.replaceOp(op, dynamicDims[dynamicDimensionIndex]);

    return success();
  }
};

}  // namespace

void TensorDimOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.insert<FoldTensorDimThroughShapeAware>(context);
}

}  // namespace mlir::iree_compiler::IREE::Loom
