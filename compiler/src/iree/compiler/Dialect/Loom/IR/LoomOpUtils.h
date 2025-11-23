// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMOPUTILS_H_
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMOPUTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// Offset Utilities
//===----------------------------------------------------------------------===//

/// Returns true if all OpFoldResults are constant zero.
bool allOffsetsAreZero(ArrayRef<OpFoldResult> offsets);

/// Returns true if two OpFoldResult arrays are provably equal.
/// Compares element-wise: static values must match exactly, and Value operands
/// must be the same SSA value. Returns false if equality cannot be proven.
bool offsetsAreEqual(ArrayRef<OpFoldResult> lhs, ArrayRef<OpFoldResult> rhs);

/// Returns true if two regions are provably disjoint (non-overlapping).
/// Takes offset and size arrays for both regions and checks if any dimension
/// has the regions completely separated. Returns false if disjointness cannot
/// be proven (e.g., dynamic offsets).
bool regionsAreDisjoint(ArrayRef<OpFoldResult> offset1,
                        ArrayRef<OpFoldResult> size1,
                        ArrayRef<OpFoldResult> offset2,
                        ArrayRef<OpFoldResult> size2);

//===----------------------------------------------------------------------===//
// Poison Utilities
//===----------------------------------------------------------------------===//

/// Creates a ub.poison op and emits a structured ERR_LOOM_FOLD_0001 remark.
/// Use this whenever folding an operation to poison to ensure consistent
/// diagnostics with fixHint and examples.
///
/// If `sourcePoison` is provided (the operation consumes an existing poison),
/// the locations are fused and the reason is chained.
///
/// Example:
///   Value poison = createPoisonWithRemark(rewriter, op, op.getType(),
///                                         "slice source is poison");
///   rewriter.replaceOp(op, poison);
Value createPoisonWithRemark(PatternRewriter& rewriter, Operation* op,
                             Type resultType, StringRef reason,
                             Operation* sourcePoison = nullptr);

/// Replaces an operation with ub.poison and emits structured remark.
/// Convenience wrapper that combines createPoisonWithRemark + replaceOp.
///
/// If `sourcePoison` is provided (the operation consumes an existing poison),
/// the locations are fused and the reason is chained.
///
/// Example:
///   return replaceWithPoisonAndRemark(rewriter, op,
///                                     "slice extends beyond source bounds");
LogicalResult replaceWithPoisonAndRemark(PatternRewriter& rewriter,
                                         Operation* op, StringRef reason,
                                         Operation* sourcePoison = nullptr);

}  // namespace mlir::iree_compiler::IREE::Loom

#endif  // IREE_COMPILER_DIALECT_LOOM_IR_LOOMOPUTILS_H_
