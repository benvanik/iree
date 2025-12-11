// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomOpUtils.h"

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Attributes.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// Offset Utilities
//===----------------------------------------------------------------------===//

bool allOffsetsAreZero(ArrayRef<OpFoldResult> offsets) {
  return llvm::all_of(offsets, [](OpFoldResult foldResult) {
    if (auto attr = dyn_cast<Attribute>(foldResult)) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      return intAttr && intAttr.getInt() == 0;
    }
    return false;  // Dynamic offset, can't prove it's 0.
  });
}

bool offsetsAreEqual(ArrayRef<OpFoldResult> lhs, ArrayRef<OpFoldResult> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    // Both must be the same kind (both Attribute or both Value).
    auto lhsAttr = dyn_cast<Attribute>(lhs[i]);
    auto rhsAttr = dyn_cast<Attribute>(rhs[i]);

    if (lhsAttr && rhsAttr) {
      // Both static - compare values.
      auto lhsInt = dyn_cast<IntegerAttr>(lhsAttr);
      auto rhsInt = dyn_cast<IntegerAttr>(rhsAttr);
      if (!lhsInt || !rhsInt || lhsInt.getInt() != rhsInt.getInt()) {
        return false;
      }
    } else if (!lhsAttr && !rhsAttr) {
      // Both dynamic - must be same SSA value.
      if (cast<Value>(lhs[i]) != cast<Value>(rhs[i])) {
        return false;
      }
    } else {
      // One static, one dynamic - can't prove equal.
      return false;
    }
  }
  return true;
}

bool regionsAreDisjoint(ArrayRef<OpFoldResult> offset1,
                        ArrayRef<OpFoldResult> size1,
                        ArrayRef<OpFoldResult> offset2,
                        ArrayRef<OpFoldResult> size2) {
  if (offset1.size() != offset2.size() || offset1.size() != size1.size() ||
      offset1.size() != size2.size()) {
    return false;
  }

  // Regions are disjoint if ANY dimension is completely separated.
  // For dimension i: region1 ends before region2 starts, OR vice versa.
  // i.e., offset1[i] + size1[i] <= offset2[i]  OR  offset2[i] + size2[i] <=
  // offset1[i]
  for (size_t i = 0; i < offset1.size(); ++i) {
    // Need all four to be static to prove disjointness.
    auto offset1Attr = dyn_cast<Attribute>(offset1[i]);
    auto offset2Attr = dyn_cast<Attribute>(offset2[i]);
    auto size1Attr = dyn_cast<Attribute>(size1[i]);
    auto size2Attr = dyn_cast<Attribute>(size2[i]);

    if (!offset1Attr || !offset2Attr || !size1Attr || !size2Attr) {
      continue;  // Can't prove this dimension, try others.
    }

    int64_t offset1Value = cast<IntegerAttr>(offset1Attr).getInt();
    int64_t offset2Value = cast<IntegerAttr>(offset2Attr).getInt();
    int64_t size1Value = cast<IntegerAttr>(size1Attr).getInt();
    int64_t size2Value = cast<IntegerAttr>(size2Attr).getInt();

    // Region 1: [offset1Value, offset1Value + size1Value)
    // Region 2: [offset2Value, offset2Value + size2Value)
    // Disjoint if: offset1Value + size1Value <= offset2Value
    //          OR  offset2Value + size2Value <= offset1Value
    if (offset1Value + size1Value <= offset2Value ||
        offset2Value + size2Value <= offset1Value) {
      return true;  // Definitely disjoint in this dimension.
    }
  }

  return false;  // Couldn't prove disjoint.
}

//===----------------------------------------------------------------------===//
// Poison Utilities
//===----------------------------------------------------------------------===//

// Extract the poison reason chain from a FusedLoc's metadata if present.
// Returns empty string if no reason is stored.
static std::string extractPoisonReasonFromLoc(Location loc) {
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (auto metadata = fusedLoc.getMetadata()) {
      if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
        return strAttr.getValue().str();
      }
    }
    // Recursively check nested locations for chained reasons.
    for (Location nested : fusedLoc.getLocations()) {
      std::string nestedReason = extractPoisonReasonFromLoc(nested);
      if (!nestedReason.empty()) {
        return nestedReason;
      }
    }
  }
  return "";
}

// Flatten all locations from a potentially nested FusedLoc into a list.
static void flattenLocations(Location loc, SmallVectorImpl<Location>& result) {
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location nested : fusedLoc.getLocations()) {
      flattenLocations(nested, result);
    }
  } else {
    result.push_back(loc);
  }
}

Value createPoisonWithRemark(PatternRewriter& rewriter, Operation* op,
                             Type resultType, StringRef reason,
                             Operation* sourcePoison) {
  MLIRContext* ctx = op->getContext();

  // Build the full reason chain.
  // Format: "reason1 <- reason2 <- reason3" (newest first)
  std::string fullReason(reason);
  if (sourcePoison) {
    // Try to extract the previous reason chain from the source poison's loc.
    std::string previousReason =
        extractPoisonReasonFromLoc(sourcePoison->getLoc());
    if (!previousReason.empty()) {
      fullReason += " <- ";
      fullReason += previousReason;
    }
  }

  // Build the fused location chain.
  // Flatten any existing chains to avoid deeply nested FusedLocs.
  SmallVector<Location> allLocs;
  allLocs.push_back(op->getLoc());
  if (sourcePoison) {
    flattenLocations(sourcePoison->getLoc(), allLocs);
  }

  // Create FusedLoc with the reason chain stored as StringAttr metadata.
  // This allows future poison propagation to extract and extend the chain.
  Location fusedLoc =
      FusedLoc::get(ctx, allLocs, StringAttr::get(ctx, fullReason));

  // Emit structured remark with fixHint and examples.
  Errors::ERR_LOOM_FOLD_0001::emit(op,
                                   {op->getName().getStringRef(), fullReason});

  // Create the poison value at the fused location with reason metadata.
  return ub::PoisonOp::create(rewriter, fusedLoc, resultType,
                              /*value=*/nullptr);
}

LogicalResult replaceWithPoisonAndRemark(PatternRewriter& rewriter,
                                         Operation* op, StringRef reason,
                                         Operation* sourcePoison) {
  assert(op->getNumResults() == 1 && "expected single-result op");
  Value poison = createPoisonWithRemark(
      rewriter, op, op->getResult(0).getType(), reason, sourcePoison);
  rewriter.replaceOp(op, poison);
  return success();
}

}  // namespace mlir::iree_compiler::IREE::Loom
