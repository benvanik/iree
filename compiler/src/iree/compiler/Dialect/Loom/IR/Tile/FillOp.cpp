// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tile.fill - Verifier
//===----------------------------------------------------------------------===//

LogicalResult TileFillOp::verify() {
  // Verify value type matches tile element type.
  Type valueType = getValue().getType();
  Type elementType = cast<TileType>(getTarget().getType()).getElementType();

  if (valueType != elementType) {
    Errors::ERR_LOOM_FILL_0001::emit(getOperation(),
                                     Errors::ERR_LOOM_FILL_0001::Args{
                                         .valueType = valueType,
                                         .elementType = elementType,
                                     });
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// loom.tile.fill - TiedOpInterface
//===----------------------------------------------------------------------===//

Value TileFillOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getTarget());
}

std::optional<unsigned> TileFillOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target is operand 0
}

SmallVector<int64_t> TileFillOp::getTiedResultOperandIndices() {
  return {0};  // target
}

}  // namespace mlir::iree_compiler::IREE::Loom
