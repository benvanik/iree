// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"

#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.h"
#include "iree/compiler/Dialect/Loom/IR/Test/TestOps.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectImplementation.h"

//===----------------------------------------------------------------------===//
// `loom` dialect
//===----------------------------------------------------------------------===//

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/Loom/IR/LoomDialect.cpp.inc"
// clang-format on

namespace mlir::iree_compiler::IREE::Loom {

void LoomDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerLoomTensorOps(*this);
  registerLoomTileOps(*this);
  registerLoomTestOps(*this);
}

Operation* LoomDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  // Delegate to arith dialect for standard constants.
  if (auto op = arith::ConstantOp::materialize(builder, value, type, loc)) {
    return op;
  }
  return nullptr;
}

}  // namespace mlir::iree_compiler::IREE::Loom
