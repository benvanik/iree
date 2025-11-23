// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/Test/TestOps.h"

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Loom/IR/Test/TestOps.cpp.inc"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void registerLoomTestOps(LoomDialect& dialect) {
  dialect.registerOperations<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Loom/IR/Test/TestOps.cpp.inc"
      >();
}

}  // namespace mlir::iree_compiler::IREE::Loom
