// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.h"

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilErrors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

// clang-format off: must be included after all LLVM/MLIR headers

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.cpp.inc"

// Include generated tag dispatch implementations for constraint-detected
// errors.
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOpErrors.cpp.inc"

// clang-format on

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void registerLoomTensorOps(LoomDialect& dialect) {
  dialect.registerOperations<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.cpp.inc"
      >();
}

}  // namespace mlir::iree_compiler::IREE::Loom
