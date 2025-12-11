// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_TENSOROPS_H_
#define IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_TENSOROPS_H_

#include "iree/compiler/Dialect/Loom/IR/LoomConstraints.h"
#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off: must be included after all LLVM/MLIR headers
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Loom/IR/Tensor/TensorOps.h.inc"
// clang-format on

namespace mlir::iree_compiler::IREE::Loom {

// Registers all tensor ops with the dialect.
void registerLoomTensorOps(LoomDialect& dialect);

}  // namespace mlir::iree_compiler::IREE::Loom

#endif  // IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_TENSOROPS_H_
