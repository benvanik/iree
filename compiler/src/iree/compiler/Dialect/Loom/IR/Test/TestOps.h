// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TEST_TESTOPS_H_
#define IREE_COMPILER_DIALECT_LOOM_IR_TEST_TESTOPS_H_

#include "iree/compiler/Dialect/Loom/IR/LoomConstraints.h"
#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Loom/IR/Test/TestOps.h.inc"

namespace mlir::iree_compiler::IREE::Loom {

// Registers all Test ops with the Loom dialect.
void registerLoomTestOps(LoomDialect& dialect);

}  // namespace mlir::iree_compiler::IREE::Loom

#endif  // IREE_COMPILER_DIALECT_LOOM_IR_TEST_TESTOPS_H_
