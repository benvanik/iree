// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_H_
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off: must be included after all LLVM/MLIR headers
#define GET_OP_INTERFACE_CLASSES
#include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.h.inc"
// clang-format on

#endif  // IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_H_
