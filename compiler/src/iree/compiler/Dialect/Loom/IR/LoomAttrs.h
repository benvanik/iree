// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMATTRS_H_
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMATTRS_H_

#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.h.inc"
// clang-format on

#endif  // IREE_COMPILER_DIALECT_LOOM_IR_LOOMATTRS_H_
