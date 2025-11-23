// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMTYPES_H_
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMTYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.h.inc"
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h.inc"
// clang-format on

namespace mlir::iree_compiler::IREE::Loom {

// Custom parsing/printing for shape and type.
// Format: <shape x element_type>
// Example: <16x16xf32> or <?x128xf16>
ParseResult parseShapeAndType(AsmParser& parser,
                              SmallVectorImpl<int64_t>& shape,
                              Type& elementType);

void printShapeAndType(AsmPrinter& printer, ArrayRef<int64_t> shape,
                       Type elementType);

}  // namespace mlir::iree_compiler::IREE::Loom

#endif  // IREE_COMPILER_DIALECT_LOOM_IR_LOOMTYPES_H_
