// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"

// clang-format off: must be included after all LLVM/MLIR headers
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.cpp.inc"
// clang-format on

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

TensorType TensorType::get(ArrayRef<int64_t> shape, Type elementType,
                           EncodingAttrInterface encoding) {
  return Base::get(elementType.getContext(), shape, elementType, encoding);
}

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

TileType TileType::get(ArrayRef<int64_t> shape, Type elementType,
                       EncodingAttrInterface encoding) {
  return Base::get(elementType.getContext(), shape, elementType, encoding);
}

//===----------------------------------------------------------------------===//
// Custom parsing/printing for shape and type
//===----------------------------------------------------------------------===//

ParseResult parseShapeAndType(AsmParser& parser,
                              SmallVectorImpl<int64_t>& shape,
                              Type& elementType) {
  // Parse dimension list (e.g., "3x4" or "?x128").
  if (parser.parseDimensionList(shape)) {
    return failure();
  }

  // Parse element type (e.g., "f32").
  if (parser.parseType(elementType)) {
    return failure();
  }

  return success();
}

void printShapeAndType(AsmPrinter& printer, ArrayRef<int64_t> shape,
                       Type elementType) {
  // Print dimension list (e.g., "3x4x").
  printer.printDimensionList(shape);

  // Add 'x' separator between dimensions and element type if there are dims.
  if (!shape.empty()) {
    printer << "x";
  }

  // Print element type (e.g., "f32").
  printer.printType(elementType);
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void LoomDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.cpp.inc"
      >();
}

}  // namespace mlir::iree_compiler::IREE::Loom
