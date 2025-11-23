// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.h"

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

// clang-format off: must be included after all LLVM/MLIR headers
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.cpp.inc"
// clang-format on

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// MMAEncoding verification
//===----------------------------------------------------------------------===//

::mlir::LogicalResult MMAEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int64_t m,
    int64_t n, int64_t k, ::llvm::StringRef fragment) {
  // Verify fragment is one of the valid values.
  if (fragment != "a" && fragment != "b" && fragment != "accum") {
    return emitError() << "fragment must be one of \"a\", \"b\", or \"accum\", "
                       << "got \"" << fragment << "\"";
  }

  // Verify dimensions are positive.
  if (m <= 0) {
    return emitError() << "m dimension must be positive, got " << m;
  }
  if (n <= 0) {
    return emitError() << "n dimension must be positive, got " << n;
  }
  if (k <= 0) {
    return emitError() << "k dimension must be positive, got " << k;
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// DenseAttr custom parser/printer
//===----------------------------------------------------------------------===//

namespace {

// Helper to recursively parse a nested array literal and collect values.
// Returns the shape dimensions discovered during parsing.
// For example, [[1.0, 2.0], [3.0, 4.0]] produces shape [2, 2] and 4 values.
LogicalResult parseArrayLiteral(AsmParser& parser,
                                SmallVectorImpl<Attribute>& values,
                                SmallVectorImpl<int64_t>& shape, int depth) {
  // Check if we're at a nested array
  if (parser.parseOptionalLSquare().succeeded()) {
    // Parse first element to establish this dimension
    int64_t count = 0;
    do {
      if (parseArrayLiteral(parser, values, shape, depth + 1).failed()) {
        return failure();
      }
      ++count;
    } while (parser.parseOptionalComma().succeeded());

    if (parser.parseRSquare()) {
      return failure();
    }

    // Record this dimension's size (only on first encounter at this depth)
    if (static_cast<int64_t>(shape.size()) <= depth) {
      shape.push_back(count);
    } else if (shape[depth] != count) {
      return parser.emitError(parser.getCurrentLocation(),
                              "jagged array: inconsistent dimension sizes");
    }
    return success();
  }

  // Parse a scalar value (integer or float literal)
  Attribute value;
  if (parser.parseAttribute(value)) {
    return failure();
  }
  values.push_back(value);
  return success();
}

// Print dense elements values without the type suffix.
void printDenseValues(AsmPrinter& printer, DenseElementsAttr attr) {
  auto type = attr.getType();
  auto shape = type.getShape();

  // For scalar (rank 0), just print the value
  if (shape.empty()) {
    if (type.getElementType().isIntOrIndex()) {
      printer << attr.getSplatValue<APInt>();
    } else {
      printer << attr.getSplatValue<APFloat>();
    }
    return;
  }

  // For splat, print single value
  if (attr.isSplat()) {
    if (type.getElementType().isIntOrIndex()) {
      printer << attr.getSplatValue<APInt>();
    } else {
      printer << attr.getSplatValue<APFloat>();
    }
    return;
  }

  // For non-splat arrays, we need to print with brackets
  // This is a simplified implementation that handles common cases
  std::function<void(ArrayRef<int64_t>, size_t&)> printRecursive;
  printRecursive = [&](ArrayRef<int64_t> dims, size_t& idx) {
    if (dims.size() == 1) {
      printer << "[";
      for (int64_t i = 0; i < dims[0]; ++i) {
        if (i > 0) {
          printer << ", ";
        }
        if (type.getElementType().isIntOrIndex()) {
          printer << *(attr.value_begin<APInt>() + idx);
        } else {
          printer << *(attr.value_begin<APFloat>() + idx);
        }
        ++idx;
      }
      printer << "]";
    } else {
      printer << "[";
      for (int64_t i = 0; i < dims[0]; ++i) {
        if (i > 0) {
          printer << ", ";
        }
        printRecursive(dims.drop_front(), idx);
      }
      printer << "]";
    }
  };

  size_t idx = 0;
  printRecursive(shape, idx);
}

}  // namespace

Attribute DenseAttr::parse(AsmParser& parser, Type type) {
  // The `type` parameter is the tile type parsed by MLIR's TypedAttrInterface
  // handling from the `: !loom.tile<...>` suffix.
  auto tileType = dyn_cast_or_null<TileType>(type);
  if (!tileType) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected tile type for #loom.dense attribute");
    return {};
  }

  if (parser.parseLess()) {
    return {};
  }

  SmallVector<Attribute> values;
  SmallVector<int64_t> shape;

  // Parse the nested array literal.
  if (parseArrayLiteral(parser, values, shape, 0).failed()) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  // Get expected shape and element type from the tile type.
  ArrayRef<int64_t> tileShape = tileType.getShapeDims();
  Type elemType = tileType.getElementType();

  // Handle scalar case (no brackets, just a value) - for rank-0 tiles.
  if (shape.empty() && values.size() == 1) {
    if (!tileShape.empty()) {
      // Splat: single value broadcast to all positions.
      auto tensorType = RankedTensorType::get(tileShape, elemType);
      DenseElementsAttr denseElements;
      if (auto floatAttr = dyn_cast<FloatAttr>(values[0])) {
        APFloat value = floatAttr.getValue();
        if (auto floatType = dyn_cast<FloatType>(elemType)) {
          bool losesInfo;
          value.convert(floatType.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven, &losesInfo);
        }
        denseElements = DenseElementsAttr::get(tensorType, value);
      } else if (auto intAttr = dyn_cast<IntegerAttr>(values[0])) {
        APInt value = intAttr.getValue();
        if (auto intType = dyn_cast<IntegerType>(elemType)) {
          unsigned targetWidth = intType.getWidth();
          if (value.getBitWidth() != targetWidth) {
            value = value.getBitWidth() < targetWidth
                        ? value.sext(targetWidth)
                        : value.trunc(targetWidth);
          }
        }
        denseElements = DenseElementsAttr::get(tensorType, value);
      } else {
        parser.emitError(parser.getCurrentLocation(),
                         "expected float or integer literal");
        return {};
      }
      return DenseAttr::get(parser.getContext(), tileType, denseElements);
    }
    // Rank-0 tile (scalar).
    auto tensorType = RankedTensorType::get({}, elemType);
    DenseElementsAttr denseElements;
    if (auto floatAttr = dyn_cast<FloatAttr>(values[0])) {
      APFloat value = floatAttr.getValue();
      if (auto floatType = dyn_cast<FloatType>(elemType)) {
        bool losesInfo;
        value.convert(floatType.getFloatSemantics(),
                      APFloat::rmNearestTiesToEven, &losesInfo);
      }
      denseElements = DenseElementsAttr::get(tensorType, value);
    } else if (auto intAttr = dyn_cast<IntegerAttr>(values[0])) {
      APInt value = intAttr.getValue();
      if (auto intType = dyn_cast<IntegerType>(elemType)) {
        unsigned targetWidth = intType.getWidth();
        if (value.getBitWidth() != targetWidth) {
          value = value.getBitWidth() < targetWidth ? value.sext(targetWidth)
                                                    : value.trunc(targetWidth);
        }
      }
      denseElements = DenseElementsAttr::get(tensorType, value);
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected float or integer literal");
      return {};
    }
    return DenseAttr::get(parser.getContext(), tileType, denseElements);
  }

  // Validate shape matches tile shape.
  if (shape != tileShape) {
    auto diag = parser.emitError(parser.getCurrentLocation())
                << "dense values shape [";
    llvm::interleaveComma(shape, diag);
    diag << "] doesn't match tile shape [";
    llvm::interleaveComma(tileShape, diag);
    diag << "]";
    return {};
  }

  // Convert values to the tile's element type.
  SmallVector<Attribute> convertedValues;
  convertedValues.reserve(values.size());
  for (Attribute attrValue : values) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attrValue)) {
      if (auto floatType = dyn_cast<FloatType>(elemType)) {
        APFloat value = floatAttr.getValue();
        bool losesInfo;
        value.convert(floatType.getFloatSemantics(),
                      APFloat::rmNearestTiesToEven, &losesInfo);
        convertedValues.push_back(FloatAttr::get(floatType, value));
      } else {
        convertedValues.push_back(attrValue);
      }
    } else if (auto intAttr = dyn_cast<IntegerAttr>(attrValue)) {
      if (auto intType = dyn_cast<IntegerType>(elemType)) {
        APInt value = intAttr.getValue();
        unsigned targetWidth = intType.getWidth();
        if (value.getBitWidth() != targetWidth) {
          value = value.getBitWidth() < targetWidth ? value.sext(targetWidth)
                                                    : value.trunc(targetWidth);
        }
        convertedValues.push_back(IntegerAttr::get(intType, value));
      } else {
        convertedValues.push_back(attrValue);
      }
    } else {
      convertedValues.push_back(attrValue);
    }
  }

  // Build DenseElementsAttr with the tile's element type.
  auto tensorType = RankedTensorType::get(shape, elemType);
  auto denseElements = DenseElementsAttr::get(tensorType, convertedValues);
  if (!denseElements) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to build dense elements attribute");
    return {};
  }

  return DenseAttr::get(parser.getContext(), tileType, denseElements);
}

void DenseAttr::print(AsmPrinter& printer) const {
  printer << "<";
  printDenseValues(printer, getElements());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void LoomDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.cpp.inc"
      >();
}

}  // namespace mlir::iree_compiler::IREE::Loom
