// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/LoomAttrs.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// loom.tile.constant - Parser/Printer
//===----------------------------------------------------------------------===//

// Custom parser for: #loom.dense<[[1.0, 2.0]]> : !loom.tile<2x2xf32>
// MLIR's parseAttribute() handles the `: !loom.tile<...>` suffix for
// TypedAttrInterface and passes the tile type to DenseAttr::parse().
ParseResult TileConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  Attribute valueAttr;

  if (parser.parseAttribute(valueAttr)) {
    return failure();
  }

  auto denseAttr = dyn_cast<DenseAttr>(valueAttr);
  if (!denseAttr) {
    return parser.emitError(parser.getNameLoc(),
                            "expected #loom.dense<...> attribute");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  result.addAttribute("value", valueAttr);
  result.addTypes(denseAttr.getType());
  return success();
}

void TileConstantOp::print(OpAsmPrinter& p) {
  p << ' ';
  p.printAttribute(getValue());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

//===----------------------------------------------------------------------===//
// loom.tile.constant - Verifier
//===----------------------------------------------------------------------===//

LogicalResult TileConstantOp::verify() {
  auto resultTy = cast<TileType>(getResult().getType());

  // Tiles cannot have dynamic dimensions for constants.
  if (resultTy.getNumDynamicDims() > 0) {
    return emitOpError("constant tiles must have static shape");
  }

  // Shape and element type are validated during parsing; just verify
  // the attribute's tile type matches the result type.
  auto denseAttr = dyn_cast<DenseAttr>(getValue());
  if (!denseAttr) {
    return emitOpError("expected #loom.dense<...> attribute");
  }

  if (denseAttr.getType() != resultTy) {
    return emitOpError("dense attribute type doesn't match result type");
  }

  return success();
}

bool TileConstantOp::isSplat() { return cast<DenseAttr>(getValue()).isSplat(); }

Attribute TileConstantOp::getSplatValue() {
  auto elements = getElements();
  if (elements.isSplat()) {
    return elements.getSplatValue<Attribute>();
  }
  return nullptr;
}

DenseElementsAttr TileConstantOp::getElements() {
  return cast<DenseAttr>(getValue()).getElements();
}

TileConstantOp TileConstantOp::createSplat(OpBuilder& builder, Location loc,
                                           TileType tileType,
                                           Attribute splatValue) {
  // Create a DenseElementsAttr with the splat value.
  auto tensorType =
      RankedTensorType::get(tileType.getShapeDims(), tileType.getElementType());
  DenseElementsAttr elements;
  if (auto floatAttr = dyn_cast<FloatAttr>(splatValue)) {
    elements = DenseElementsAttr::get(tensorType, floatAttr.getValue());
  } else if (auto intAttr = dyn_cast<IntegerAttr>(splatValue)) {
    elements = DenseElementsAttr::get(tensorType, intAttr.getValue());
  } else {
    llvm_unreachable("unsupported splat value type");
  }

  auto denseAttr = DenseAttr::get(builder.getContext(), tileType, elements);
  return TileConstantOp::create(builder, loc, tileType, denseAttr);
}

OpFoldResult TileConstantOp::fold(FoldAdaptor adaptor) {
  // Constants fold to themselves.
  return getValue();
}

}  // namespace mlir::iree_compiler::IREE::Loom
