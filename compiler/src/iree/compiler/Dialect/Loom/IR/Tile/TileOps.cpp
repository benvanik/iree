// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.h"

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "iree/compiler/Dialect/Loom/IR/LoomErrors.h"
#include "iree/compiler/Dialect/Loom/IR/LoomTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilErrors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

//===----------------------------------------------------------------------===//
// custom<ElementwiseBindings>
//===----------------------------------------------------------------------===//
//
// Syntax: (%elem_a = %a : !loom.tile<...>{%d0, %d1}, ...) { ... }
//
// The custom parser handles parens, bindings, and region together because
// the `)` comes before the region `{` and we need to set up block args.
// Block arguments have element types (not tile types).
// Dynamic dims are parsed inline with each binding.

namespace mlir::iree_compiler::IREE::Loom {

static ParseResult parseElementwiseBindings(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& inputs,
    SmallVectorImpl<Type>& inputTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& inputDims, Region& body) {
  // Parse "(".
  if (failed(parser.parseLParen())) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument, 8> regionArgs;

  // Parse bindings (can be empty for zero-input elementwise).
  // Check for immediate ")" to handle empty case.
  if (failed(parser.parseOptionalRParen())) {
    do {
      // Parse: %elem_name = %tile_operand : !loom.tile<...>{%dims}
      OpAsmParser::Argument arg;
      OpAsmParser::UnresolvedOperand operand;
      Type tileType;

      // Parse block argument name (the element binding).
      if (failed(parser.parseArgument(arg))) {
        return failure();
      }

      // Parse "=".
      if (failed(parser.parseEqual())) {
        return failure();
      }

      // Parse tile operand.
      if (failed(parser.parseOperand(operand))) {
        return failure();
      }

      // Parse ":".
      if (failed(parser.parseColon())) {
        return failure();
      }

      // Parse tile type.
      if (failed(parser.parseType(tileType))) {
        return failure();
      }

      // Validate it's a TileType.
      auto tileTy = dyn_cast<TileType>(tileType);
      if (!tileTy) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected !loom.tile type");
      }

      // Parse optional dynamic dims: {%d0, %d1}.
      if (succeeded(parser.parseOptionalLBrace())) {
        if (failed(parser.parseOperandList(inputDims,
                                           AsmParser::Delimiter::None))) {
          return failure();
        }
        if (failed(parser.parseRBrace())) {
          return failure();
        }
      }

      // Set block arg type to element type (not tile type!).
      arg.type = tileTy.getElementType();

      inputs.push_back(operand);
      inputTypes.push_back(tileType);
      regionArgs.push_back(arg);

    } while (succeeded(parser.parseOptionalComma()));

    // Parse ")".
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  // Parse the region body.
  return parser.parseRegion(body, regionArgs);
}

static void printElementwiseBindings(OpAsmPrinter& p, Operation* op,
                                     ValueRange inputs, TypeRange inputTypes,
                                     ValueRange inputDims, Region& body) {
  p << "(";

  // Track dims consumed per input.
  size_t dimOffset = 0;

  // Print bindings.
  llvm::interleaveComma(
      llvm::zip_equal(inputs, inputTypes, body.getArguments()), p,
      [&](auto tuple) {
        auto input = std::get<0>(tuple);
        auto inputType = std::get<1>(tuple);
        auto arg = std::get<2>(tuple);

        // Print: %arg = %input : type
        p.printRegionArgument(arg, /*attrs=*/{}, /*omitType=*/true);
        p << " = ";
        p.printOperand(input);
        p << " : ";
        p.printType(inputType);

        // Print dynamic dims if any.
        auto tileTy = cast<TileType>(inputType);
        int64_t numDynDims = tileTy.getNumDynamicDims();
        if (numDynDims > 0) {
          p << "{";
          llvm::interleaveComma(inputDims.slice(dimOffset, numDynDims), p,
                                [&](Value dim) { p.printOperand(dim); });
          p << "}";
          dimOffset += numDynDims;
        }
      });

  p << ") ";

  // Print region body without block arguments (we print them in bindings).
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

}  // namespace mlir::iree_compiler::IREE::Loom

// clang-format off: must be included after all LLVM/MLIR headers

#include "iree/compiler/Dialect/Loom/IR/Tile/TileEnums.cpp.inc"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.cpp.inc"

// Include generated tag dispatch implementations for constraint-detected
// errors.
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOpErrors.cpp.inc"

// clang-format on

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void registerLoomTileOps(LoomDialect& dialect) {
  dialect.registerOperations<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Loom/IR/Tile/TileOps.cpp.inc"
      >();
}

}  // namespace mlir::iree_compiler::IREE::Loom
