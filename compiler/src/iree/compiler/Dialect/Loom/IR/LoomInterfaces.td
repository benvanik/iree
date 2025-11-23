// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"

//===----------------------------------------------------------------------===//
// Op interfaces
//===----------------------------------------------------------------------===//

// Placeholder for future custom op interfaces.

//===----------------------------------------------------------------------===//
// Type interfaces
//===----------------------------------------------------------------------===//

// Placeholder for future custom type interfaces.

//===----------------------------------------------------------------------===//
// Attribute interfaces
//===----------------------------------------------------------------------===//

def Loom_EncodingAttrInterface : Loom_AttrInterface<"EncodingAttrInterface"> {
  let description = [{
    Interface for tile encoding attributes that describe physical memory layout
    and access patterns for tiles.

    Encoding attributes control how tile data is laid out in memory and provide
    hints for code generation and optimization. Examples include:
    - Row-major / column-major layouts
    - Tiled layouts (e.g., 8x8 blocks)
    - Hardware-specific layouts (e.g., MMA layouts for tensor cores)
    - Padding and alignment constraints

    All attributes used as tile encodings must implement this interface.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns a human-readable description of the encoding layout",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getEncodingName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return "unknown";
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns true if this encoding is compatible with the given shape and element type",
      /*retTy=*/"bool",
      /*methodName=*/"isCompatibleWith",
      /*args=*/(ins "::llvm::ArrayRef<int64_t>":$shape, "::mlir::Type":$elementType),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Default: all encodings are compatible with all shapes/types.
        // Override to add specific constraints.
        return true;
      }]
    >
  ];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_TD
