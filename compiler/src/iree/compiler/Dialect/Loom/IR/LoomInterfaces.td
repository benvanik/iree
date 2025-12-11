// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMINTERFACES_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"

//===----------------------------------------------------------------------===//
// Op interfaces
//===----------------------------------------------------------------------===//

def Loom_CopySubrangeOpInterface : Loom_OpInterface<"CopySubrangeOpInterface"> {
  let description = [{
    Interface for operations that copy rectangular block regions between shaped
    values. Supports both source (read) and target (write) access patterns.

    This interface is specifically scoped to **rectangular block copies** with
    source/target operands. It enables shared canonicalization patterns for
    poison folding, bounds checking, and constant offset folding across:
    - **Slice ops**: read from source at offset → result
    - **Update ops**: write source → target at offset
    - **Copy ops**: read from source at offset → write to target at offset

    For slice-only ops: target methods return null.
    For update-only ops: both source and target are valid.
    For copy ops: both source and target methods return meaningful values.

    Out of scope: Gather/Scatter (indirect indexing), strided access patterns.
    Those would use separate interfaces like `IndirectSubrangeOpInterface`.
  }];

  let cppNamespace = "::mlir::iree_compiler::IREE::Loom";

  let methods = [
    //===------------------------------------------------------------------===//
    // Source Access (for slice-like behavior)
    //===------------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Returns the source operand being read from.
        For slice ops, this is the tensor/tile being sliced.
        For update ops, this is the tile being written (update source).
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getSubrangeSource",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns dynamic dims for the source operand.}],
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSourceDynamicDims",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns the static shape of the source.}],
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getSourceShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns mixed offsets into the source.
        For slice ops, these are the read offsets.
        For update ops, these are [0,0,...] (read entire update tile).
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getSourceMixedOffsets",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns mixed sizes read from the source.
        For slice ops, these are the slice sizes.
        For update ops, these are the update tile sizes.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getSourceMixedSizes",
      /*args=*/(ins)
    >,

    //===------------------------------------------------------------------===//
    // Target Access (for update-like behavior)
    //===------------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{
        Returns the target operand being written to, or null for slice-like ops.
        For slice ops, this returns null (no target operand).
        For update ops, this is the tensor/tile being updated.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getSubrangeTarget",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns dynamic dims for the target operand (empty for slice ops).}],
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTargetDynamicDims",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns the static shape of the target.}],
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getTargetShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns mixed offsets into the target.
        For slice ops, these are [0,0,...] (no target).
        For update ops, these are the write offsets.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getTargetMixedOffsets",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns mixed sizes written to the target.
        For slice ops, these are the result sizes.
        For update ops, these are the update tile sizes.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getTargetMixedSizes",
      /*args=*/(ins)
    >,

    //===------------------------------------------------------------------===//
    // Offset Operands (for constant folding)
    //===------------------------------------------------------------------===//

    InterfaceMethod<
      /*desc=*/[{Returns dynamic offset operands (Values, not static attrs).}],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getOffsets",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns static offsets array (ShapedType::kDynamic for dynamic).}],
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getStaticOffsets",
      /*args=*/(ins)
    >,
  ];

  // Errors emitted by interface-based canonicalization patterns.
  let errors = [
    ERR_LOOM_FOLD_0002,  // Slice source poison.
    ERR_LOOM_FOLD_0003,  // Update target poison.
    ERR_LOOM_FOLD_0004,  // Out-of-bounds access.
    ERR_LOOM_FOLD_0005,  // Negative offset.
  ];
}

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
