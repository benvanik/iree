// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMATTRS_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMATTRS_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Encoding attributes
//===----------------------------------------------------------------------===//

// TODO(benvanik): move to an Encoding/ group (with encoding ops, if we have them).

def Loom_EncodingRowMajorAttr : Loom_AttrDef<"loom.encoding.row_major", "RowMajorEncoding"> {
  let summary = "Row-major tile encoding";
  let description = [{
    Represents a row-major (C-style) tile layout. This is the default encoding
    for most tile operations and maps to contiguous memory access patterns.

    Optional alignment parameter specifies minimum alignment in bytes for the
    tile's base address.

    Examples:
      #loom.encoding.row
      #loom.encoding.row<16>  // 16-byte alignment
  }];

  let parameters = (ins
    OptionalParameter<"IntegerAttr">:$alignment
  );

  let assemblyFormat = "(`<` $alignment^ `>`)?";
}

def Loom_EncodingMMAAttr : Loom_AttrDef<"loom.encoding.mma", "MMAEncoding"> {
  let summary = "MMA (matrix multiply-accumulate) tile encoding";
  let description = [{
    Represents a tile encoding optimized for matrix multiply-accumulate
    operations on GPUs (e.g., Tensor Cores, WMMA, MMA).

    Parameters specify the MMA shape (m x n x k) and which fragment role
    this tile plays:
    - "a": Left-hand matrix (M x K)
    - "b": Right-hand matrix (K x N)
    - "accum": Accumulator/result matrix (M x N)

    The encoding determines how data is laid out for optimal MMA execution,
    which may involve swizzling, padding, or other transformations.

    Examples:
      #loom.encoding.mma<m=16, n=16, k=16, "a">      // LHS for 16x16x16 MMA
      #loom.encoding.mma<m=16, n=8, k=16, "b">       // RHS for 16x8x16 MMA
      #loom.encoding.mma<m=16, n=16, k=16, "accum">  // Accumulator for 16x16x16
  }];

  let parameters = (ins
    "int64_t":$m,
    "int64_t":$n,
    "int64_t":$k,
    StringRefParameter<>:$fragment  // "a", "b", "accum"
  );

  let assemblyFormat = "`<` `m` `=` $m `,` `n` `=` $n `,` `k` `=` $k `,` $fragment `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // Verify fragment is one of the valid values
    ::mlir::LogicalResult verify(function_ref<::mlir::InFlightDiagnostic()> emitError) const;
  }];
}

//===----------------------------------------------------------------------===//
// Dense elements attribute
//===----------------------------------------------------------------------===//

def Loom_DenseAttr : Loom_AttrDef<"loom.dense", "Dense", [TypedAttrInterface]> {
  let summary = "Dense elements attribute for tile constants";
  let description = [{
    A dense elements attribute that stores values for tile constants.
    The attribute's type (returned by `getType()`) is the tile type, NOT a
    tensor type. This allows MLIR's TypedAttrInterface handling to naturally
    parse `#loom.dense<...> : !loom.tile<...>` syntax.

    Internally stores a `DenseElementsAttr` to leverage MLIR's optimized
    dense storage and iteration infrastructure.

    Examples:
      #loom.dense<[[1.0, 2.0], [3.0, 4.0]]> : !loom.tile<2x2xf32>
      #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
      #loom.dense<42.0> : !loom.tile<f32>
  }];

  let parameters = (ins
    "TileType":$tileType,          // The tile type (returned by getType())
    "DenseElementsAttr":$elements  // Raw values storage
  );

  // Custom parsing to handle array literals
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the type - returns the TILE type for TypedAttrInterface.
    Type getType() const { return getTileType(); }

    /// Get the shape of the tile.
    ArrayRef<int64_t> getShape() const {
      return getTileType().getShapeDims();
    }

    /// Get the element type.
    Type getElementType() const {
      return getTileType().getElementType();
    }

    /// Check if this is a splat (all elements are the same).
    bool isSplat() const { return getElements().isSplat(); }
  }];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_LOOMATTRS_TD
