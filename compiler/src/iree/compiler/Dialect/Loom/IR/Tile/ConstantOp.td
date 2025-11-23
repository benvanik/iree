// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_CONSTANTOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_CONSTANTOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomAttrs.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//------------------------------------------------------------------------------
// loom.tile.constant
//------------------------------------------------------------------------------

def Loom_TileConstantDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/constant_op.mlir";
  let exampleRefs = ["@tile_constant_dense", "@tile_constant_splat", "@tile_constant_scalar"];
  let relatedOps = ["arith.constant", "loom.tile.fill"];
  let tags = ["tile", "constant", "creation"];
}

def Loom_TileConstantOp : Loom_PureOp<"loom.tile.constant", [
  ConstantLike
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileConstantDocMetadata;

  let summary = "Create a tile from a constant attribute";

  let description = [{
    Creates a tile from a constant value. The value can be specified as:
    - A `#loom.dense<...>` attribute with explicit elements matching the tile shape
    - A scalar splat value that broadcasts to all tile positions

    **Semantics:**
    Produces a tile with compile-time known values. The values are embedded
    in the IR as attributes and can be propagated through constant folding.

    **When to Use:**
    Use `loom.tile.constant` for tiles with values known at compile time:
    - Weight matrices in neural network layers
    - Lookup tables and coefficient arrays
    - Identity matrices, zero tiles, and other fixed patterns

    For tiles filled at runtime from a scalar, use `loom.tile.fill`.
    For tiles loaded from tensor data, use `loom.tensor.slice`.

    **Dense Attribute Form:**
    Uses `#loom.dense<...>` which infers shape from the nested array structure.
    The inferred shape must match the result tile's shape.

    ```mlir
    %matrix = loom.tile.constant #loom.dense<[[1.0, 2.0], [3.0, 4.0]]> : !loom.tile<2x2xf32>
    %vector = loom.tile.constant #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
    ```

    **Scalar Splat Form:**
    A single scalar value is broadcast to fill all positions in the tile.
    This is efficient for initializing tiles with uniform values.

    ```mlir
    %zeros = loom.tile.constant 0.0 : !loom.tile<64x64xf32>
    %ones = loom.tile.constant 1 : !loom.tile<128xi32>
    ```

    **Encoding:**
    The result tile may have an encoding attribute specifying its memory
    layout. The constant values are conceptually in logical (row-major)
    order; physical layout is determined by the encoding.

    ```mlir
    %encoded = loom.tile.constant #loom.dense<[[1.0]]> : !loom.tile<1x1xf32, #loom.encoding.mma<...>>
    ```

    **Constraints:**
    - Dense attribute shape must match result tile shape
    - Element types must be compatible (checked by verifier)
    - Result tile must have static shape (no dynamic dimensions)

    **Canonicalization:**
    - Scalar splats may be converted to dense splat attributes internally
    - Constants with identical values may be deduplicated (CSE)
  }];

  let arguments = (ins
    TypedAttrInterface:$value
  );
  let results = (outs Loom_Tile:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Return true if this constant represents a splat value.
    bool isSplat();

    /// Get the splat value if this is a splat constant, otherwise nullptr.
    Attribute getSplatValue();

    /// Get the underlying DenseElementsAttr.
    DenseElementsAttr getElements();

    /// Create a splat constant with the given tile type and scalar value.
    static TileConstantOp createSplat(OpBuilder &builder, Location loc,
                                      TileType tileType, Attribute splatValue);
  }];

  let errors = [
    ERR_LOOM_CONSTANT_0001,
    ERR_LOOM_CONSTANT_0002,
  ];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_CONSTANTOP_TD
