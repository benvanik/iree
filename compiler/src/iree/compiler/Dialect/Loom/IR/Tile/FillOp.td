// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_FILLOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_FILLOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Loom/IR/LoomTypes.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.fill
//------------------------------------------------------------------------------

def Loom_TileFillDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/fill_op.mlir";
  let exampleRefs = ["@tile_fill_static", "@tile_fill_dynamic"];
  let relatedOps = ["loom.tile.constant", "linalg.fill"];
  let tags = ["tile", "initialization", "in-place"];
}

def Loom_TileFillOp : Loom_PureOp<"loom.tile.fill", [
  AllTypesMatch<["target", "result"]>,
  Util_ShapeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileFillDocMetadata;

  let summary = "Fill a tile with a scalar value";

  let description = [{
    Fills all elements of the target tile with the given scalar value.
    The result is tied to the target operand (in-place mutation semantics).

    **Semantics:**
    For a tile with shape [d0, d1, ..., dn-1] and scalar value `v`:
    ```
    result[i0, i1, ...] == v  for all valid indices
    ```
    The fill overwrites all elements regardless of their previous values.

    **When to Use:**
    Use `loom.tile.fill` for uniform initialization of tiles:
    - Zero-initialization of accumulator tiles before reductions
    - Setting tiles to identity values (0 for add, 1 for mul)
    - Initializing newly allocated tiles from `loom.tile.alloca`

    For creating tiles with compile-time constant values, consider
    `loom.tile.constant` which can enable more optimizations.

    **Tied Semantics:**
    The result is tied to the target operand, enabling in-place updates
    when the target is not used after this operation. The target tile
    provides the shape; the value type must match the element type.

    **Assembly Format:**
    ```mlir
    // Static shape
    %zeros = loom.tile.fill %cst, %tile : f32 -> !loom.tile<4x4xf32>

    // Dynamic shape
    %filled = loom.tile.fill %val, %tile : f32 -> !loom.tile<?x?xf32>{%d0, %d1}
    ```
  }];

  let arguments = (ins
    Loom_Tile:$target,
    Variadic<Index>:$target_dims,
    AnyTypeOf<[AnyInteger, AnyFloat, Index]>:$value
  );
  let results = (outs Loom_Tile:$result);

  let assemblyFormat = [{
    $value `,` $target attr-dict `:`
    type($value) `->` qualified(type($result)) (`{` $target_dims^ `}`)?
  }];

  let hasVerifier = 1;

  let errors = [
    ERR_LOOM_FILL_0001,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) {
      assert(idx == 0 && "fill has only one shaped operand");
      return getTargetDims();
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      assert(idx == 0 && "fill has only one result");
      return getTargetDims();  // Result shares target's dims
    }
  }];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_FILLOP_TD
