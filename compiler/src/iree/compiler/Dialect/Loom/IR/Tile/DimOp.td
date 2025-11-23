// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TILE_DIMOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TILE_DIMOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//------------------------------------------------------------------------------
// loom.tile.dim
//------------------------------------------------------------------------------

def Loom_TileDimDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/dim_op.mlir";
  let exampleRefs = ["@tile_dim_static", "@tile_dim_dynamic", "@tile_dim_fold"];
  let relatedOps = ["loom.tensor.dim", "tensor.dim", "memref.dim"];
  let tags = ["tile", "shape", "query"];
}

def Loom_TileDimOp : Loom_PureOp<"loom.tile.dim", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Loom_TileDimIndexInBounds<"source", "index">
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileDimDocMetadata;

  let summary = "Query the size of a tile dimension";

  let description = [{
    Returns the size of the specified dimension of a tile. The dimension
    index can be a constant or an SSA value.

    **Semantics:**
    For a tile with shape [d0, d1, ..., dn-1], querying dimension `k` returns
    the value `dk`. For static dimensions, this is known at compile time.
    For dynamic dimensions (`?`), the value comes from shape propagation.

    **When to Use:**
    Use `loom.tile.dim` to query tile shapes for:
    - Computing loop bounds when iterating over tiles
    - Deriving offsets and sizes for slice/update operations
    - Passing shape information to other operations

    For querying tensor shapes (global memory), use `loom.tensor.dim` instead.

    **Folding Behavior:**
    - Static dimension: Folds to `arith.constant` when index is constant
    - Dynamic dimension: Resolves through `ShapeAwareOpInterface` when possible
    - Unknown index: Cannot fold at compile time

    **Error Handling:**
    Querying a dimension outside the valid range [0, rank) is a verification
    error when the index is statically known. Dynamic indices that are out
    of bounds result in undefined behavior.

    **Assembly Format:**
    ```mlir
    // Query static dimension (folds to arith.constant 64)
    %d0 = loom.tile.dim %tile, 0 : !loom.tile<64x?xf32>

    // Query dynamic dimension
    %d1 = loom.tile.dim %tile, 1 : !loom.tile<64x?xf32>

    // Query with SSA index
    %dim = loom.tile.dim %tile, %idx : !loom.tile<?x?xf32>
    ```
  }];

  let arguments = (ins
    Loom_Tile:$source,
    Index:$index
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $source `,` $index attr-dict `:` qualified(type($source))
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let errors = [
    ERR_LOOM_DIM_0001,
  ];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TILE_DIMOP_TD
