// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_YIELDOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_YIELDOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.yield
//------------------------------------------------------------------------------

def Loom_TileYieldDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/yield_op.mlir";
  let exampleRefs = ["@yield_basic"];
  let relatedOps = ["loom.tile.elementwise"];
  let tags = ["tile", "terminator", "region"];
}

def Loom_TileYieldOp : Loom_Op<"loom.tile.yield", [
  Pure,
  Terminator,
  HasParent<"TileElementwiseOp">,
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileYieldDocMetadata;

  let summary = "Yield a scalar value from a tile elementwise region";

  let description = [{
    Terminates the region of a `loom.tile.elementwise` operation, yielding
    a scalar value that becomes the output element at the current iteration
    position.

    **Semantics:**
    The yielded value is written to the output tile at the implicit iteration
    position. For a tile with shape [d0, d1, ...], the region executes once
    per element position, and `loom.tile.yield` provides the value for that
    position.

    **When to Use:**
    This operation is **required** as the terminator of every
    `loom.tile.elementwise` region. It appears exactly once at the end of
    the region's single block.

    **Constraints:**
    - The yielded value type must match the element type of the result tile
    - Must be the last operation in the elementwise region block
    - Only valid inside `loom.tile.elementwise` regions

    **Assembly Format:**
    ```mlir
    %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xf32>) {
      %neg = arith.negf %element : f32
      loom.tile.yield %neg : f32
    } -> !loom.tile<4x4xf32>
    ```
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_YIELDOP_TD
