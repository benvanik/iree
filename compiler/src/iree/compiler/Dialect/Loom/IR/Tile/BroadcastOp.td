// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_BROADCASTOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_BROADCASTOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.broadcast
//------------------------------------------------------------------------------

def Loom_TileBroadcastDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/broadcast_op.mlir";
  let exampleRefs = ["@tile_broadcast_right_align", "@tile_broadcast_left_align", "@tile_broadcast_unit_dim"];
  let relatedOps = ["loom.tile.reshape", "tensor.expand_shape"];
  let tags = ["tile", "shape-transformation", "broadcasting"];
}

def Loom_TileBroadcastOp : Loom_PureOp<"loom.tile.broadcast", [
  Util_ShapeAwareOp,
  AttrSizedOperandSegments,
  Util_RankAtLeast<"result", "operand">,
  Loom_SameElementType<["operand", "result"]>,
  Loom_SameEncoding<["operand", "result"]>
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileBroadcastDocMetadata;

  let summary = "Broadcast tile to higher rank by prepending/appending dimensions";

  let description = [{
    Broadcasts a tile to a higher-rank tile by adding dimensions. The `align`
    attribute controls whether source dimensions align to the left (leading positions,
    prepending new dimensions to the result) or right (trailing positions, appending
    new dimensions to the result).

    **Semantics:**
    - Source rank must be ≤ result rank (verified by trait)
    - Source dimensions align to left (indices 0..N-1) or right (indices M-N..M-1) based on `align`
    - Each source dimension must either be 1 (broadcast-compatible) or exactly match the corresponding result dimension
    - Element type and encoding are preserved from source to result (verified by trait)
    - The operation is pure with no side effects

    **Alignment Modes:**
    - `left`: Source dimensions occupy leading positions in result shape
      - New dimensions are appended to the right
      - Example: `<4x8>` → `<4x8x2>` (adds trailing dimension)
    - `right` (default): Source dimensions occupy trailing positions in result shape
      - New dimensions are prepended to the left
      - Example: `<4x8>` → `<2x4x8>` (adds leading dimension)

    **Dimension Size Broadcasting:**
    Dimensions of size 1 in the source tile are broadcast (replicated) to match
    the corresponding result dimension size:
    - `<1x8>` → `<4x8>` (first dimension broadcasted from 1 to 4)
    - `<4x1>` → `<4x8>` (second dimension broadcasted from 1 to 8)

    **Encoding Preservation:**
    The tile encoding (memory layout) is preserved during broadcast. Implementations
    may optimize broadcasts by reusing memory when possible (e.g., for size-1 dimensions).

    **When to Use:**
    Use `loom.tile.broadcast` to expand tile rank or replicate data:
    - Adding batch dimensions to tiles for batched operations
    - Broadcasting operands before element-wise operations
    - Expanding tiles for replication patterns

    **Canonicalization:**
    - Identity broadcasts (same shape) are eliminated
    - Chains of broadcasts are fused into single operations
    - Broadcasts of constant splats generate new constants with result shape
  }];

  let arguments = (ins
    Loom_Tile:$operand,
    Variadic<Index>:$operand_dims,
    Variadic<Index>:$result_dims,
    DefaultValuedAttr<Loom_TileBroadcastAlignAttr, "TileBroadcastAlign::Right">:$align
  );
  let results = (outs Loom_Tile:$result);

  let assemblyFormat = [{
    `` `<` $align `>`
    $operand
    attr-dict `:`
    qualified(type($operand)) (`` `{` $operand_dims^ `}`)? `->`
    qualified(type($result)) (`` `{` $result_dims^ `}`)?
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) {
      assert(idx == 0 && "broadcast has only one operand");
      return getOperandDims();
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      assert(idx == 0 && "broadcast has only one result");
      return getResultDims();
    }
  }];

  let errors = [
    ERR_LOOM_BROADCAST_0001,
  ];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_BROADCASTOP_TD
