// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TILE_UPDATEOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TILE_UPDATEOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.update
//------------------------------------------------------------------------------

def Loom_TileUpdateDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/update_op.mlir";
  let exampleRefs = ["@update_static", "@update_dynamic_offsets", "@update_fully_dynamic"];
  let relatedOps = ["loom.tile.slice", "loom.tensor.update", "tensor.insert_slice"];
  let tags = ["tile", "update", "in-place"];
}

def Loom_TileUpdateOp : Loom_PureOp<"loom.tile.update", [
  AllTypesMatch<["target", "result"]>,
  Util_ShapeAwareOp,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
  Loom_TileUpdateOffsetCountMatchesRank<"target", "static_offsets">,
  Loom_TileUpdateRanksMatch<"update", "target">,
  Loom_SameElementType<["update", "target"]>,
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileUpdateDocMetadata;

  let summary = "Update a tile region with a sub-tile (in-place semantics)";

  let description = [{
    Updates a region of the target tile with the contents of the update tile.
    This is for tile->tile updates with tied semantics.

    **Semantics:**
    For an update with offsets [o0, o1, ...] and sub-tile shape [s0, s1, ...]:
    ```
    result[o0 + i0, o1 + i1, ...] == update[i0, i1, ...]
    ```
    where 0 <= ik < sk for all dimensions k.
    Other positions in the tile are unchanged from target.

    **When to Use:**
    Use `loom.tile.update` to assemble smaller tiles into larger tiles
    within local compute memory. This typically appears in hierarchical tiling:
    - Accumulate results from register-level computation back into larger tiles
    - Implement multi-level tiling (e.g., subgroup tile -> workgroup tile)
    - Work within the tile domain without tensor boundary crossings

    To write from local memory (tile) back to global memory (tensor), use
    `loom.tensor.update` instead.

    The update is defined by:
    - `offsets`: Starting index in each dimension (mixed static/dynamic)
    - `sizes`: Implicit in update tile type shape
    - `strides`: Always unit stride (1) - not specified

    The result is **tied** to the target operand, enabling in-place updates
    when the target is not used elsewhere after this operation.

    **Operand Order:**
    The update tile comes first: `update %sub, %tile[offsets]`.
    This follows assignment semantics (`tile[offsets] = sub`).

    **Bounds Behavior:**
    Accessing regions outside target tile bounds is undefined behavior.
    Users must ensure `offset[k] + size[k] <= dim[k]` for all dimensions.

    **Assembly Format:**
    ```mlir
    // Static sizes:
    %updated = loom.tile.update %sub, %tile[0, 0]
        : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>

    // Dynamic offsets:
    %updated = loom.tile.update %sub, %tile[%off0, %off1]
        : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>

    // Fully dynamic:
    %updated = loom.tile.update %sub, %tile[%off0, %off1]
        : !loom.tile<?x?xf16>{%ssz0, %ssz1} -> !loom.tile<?x?xf16>{%d0, %d1}
    ```
  }];

  let arguments = (ins
    Loom_Tile:$target,
    Variadic<Index>:$target_dims,
    Loom_Tile:$update,
    Variadic<Index>:$update_dims,
    Variadic<Index>:$offsets,
    DenseI64ArrayAttr:$static_offsets
  );
  let results = (outs Loom_Tile:$result);

  let assemblyFormat = [{
    $update `,` $target ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    attr-dict `:`
    qualified(type($update)) (`` `{` $update_dims^ `}`)? `->`
    qualified(type($result)) (`` `{` $target_dims^ `}`)?
  }];

  let hasVerifier = 0;  // Use constraints only
  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    // ShapeAwareOpInterface implementation.
    // idx is the SSA operand index; we check which shaped operand it refers to.
    ValueRange getOperandDynamicDims(unsigned idx) {
      Value operand = (*this)->getOperand(idx);
      if (operand == getTarget()) return getTargetDims();
      if (operand == getUpdate()) return getUpdateDims();
      llvm_unreachable("idx does not refer to a shaped operand");
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      assert(idx == 0 && "update has only one result");
      return getTargetDims();  // Result shares target's dims (tied)
    }

    // Mixed offset/size/stride accessors (for analysis and transformations).
    // These combine static attributes with dynamic operands.
    SmallVector<OpFoldResult> getMixedOffsets();

    // Returns sizes derived from update tile type shape + update_dims.
    SmallVector<OpFoldResult> getMixedSizes();

    // Returns unit strides [1, 1, ...] for all dimensions.
    SmallVector<OpFoldResult> getMixedStrides();

    // Convenience accessors.
    IREE::Loom::TileType getTargetType() {
      return cast<IREE::Loom::TileType>(getTarget().getType());
    }
    IREE::Loom::TileType getUpdateType() {
      return cast<IREE::Loom::TileType>(getUpdate().getType());
    }
    IREE::Loom::TileType getResultType() {
      return cast<IREE::Loom::TileType>(getResult().getType());
    }
    int64_t getRank() { return getTargetType().getRank(); }
  }];

  let errors = [
    ERR_LOOM_UPDATE_0001,
    ERR_LOOM_UPDATE_0002,
    ERR_LOOM_TYPE_0002,
  ];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TILE_UPDATEOP_TD
