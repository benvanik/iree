// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TILE_SLICEOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TILE_SLICEOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.slice
//------------------------------------------------------------------------------

def Loom_TileSliceDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/slice_op.mlir";
  let exampleRefs = ["@slice_static", "@slice_dynamic_offsets", "@slice_fully_dynamic"];
  let relatedOps = ["loom.tensor.slice", "loom.tile.update", "tensor.extract_slice"];
  let tags = ["tile", "slice", "extraction"];
}

def Loom_TileSliceOp : Loom_PureOp<"loom.tile.slice", [
  Util_ShapeAwareOp,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Loom_CopySubrangeOpInterface>,
  Loom_TileSliceOffsetCountMatchesRank<"source", "static_offsets">,
  Loom_TileSliceRanksMatch<"source", "result">,
  Loom_SameElementType<["source", "result"]>,
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileSliceDocMetadata;

  let summary = "Extract a sub-tile from a tile (copy semantics)";

  let description = [{
    Extracts a dense, rectangular, unit-stride sub-tile from a source tile.
    This is for tile->tile extraction with value semantics.

    **Semantics:**
    For a slice with offsets [o0, o1, ...] and result tile shape [s0, s1, ...]:
    ```
    result[i0, i1, ...] == source[o0 + i0, o1 + i1, ...]
    ```
    where 0 <= ik < sk for all dimensions k.

    **When to Use:**
    Use `loom.tile.slice` to subdivide tiles within local compute memory.
    This typically appears in hierarchical tiling patterns:
    - Extract smaller work units from larger tiles for register-level computation
    - Implement multi-level tiling (e.g., workgroup tile -> subgroup tile)
    - Work within the tile domain without tensor boundary crossings

    To load from global memory (tensor) into local memory (tile), use
    `loom.tensor.slice` instead.

    The slice is defined by:
    - `offsets`: Starting index in each dimension (mixed static/dynamic)
    - `sizes`: Implicit in result tile type shape
    - `strides`: Always unit stride (1) - not specified

    Unlike "view" operations, this has **copy semantics**: the result tile
    is an independent value that does not alias the source tile.

    **Bounds Behavior:**
    Accessing regions outside source tile bounds is undefined behavior.
    Users must ensure `offset[k] + size[k] <= dim[k]` for all dimensions.

    **Assembly Format:**
    ```mlir
    // Static tile sizes:
    %sub = loom.tile.slice %tile[0, 0]
        : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>

    // Dynamic offsets:
    %subtile = loom.tile.slice %tile[%offset0, %offset1]
        : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>

    // Fully dynamic:
    %subtile = loom.tile.slice %tile[%offset0, %offset1]
        : !loom.tile<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%size0, %size1}
    ```
  }];

  let arguments = (ins
    Loom_Tile:$source,
    Variadic<Index>:$source_dims,
    Variadic<Index>:$offsets,
    Variadic<Index>:$result_dims,
    DenseI64ArrayAttr:$static_offsets
  );
  let results = (outs Loom_Tile:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    attr-dict `:`
    qualified(type($source)) (`` `{` $source_dims^ `}`)? `->`
    qualified(type($result)) (`` `{` $result_dims^ `}`)?
  }];

  let hasVerifier = 0;  // Use constraints only
  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    // ShapeAwareOpInterface implementation.
    ValueRange getOperandDynamicDims(unsigned idx) {
      assert(idx == 0 && "slice has only one operand");
      return getSourceDims();
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      assert(idx == 0 && "slice has only one result");
      return getResultDims();
    }

    // Mixed offset/size/stride accessors (for analysis and transformations).
    // These combine static attributes with dynamic operands.
    SmallVector<OpFoldResult> getMixedOffsets();

    // Returns sizes derived from result tile type shape + result_dims.
    SmallVector<OpFoldResult> getMixedSizes();

    // Returns unit strides [1, 1, ...] for all dimensions.
    SmallVector<OpFoldResult> getMixedStrides();

    // Convenience accessors.
    IREE::Loom::TileType getSourceType() {
      return cast<IREE::Loom::TileType>(getSource().getType());
    }
    IREE::Loom::TileType getResultType() {
      return cast<IREE::Loom::TileType>(getResult().getType());
    }
    int64_t getRank() { return getSourceType().getRank(); }
  }];

  let errors = [
    ERR_LOOM_SLICE_0001,
    ERR_LOOM_SLICE_0002,
    ERR_LOOM_TYPE_0002,
  ];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TILE_SLICEOP_TD
