// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TILE_COPYOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TILE_COPYOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.copy
//------------------------------------------------------------------------------

def Loom_TileCopyDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/copy_op.mlir";
  let exampleRefs = ["@copy_static", "@copy_dynamic_offsets", "@copy_fully_dynamic"];
  let relatedOps = ["loom.tile.slice", "loom.tile.update", "memref.copy"];
  let tags = ["tile", "copy", "subrange"];
}

def Loom_TileCopyOp : Loom_PureOp<"loom.tile.copy", [
  // Note: We use InferTypeOpInterface instead of AllTypesMatch<["target", "result"]>
  // because variadic operands make operand indices unstable for auto-generated
  // type inference. The manual implementation correctly looks up target by name.
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  Util_ShapeAwareOp,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
  DeclareOpInterfaceMethods<Loom_CopySubrangeOpInterface>,
  // Generic constraints - multiple instances on same op.
  Loom_OffsetCountMatchesTileRank<"source", "static_source_offsets">,
  Loom_OffsetCountMatchesTileRank<"target", "static_target_offsets">,
  Loom_SizeCountMatchesTileRank<"source", "static_sizes">,
  Loom_TileRanksMatch<"source", "target">,
  Loom_SameElementType<["source", "target"]>,
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileCopyDocMetadata;

  let summary = "Copy a rectangular region between tiles";

  let description = [{
    Copies a rectangular region from the source tile to the target tile.
    This completes the slice/update/copy trio by supporting both source and
    target offsets in a single operation.

    **Semantics:**
    For a copy with source offsets [so0, so1, ...], target offsets [to0, to1, ...],
    and copy sizes [s0, s1, ...]:
    ```
    result[to0 + i0, to1 + i1, ...] == source[so0 + i0, so1 + i1, ...]
    ```
    where 0 <= ik < sk for all dimensions k.
    Other positions in the target are unchanged.

    **When to Use:**
    Use `loom.tile.copy` when you need to copy data between tiles with both
    source and target offsets:
    - Implement sliding window patterns (read from one region, write to another)
    - Copy partial tiles in accumulation patterns
    - Data reshuffling within local tile memory

    For simpler cases:
    - If only reading at an offset, use `loom.tile.slice` instead
    - If only writing at an offset (source is full tile), use `loom.tile.update`

    **Differences from slice/update:**
    - `slice`: Reads at offset, writes at [0,0,...], size = result shape
    - `update`: Reads at [0,0,...], writes at offset, size = update tile shape
    - `copy`: Reads at source offset, writes at target offset, size = explicit

    The copy is defined by:
    - `source_offsets`: Starting index in source for each dimension
    - `target_offsets`: Starting index in target for each dimension
    - `sizes`: Number of elements to copy in each dimension

    The result is **tied** to the target operand, enabling in-place modification.

    **Bounds Behavior:**
    Accessing regions outside bounds is undefined behavior. Users must ensure:
    - `source_offset[k] + size[k] <= source_dim[k]` for all dimensions
    - `target_offset[k] + size[k] <= target_dim[k]` for all dimensions

    **Assembly Format:**
    ```mlir
    // Static offsets and sizes
    %result = loom.tile.copy %src[0, 0], %tgt[16, 32], [8, 8]
        : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>

    // Dynamic source offsets
    %result = loom.tile.copy %src[%so0, %so1], %tgt[16, 32], [8, 8]
        : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>

    // Dynamic target offsets
    %result = loom.tile.copy %src[0, 0], %tgt[%to0, %to1], [8, 8]
        : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>

    // Fully dynamic
    %result = loom.tile.copy %src[%so0, %so1], %tgt[%to0, %to1], [%sz0, %sz1]
        : !loom.tile<?x?xf16>{%src_d0, %src_d1} -> !loom.tile<?x?xf16>{%tgt_d0, %tgt_d1}
    ```
  }];

  let arguments = (ins
    Loom_Tile:$source,
    Variadic<Index>:$source_dims,
    Variadic<Index>:$source_offsets,
    Loom_Tile:$target,
    Variadic<Index>:$target_dims,
    Variadic<Index>:$target_offsets,
    Variadic<Index>:$sizes,
    DenseI64ArrayAttr:$static_source_offsets,
    DenseI64ArrayAttr:$static_target_offsets,
    DenseI64ArrayAttr:$static_sizes
  );
  let results = (outs Loom_Tile:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($source_offsets, $static_source_offsets) `,`
    $target ``
    custom<DynamicIndexList>($target_offsets, $static_target_offsets) `,`
    custom<DynamicIndexList>($sizes, $static_sizes)
    attr-dict `:`
    qualified(type($source)) (`` `{` $source_dims^ `}`)? `->`
    qualified(type($target)) (`` `{` $target_dims^ `}`)?
  }];

  let hasVerifier = 0;  // Use constraints only
  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    // ShapeAwareOpInterface implementation.
    // idx is the SSA operand index; we check which shaped operand it refers to.
    ValueRange getOperandDynamicDims(unsigned idx) {
      Value operand = (*this)->getOperand(idx);
      if (operand == getSource()) return getSourceDims();
      if (operand == getTarget()) return getTargetDims();
      llvm_unreachable("idx does not refer to a shaped operand");
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      assert(idx == 0 && "copy has only one result");
      return getTargetDims();  // Result shares target's dims (tied)
    }

    // Mixed offset/size/stride accessors (for analysis and transformations).
    SmallVector<OpFoldResult> getMixedSourceOffsets();
    SmallVector<OpFoldResult> getMixedTargetOffsets();
    SmallVector<OpFoldResult> getMixedSizes();

    // Returns unit strides [1, 1, ...] for all dimensions.
    SmallVector<OpFoldResult> getMixedStrides();

    // Convenience accessors.
    IREE::Loom::TileType getSourceType() {
      return cast<IREE::Loom::TileType>(getSource().getType());
    }
    IREE::Loom::TileType getTargetType() {
      return cast<IREE::Loom::TileType>(getTarget().getType());
    }
    IREE::Loom::TileType getResultType() {
      return cast<IREE::Loom::TileType>(getResult().getType());
    }
    int64_t getRank() { return getSourceType().getRank(); }
  }];

  let errors = [
    ERR_LOOM_SUBRANGE_0001,
    ERR_LOOM_SUBRANGE_0002,
    ERR_LOOM_SUBRANGE_0003,
    ERR_LOOM_TYPE_0002,
  ];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TILE_COPYOP_TD
