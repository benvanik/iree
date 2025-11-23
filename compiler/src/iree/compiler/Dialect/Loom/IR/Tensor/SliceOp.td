// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_SLICEOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_SLICEOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tensor.slice
//------------------------------------------------------------------------------

def Loom_TensorSliceDocMetadata : Util_OpDocMetadata {
  let docGroup = TensorOpsGroup;
  let testFile = "Tensor/test/slice_op.mlir";
  let exampleRefs = ["@slice_static", "@slice_dynamic_offsets", "@slice_fully_dynamic"];
  let relatedOps = ["tensor.extract_slice", "loom.tensor.update", "loom.tile.slice"];
  let tags = ["tensor", "tile", "slice", "boundary"];
}

def Loom_TensorSliceOp : Loom_PureOp<"loom.tensor.slice", [
  Util_ShapeAwareOp,
  AttrSizedOperandSegments,
  Loom_TensorSliceOffsetCountMatchesRank<"source", "static_offsets">,
  Loom_TensorSliceRanksMatch<"source", "result">,
  Loom_SameElementType<["source", "result"]>,
]> {
  let opDocGroup = TensorOpsGroup;
  let docMetadata = Loom_TensorSliceDocMetadata;

  let summary = "Extract a tile from a tensor (copy semantics)";

  let description = [{
    Extracts a dense, rectangular, unit-stride tile from a tensor.
    This is the primary tensor->tile boundary operation with value semantics.

    **Semantics:**
    For a slice with offsets [o0, o1, ...] and result tile shape [s0, s1, ...]:
    ```
    tile[i0, i1, ...] == tensor[o0 + i0, o1 + i1, ...]
    ```
    where 0 <= ik < sk for all dimensions k.

    **When to Use:**
    Use `loom.tensor.slice` to load data from global memory (tensor) into
    local compute memory (tile). This typically appears:
    - At the start of tiled loops to extract work items from input tensors
    - When transitioning from tensor-level to tile-level computation
    - As the inverse of `loom.tensor.update`

    For subdividing tiles already in local memory, use `loom.tile.slice`.

    The slice is defined by:
    - `offsets`: Starting index in each dimension (mixed static/dynamic)
    - `sizes`: Implicit in result tile type shape
    - `strides`: Always unit stride (1) - not specified

    Unlike "view" operations, this has **copy semantics**: the result tile
    is an independent value that does not alias the source tensor.

    **Bounds Behavior:**
    Accessing regions outside source tensor bounds is undefined behavior.
    Users must ensure `offset[k] + size[k] <= dim[k]` for all dimensions.

    **Assembly Format:**
    ```mlir
    // Static tensor, static tile sizes:
    %tile = loom.tensor.slice %src[0, 0]
        : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>

    // Dynamic offsets:
    %tile = loom.tensor.slice %src[%offset0, %offset1]
        : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>

    // Dynamic tensor and tile:
    %tile = loom.tensor.slice %src[%offset0, %offset1]
        : !loom.tensor<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%size0, %size1}
    ```

  }];

  let arguments = (ins
    Loom_Tensor:$source,
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
    IREE::Loom::TensorType getSourceType() {
      return cast<IREE::Loom::TensorType>(getSource().getType());
    }
    IREE::Loom::TileType getResultType() {
      return cast<IREE::Loom::TileType>(getResult().getType());
    }
    int64_t getRank() { return getSourceType().getRank(); }
  }];

}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_SLICEOP_TD
