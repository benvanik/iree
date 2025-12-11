// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_UPDATEOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_UPDATEOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tensor.update
//------------------------------------------------------------------------------

def Loom_TensorUpdateDocMetadata : Util_OpDocMetadata {
  let docGroup = TensorOpsGroup;
  let testFile = "Tensor/test/update_op.mlir";
  let exampleRefs = ["@update_static", "@update_dynamic_offsets", "@update_fully_dynamic"];
  let relatedOps = ["tensor.insert_slice", "loom.tensor.slice", "loom.tile.update"];
  let tags = ["tensor", "tile", "update", "boundary", "in-place"];
}

def Loom_TensorUpdateOp : Loom_PureOp<"loom.tensor.update", [
  AllTypesMatch<["target", "result"]>,
  Util_ShapeAwareOp,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
  DeclareOpInterfaceMethods<Loom_CopySubrangeOpInterface>,
  Loom_TensorUpdateOffsetCountMatchesRank<"target", "static_offsets">,
  Loom_TensorUpdateRanksMatch<"update", "target">,
  Loom_SameElementType<["update", "target"]>,
]> {
  let opDocGroup = TensorOpsGroup;
  let docMetadata = Loom_TensorUpdateDocMetadata;

  let summary = "Update a tensor region with a tile (copy back, in-place semantics)";

  let description = [{
    Updates a region of the target tensor with the contents of the update tile.
    This is the primary tile->tensor boundary operation with tied semantics.

    **Semantics:**
    For an update with offsets [o0, o1, ...] and tile shape [s0, s1, ...]:
    ```
    result[o0 + i0, o1 + i1, ...] == tile[i0, i1, ...]
    ```
    where 0 <= ik < sk for all dimensions k.
    Other positions in the tensor are unchanged from target.

    **When to Use:**
    Use `loom.tensor.update` to write computed results from local memory (tile)
    back to global memory (tensor). This typically appears:
    - At the end of tiled loops to store computed results to output tensors
    - When transitioning from tile-level back to tensor-level representation
    - As the inverse of `loom.tensor.slice`

    For updating regions within tiles in local memory, use `loom.tile.update`.

    The update is defined by:
    - `offsets`: Starting index in each dimension (mixed static/dynamic)
    - `sizes`: Implicit in update tile type shape
    - `strides`: Always unit stride (1) - not specified

    The result is **tied** to the target operand, enabling in-place updates
    when the target is not used elsewhere after this operation.

    **Operand Order:**
    The update tile comes first: `update %tile, %target[offsets]`.
    This follows assignment semantics (`target[offsets] = tile`).

    **Bounds Behavior:**
    Accessing regions outside target tensor bounds is undefined behavior.
    Users must ensure `offset[k] + size[k] <= dim[k]` for all dimensions.

    **Assembly Format:**
    ```mlir
    // Static tensor, static tile sizes:
    %updated = loom.tensor.update %tile, %tensor[0, 0]
        : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>

    // Dynamic offsets:
    %updated = loom.tensor.update %tile, %tensor[%off0, %off1]
        : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>

    // Fully dynamic:
    %updated = loom.tensor.update %tile, %tensor[%off0, %off1]
        : !loom.tile<?x?xf16>{%tsz0, %tsz1} -> !loom.tensor<?x?xf16>{%d0, %d1}
    ```
  }];

  let arguments = (ins
    Loom_Tensor:$target,
    Variadic<Index>:$target_dims,
    Loom_Tile:$update,
    Variadic<Index>:$update_dims,
    Variadic<Index>:$offsets,
    DenseI64ArrayAttr:$static_offsets
  );
  let results = (outs Loom_Tensor:$result);

  let assemblyFormat = [{
    $update `,` $target ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    attr-dict `:`
    qualified(type($update)) (`` `{` $update_dims^ `}`)? `->`
    qualified(type($result)) (`` `{` $target_dims^ `}`)?
  }];

  let hasVerifier = 0;  // Use constraints only
  let hasCanonicalizer = 1;

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
    IREE::Loom::TensorType getTargetType() {
      return cast<IREE::Loom::TensorType>(getTarget().getType());
    }
    IREE::Loom::TileType getUpdateType() {
      return cast<IREE::Loom::TileType>(getUpdate().getType());
    }
    IREE::Loom::TensorType getResultType() {
      return cast<IREE::Loom::TensorType>(getResult().getType());
    }
    int64_t getRank() { return getTargetType().getRank(); }
  }];

}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_UPDATEOP_TD
