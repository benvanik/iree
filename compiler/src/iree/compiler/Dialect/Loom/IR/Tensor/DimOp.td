// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_DIMOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_DIMOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//------------------------------------------------------------------------------
// loom.tensor.dim
//------------------------------------------------------------------------------

def Loom_TensorDimDocMetadata : Util_OpDocMetadata {
  let docGroup = TensorOpsGroup;
  let testFile = "Tensor/test/dim_op.mlir";
  let exampleRefs = ["@tensor_dim_static", "@tensor_dim_dynamic"];
  let relatedOps = ["loom.tile.dim", "tensor.dim", "memref.dim"];
  let tags = ["tensor", "shape", "query"];
}

def Loom_TensorDimOp : Loom_PureOp<"loom.tensor.dim", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Loom_TensorDimIndexInBounds<"source", "index">
]> {
  let opDocGroup = TensorOpsGroup;
  let docMetadata = Loom_TensorDimDocMetadata;

  let summary = "Query the size of a tensor dimension";

  let description = [{
    Returns the size of the specified dimension of a tensor. The dimension
    index can be a constant or an SSA value.

    **Semantics:**
    For a tensor with shape [d0, d1, ..., dn-1], querying dimension `k` returns
    the value `dk`. For static dimensions, this is known at compile time.
    For dynamic dimensions (`?`), the value comes from shape propagation.

    **When to Use:**
    Use `loom.tensor.dim` to query tensor shapes for:
    - Computing tile loop bounds when iterating over input tensors
    - Deriving offsets and sizes for slice/update operations
    - Passing shape information to tiling transformations

    For querying tile shapes (local memory), use `loom.tile.dim` instead.

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
    %d0 = loom.tensor.dim %tensor, 0 : !loom.tensor<64x?xf32>

    // Query dynamic dimension
    %d1 = loom.tensor.dim %tensor, 1 : !loom.tensor<64x?xf32>

    // Query with SSA index
    %dim = loom.tensor.dim %tensor, %idx : !loom.tensor<?x?xf32>
    ```
  }];

  let arguments = (ins
    Loom_Tensor:$source,
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

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_DIMOP_TD
