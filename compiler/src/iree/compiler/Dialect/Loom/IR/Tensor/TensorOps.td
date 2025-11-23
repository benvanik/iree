// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_TENSOROPS_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_TENSOROPS_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomTypes.td"
include "iree/compiler/Utils/DocMetadata.td"

//===----------------------------------------------------------------------===//
// OpDocGroup definition
//===----------------------------------------------------------------------===//

def TensorOpsGroup : OpDocGroup {
  let summary = "Tensor operations and tensor-tile boundary";
  let description = [{
    Operations for working with `!loom.tensor` values and crossing the
    tensor/tile boundary. Tensors represent global buffer regions with
    shape and encoding information.

    This group includes:
    - **Query**: dim (get dimension size)
    - **Boundary**: slice (tensor->tile), update (tile->tensor)
  }];
}

//===----------------------------------------------------------------------===//
// Op includes
//===----------------------------------------------------------------------===//

include "iree/compiler/Dialect/Loom/IR/Tensor/DimOp.td"
include "iree/compiler/Dialect/Loom/IR/Tensor/SliceOp.td"
include "iree/compiler/Dialect/Loom/IR/Tensor/UpdateOp.td"

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TENSOR_TENSOROPS_TD
