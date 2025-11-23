// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TILE_TILEOPS_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TILE_TILEOPS_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomTypes.td"
include "iree/compiler/Utils/DocMetadata.td"

//===----------------------------------------------------------------------===//
// OpDocGroup definition
//===----------------------------------------------------------------------===//

def TileOpsGroup : OpDocGroup {
  let summary = "Tile manipulation operations";
  let description = [{
    Operations for creating, transforming, and manipulating `!loom.tile` values.
    Tiles represent shaped, encoded sub-regions of tensors used for computation.

    This group includes:
    - **Creation**: constant, iota (generate tile contents)
    - **Initialization**: fill (overwrite with scalar value)
    - **Shape transformation**: broadcast, transpose (change tile shape/layout)
    - **Compute**: elementwise (region-based pointwise operations)

    All tile operations are pure (side-effect free) except fill, which models
    an in-place write for efficiency.
  }];
}

//===----------------------------------------------------------------------===//
// Shared enums and attributes for tile operations
//===----------------------------------------------------------------------===//

def Loom_TileBroadcastAlignAttr : Loom_I32EnumAttr<
    "TileBroadcastAlign", "broadcast alignment", [
      I32EnumAttrCase<"Left", 0, "left">,
      I32EnumAttrCase<"Right", 1, "right">
    ]> {
  let summary = "Broadcast dimension alignment mode";
  let description = [{
    Controls how source dimensions align with result dimensions during broadcasting.
    Determines whether new dimensions are prepended (left) or appended (right).
  }];
}

//===----------------------------------------------------------------------===//
// Op includes
//===----------------------------------------------------------------------===//

include "iree/compiler/Dialect/Loom/IR/Tile/AllocaOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/BroadcastOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/ConstantOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/DimOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/ElementwiseOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/FillOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/SliceOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/UpdateOp.td"
include "iree/compiler/Dialect/Loom/IR/Tile/YieldOp.td"

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TILE_TILEOPS_TD
