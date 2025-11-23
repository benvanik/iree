// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_ALLOCAOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_ALLOCAOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomTypes.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.alloca
//------------------------------------------------------------------------------

def Loom_TileAllocaDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/alloca_op.mlir";
  let exampleRefs = ["@tile_alloca_static", "@tile_alloca_dynamic", "@tile_alloca_then_fill"];
  let relatedOps = ["loom.tile.fill", "memref.alloca"];
  let tags = ["tile", "allocation", "memory"];
}

def Loom_TileAllocaOp : Loom_Op<"loom.tile.alloca", [
  Util_ShapeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileAllocaDocMetadata;

  let summary = "Allocate an uninitialized tile";

  let description = [{
    Allocates a tile with undefined contents. The tile shape may include
    dynamic dimensions which are provided as operands.

    **Semantics:**
    Creates an uninitialized tile in local compute memory. Contents are
    undefined until explicitly written - reading before initialization
    is undefined behavior.

    **When to Use:**
    Use `loom.tile.alloca` when you need a tile as an output destination:
    - Allocating workspace tiles for computation results
    - Creating accumulator tiles (typically followed by `loom.tile.fill`)
    - Temporary storage for intermediate results in multi-step computations

    For tiles with known constant values, prefer `loom.tile.constant`.
    For tiles initialized from tensor data, use `loom.tensor.slice`.

    **Memory Effects:**
    The allocation has `MemAlloc` effect, which affects CSE and code motion.
    Multiple allocas with the same type are not deduplicated.

    **Assembly Format:**
    ```mlir
    // Static shape
    %tile = loom.tile.alloca : !loom.tile<4x4xf32>

    // Dynamic shape
    %tile = loom.tile.alloca : !loom.tile<?x?xf32>{%d0, %d1}

    // Typical pattern: alloca then fill
    %uninit = loom.tile.alloca : !loom.tile<?x?xf32>{%d0, %d1}
    %zeros = loom.tile.fill %cst, %uninit : f32 -> !loom.tile<?x?xf32>{%d0, %d1}
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$result_dims
  );
  let results = (outs Loom_Tile:$result);

  let assemblyFormat = [{
    attr-dict `:` qualified(type($result)) (`{` $result_dims^ `}`)?
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) {
      llvm_unreachable("alloca has no shaped operands");
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      assert(idx == 0 && "alloca has only one result");
      return getResultDims();
    }
  }];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_ALLOCAOP_TD
