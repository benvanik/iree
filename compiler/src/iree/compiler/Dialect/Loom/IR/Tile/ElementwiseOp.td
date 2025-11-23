// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_ELEMENTWISEOP_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_ELEMENTWISEOP_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Loom/IR/LoomTypes.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpBase.td"

//------------------------------------------------------------------------------
// loom.tile.elementwise
//------------------------------------------------------------------------------

def Loom_TileElementwiseDocMetadata : Util_OpDocMetadata {
  let docGroup = TileOpsGroup;
  let testFile = "Tile/test/elementwise_op.mlir";
  let exampleRefs = ["@elementwise_unary", "@elementwise_binary", "@elementwise_with_capture"];
  let relatedOps = ["loom.tile.yield", "linalg.map"];
  let tags = ["tile", "compute", "region", "elementwise"];
}

def Loom_TileElementwiseOp : Loom_PureOp<"loom.tile.elementwise", [
  Util_ShapeAwareOp,
  AttrSizedOperandSegments,
  SingleBlockImplicitTerminator<"TileYieldOp">,
  // Note: inputs can be empty (zero inputs) when all inputs are propagated
  // constants/poison. Shape comes from result_dims in that case.
  Loom_AllTileShapesMatch<"inputs">,
  Loom_TileResultShapeMatchesInputs<"result", "inputs">,
  Loom_TileBlockArgCount<"body", "inputs">,
  Loom_TileBlockArgsMatchElements<"body", "inputs">,
  Loom_TileYieldSingleValue<"body">,
  Loom_TileYieldTypeMatchesResult<"body", "result">,
]> {
  let opDocGroup = TileOpsGroup;
  let docMetadata = Loom_TileElementwiseDocMetadata;

  let summary = "Region-based elementwise map over one or more tiles";

  let description = [{
    Applies a scalar computation pointwise over one or more input tiles,
    producing a result tile of the same shape. The computation is defined
    in a region that receives scalar elements from each input tile and
    yields a single scalar value for the output.

    **Semantics:**
    - All input tiles must have identical shapes (no broadcasting)
    - The result tile has the same shape as the inputs
    - The region receives one scalar block argument per input tile
    - Values from outside the region are implicitly captured
    - The region must terminate with `loom.tile.yield` producing one value

    **Named Binding Syntax:**
    Uses inline bindings instead of explicit block arguments:
    ```mlir
    %result = loom.tile.elementwise(
        %element_a = %a : !loom.tile<?x?xf16>{%dim0, %dim1},
        %element_b = %b : !loom.tile<?x?xf16>{%dim0, %dim1}) {
      %x = arith.mulf %element_a, %scale : f16  // %scale implicitly captured
      %y = arith.addf %x, %element_b : f16
      loom.tile.yield %y : f16
    } -> !loom.tile<?x?xf16>{%dim0, %dim1}
    ```

    **When to Use:**
    Use `loom.tile.elementwise` for pointwise computations over tiles:
    - Unary operations (negation, abs, math functions)
    - Binary operations (add, mul, max, min)
    - Fused operations (scale+bias, FMA)
    - Type conversions with computation

    **Implicit Capture:**
    Scalar values defined outside the region can be used directly without
    explicit declaration. This enables clean fusion of operations like
    scale+bias without verbose capture lists.

    **Example:**
    ```mlir
    // Unary operation
    %neg = loom.tile.elementwise(%e = %t : !loom.tile<4x4xf32>) {
      %r = arith.negf %e : f32
      loom.tile.yield %r : f32
    } -> !loom.tile<4x4xf32>

    // Binary with implicit capture
    %fused = loom.tile.elementwise(
        %a = %x : !loom.tile<8xf16>,
        %b = %y : !loom.tile<8xf16>) {
      %sum = arith.addf %a, %b : f16
      %scaled = arith.mulf %sum, %scale : f16
      loom.tile.yield %scaled : f16
    } -> !loom.tile<8xf16>
    ```
  }];

  let arguments = (ins
    Variadic<Loom_Tile>:$inputs,
    Variadic<Index>:$input_dims,    // Flattened: [input0_d0, input0_d1, input1_d0, ...]
    Variadic<Index>:$result_dims    // Dynamic dims for result tile
  );
  let results = (outs Loom_Tile:$result);
  let regions = (region SizedRegion<1>:$body);

  // Hybrid format: custom<> handles parens, bindings, and region.
  // Parens must be in custom<> because `)` comes before the region `{`.
  // Use `` before custom<> to suppress the leading space.
  let assemblyFormat = [{
    `` custom<ElementwiseBindings>(
      $inputs, type($inputs), $input_dims, $body)
    `->` qualified(type($result)) (`{` $result_dims^ `}`)?
    attr-dict
  }];

  let hasCanonicalizer = 1;

  let errors = [
    ERR_LOOM_ELEMENTWISE_0001,
    ERR_LOOM_ELEMENTWISE_0002,
    ERR_LOOM_ELEMENTWISE_0003,
    ERR_LOOM_ELEMENTWISE_0004,
    ERR_LOOM_ELEMENTWISE_0005,
    ERR_LOOM_ELEMENTWISE_0006,
    ERR_LOOM_ELEMENTWISE_0007,
  ];

  let extraClassDeclaration = [{
    // ShapeAwareOpInterface: Return dynamic dims for input at idx.
    // Slices the flattened input_dims based on each input's dynamic dim count.
    ValueRange getOperandDynamicDims(unsigned idx);

    // ShapeAwareOpInterface: Return result dynamic dims.
    ValueRange getResultDynamicDims(unsigned idx) {
      assert(idx == 0 && "elementwise has only one result");
      return getResultDims();
    }
  }];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_OPS_TILE_ELEMENTWISEOP_TD
