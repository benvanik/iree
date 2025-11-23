// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_TEST_TESTOPS_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_TEST_TESTOPS_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomTypes.td"
include "iree/compiler/Dialect/Loom/IR/LoomConstraints.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"

//===----------------------------------------------------------------------===//
// Test OpDocGroup
//===----------------------------------------------------------------------===//

def Loom_TestOpsGroup : OpDocGroup {
  let summary = "Test operations for constraint validation";
  let description = [{
    These operations exist solely to test Loom constraint classes.
    They should never be used in production IR.
  }];
}

//===----------------------------------------------------------------------===//
// Type Property Constraint Tests
//===----------------------------------------------------------------------===//

// Tests: Util_SameElementType
def Loom_TestSameElementTypeOp : Loom_Op<"loom.test.same_element_type", [
  Util_SameElementType<["input", "result"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_SameElementType constraint (generic ShapedType)";
  let arguments = (ins Loom_Tile:$input);
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// Tests: Loom_SameEncoding
def Loom_TestSameEncodingOp : Loom_Op<"loom.test.same_encoding", [
  Loom_SameEncoding<["input", "result"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Loom_SameEncoding constraint";
  let arguments = (ins Loom_Tile:$input);
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// Tests: Util_SameElementType (all element types match)
def Loom_TestAllElementTypesMatchOp : Loom_Op<"loom.test.all_element_types_match", [
  Util_SameElementType<["lhs", "rhs", "result"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_SameElementType constraint with multiple operands";
  let arguments = (ins
    Loom_Tile:$lhs,
    Loom_Tile:$rhs
  );
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// Tests: Loom_OptionalGroupType
def Loom_TestOptionalGroupTypeOp : Loom_Op<"loom.test.optional_group_type", [
  Loom_OptionalGroupType<"group">
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Loom_OptionalGroupType constraint";
  let arguments = (ins
    Loom_Tile:$input,
    Optional<Loom_Group>:$group
  );
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$input (`,` $group^ `:` type($group))? attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Shape Relationship Constraint Tests
//===----------------------------------------------------------------------===//

// Tests: Util_SameRank
def Loom_TestSameRankOp : Loom_Op<"loom.test.same_rank", [
  Util_SameRank<["input", "result"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_SameRank constraint (generic ShapedType)";
  let arguments = (ins Loom_Tile:$input);
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// Tests: Util_SameShape
def Loom_TestSameShapeOp : Loom_Op<"loom.test.same_shape", [
  Util_SameShape<["lhs", "rhs"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_SameShape constraint (generic ShapedType)";
  let arguments = (ins
    Loom_Tile:$lhs,
    Loom_Tile:$rhs
  );
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// Tests: Util_SameRank (all ranks match)
def Loom_TestAllRanksMatchOp : Loom_Op<"loom.test.all_ranks_match", [
  Util_SameRank<["lhs", "rhs", "result"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_SameRank constraint with multiple operands";
  let arguments = (ins
    Loom_Tile:$lhs,
    Loom_Tile:$rhs
  );
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// Tests: Util_SameShape (all shapes match)
def Loom_TestAllShapesMatchOp : Loom_Op<"loom.test.all_shapes_match", [
  Util_SameShape<["lhs", "rhs", "result"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_SameShape constraint with multiple operands";
  let arguments = (ins
    Loom_Tile:$lhs,
    Loom_Tile:$rhs
  );
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// Tests: Util_ShapeCompatible
def Loom_TestShapeCompatibleOp : Loom_Op<"loom.test.shape_compatible", [
  Util_ShapeCompatible<["input", "result"]>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_ShapeCompatible constraint (broadcast-compatibility)";
  let arguments = (ins Loom_Tile:$input);
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// Tests: Util_RankEquals
def Loom_TestRankEqualsOp : Loom_Op<"loom.test.rank_equals", [
  Util_RankEquals<"matrix", 2>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_RankEquals constraint (requires rank-2)";
  let arguments = (ins Loom_Tile:$matrix);
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$matrix attr-dict `:` type($matrix) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Region Constraint Tests
//===----------------------------------------------------------------------===//

// Tests: Util_SingleBlockRegion
def Loom_TestSingleBlockRegionOp : Loom_Op<"loom.test.single_block_region", [
  Util_SingleBlockRegion<"body">
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_SingleBlockRegion constraint";
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

// Tests: Util_BlockArgCount
def Loom_TestBlockArgCountOp : Loom_Op<"loom.test.block_arg_count", [
  Util_SingleBlockRegion<"body">,
  Util_BlockArgCount<"body", 2>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_BlockArgCount constraint (requires 2 args)";
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

//===----------------------------------------------------------------------===//
// Numeric/Attribute Constraint Tests
//===----------------------------------------------------------------------===//

// Tests: Util_PermutationArray
def Loom_TestPermutationArrayOp : Loom_Op<"loom.test.permutation_array", [
  Util_PermutationArray<"perm", 3>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_PermutationArray constraint (rank-3)";
  let arguments = (ins
    Loom_Tile:$input,
    ArrayAttr:$perm
  );
  let results = (outs Loom_Tile:$result);
  let assemblyFormat = "$input `perm` `=` $perm attr-dict `:` type($input) `->` type($result)";
}

// Tests: Util_StaticArrayLength
def Loom_TestStaticArrayLengthOp : Loom_Op<"loom.test.static_array_length", [
  Util_StaticArrayLength<"static_values", 4>
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_StaticArrayLength constraint (requires 4 elements)";
  let arguments = (ins
    ArrayAttr:$static_values
  );
  let results = (outs);
  let assemblyFormat = "`values` `=` $static_values attr-dict";
}

// Tests: Util_DynamicCountMatchesMarkers
def Loom_TestDynamicCountMatchesMarkersOp : Loom_Op<"loom.test.dynamic_count_matches_markers", [
  Util_DynamicCountMatchesMarkers<"static_offsets", "dynamic_offsets">
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_DynamicCountMatchesMarkers constraint";
  let description = [{
    Tests mixed static/dynamic operand counting.
    static_offsets array with -1 markers indicates dynamic positions.
    dynamic_offsets operand count must match number of -1 markers.
  }];
  let arguments = (ins
    ArrayAttr:$static_offsets,
    Variadic<Index>:$dynamic_offsets
  );
  let results = (outs);
  let assemblyFormat = "`static` `=` $static_offsets (`dynamic` `=` $dynamic_offsets^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// Memory/Layout Constraint Tests
//===----------------------------------------------------------------------===//

// Tests: Util_IdentityLayout
def Loom_TestIdentityLayoutOp : Loom_Op<"loom.test.identity_layout", [
  Util_IdentityLayout<"memref_arg">
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_IdentityLayout constraint";
  let arguments = (ins AnyMemRef:$memref_arg);
  let results = (outs);
  let assemblyFormat = "$memref_arg attr-dict `:` type($memref_arg)";
}

// Tests: Util_PositiveI64Attr (for cooperative hints)
def Loom_TestCooperativeHintsValidOp : Loom_Op<"loom.test.cooperative_hints_valid", [
  Util_PositiveI64Attr<"coop_lanes">,
  Util_PositiveI64Attr<"coop_elems_per_lane">
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_PositiveI64Attr constraint (cooperative hints)";
  let arguments = (ins
    Loom_Tile:$tile,
    I64Attr:$coop_lanes,
    I64Attr:$coop_elems_per_lane
  );
  let results = (outs);
  let assemblyFormat = "$tile attr-dict `:` type($tile)";
}

//===----------------------------------------------------------------------===//
// Complex Multi-Operand Constraint Tests
//===----------------------------------------------------------------------===//

// Tests: Util_TiedOperandRange
def Loom_TestTiedOperandRangeOp : Loom_Op<"loom.test.tied_operand_range", [
  Util_TiedOperandRange<"inputs", "results">
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_TiedOperandRange constraint";
  let description = [{
    Tests that tied_operands array indices are valid.
    Each result can tie to an input operand (-1 = untied).
  }];
  let arguments = (ins
    Variadic<Loom_Tile>:$inputs,
    ArrayAttr:$tied_operands
  );
  let results = (outs Variadic<Loom_Tile>:$results);
  let assemblyFormat = "$inputs `tied` `=` $tied_operands attr-dict `:` type($inputs) `->` type($results)";
}

// Tests: Util_IterArgsMatchResults
def Loom_TestIterArgsMatchResultsOp : Loom_Op<"loom.test.iter_args_match_results", [
  Util_SingleBlockRegion<"body">,
  Util_IterArgsMatchResults<"iter_args", "results">
]> {
  let opDocGroup = Loom_TestOpsGroup;
  let summary = "Test Util_IterArgsMatchResults constraint";
  let description = [{
    Tests loop-carried state: iter_args and results must match in count and type.
  }];
  let arguments = (ins Variadic<Loom_Tile>:$iter_args);
  let results = (outs Variadic<Loom_Tile>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`(` $iter_args `)` $body attr-dict `:` type($iter_args) `->` type($results)";
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_TEST_TESTOPS_TD
