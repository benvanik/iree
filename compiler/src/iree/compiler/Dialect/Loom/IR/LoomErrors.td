// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured Loom error definitions with stable IDs and typed arguments.

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOM_ERRORS_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOM_ERRORS_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Util/IR/UtilErrorBase.td"

//===----------------------------------------------------------------------===//
// Loom Error Wrapper
//===----------------------------------------------------------------------===//

// Loom error wrapper - inherits all fields from Util_ErrorBase.
// Example: def ERR_LOOM_BROADCAST_0001 : Loom_Error<"BROADCAST", "0001"> { ... }
// The dialect prefix ("LOOM") is automatically provided.
class Loom_Error<string domainName, string codeNum>
  : Util_ErrorBase<Loom_Dialect, "LOOM", domainName, codeNum>;

//===----------------------------------------------------------------------===//
// Loom Dialect Errors
//===----------------------------------------------------------------------===//
//
// Structured error definitions for the Loom dialect.
// Each error has a unique ID, typed arguments, and generates C++ emission code.
//
// Error ID Format: ERR_LOOM_<DOMAIN>_<CODE>
// - DOMAIN: Error category/scope for this dialect
//   - BROADCAST: Broadcasting/shape expansion operations
//   - ENCODING: TileType encoding/layout errors
//   - FOLD: Canonicalization remarks (poison folds, etc.)
//   - ATTR: Attribute validation errors
// - CODE: Sequential number within domain (0001, 0002, etc.)
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Broadcast Errors (BROADCAST domain)
//===----------------------------------------------------------------------===//

def ERR_LOOM_BROADCAST_0001 : Loom_Error<"BROADCAST", "0001"> {
  let summary = [{Incompatible dimension sizes during broadcast.}];
  let description = [{
    Triggered when a source dimension cannot be broadcast to the result
    dimension.
  }];
  let message = [{
    incompatible dimension at source axis {sourceAxis} (size {sourceSize}) with result axis {resultAxis} (size {resultSize})
  }];
  let schema = (ins
    "unsigned":$sourceAxis,
    "int64_t":$sourceSize,
    "unsigned":$resultAxis,
    "int64_t":$resultSize
  );
  let fixHint = [{Source axis {sourceAxis} (size {sourceSize}) must be 1 or match result axis {resultAxis} (size {resultSize})}];
  let example = [{
    Valid: <1x8> -> <4x8> (broadcast dim 0), <4x8> -> <4x8> (exact match)
    Invalid: <2x8> -> <4x8> (dim 0 is neither 1 nor 4)
  }];
}

//===----------------------------------------------------------------------===//
// Encoding Errors (ENCODING domain)
//===----------------------------------------------------------------------===//

def ERR_LOOM_ENCODING_0001 : Loom_Error<"ENCODING", "0001"> {
  let summary = [{Operands must have the same TileType encoding.}];
  let description = [{
    Triggered when operands or results are required to have matching TileType
    encodings but differ. Encodings represent layout, memory order, or hardware
    format requirements.
  }];
  let message = [{encoding mismatch: {operandNames} have different encodings}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$operandNames,
    ArrayRefParameter<"Attribute">:$encodings
  );
  let fixHint = [{Operands '{operandNames}' have encodings {encodings} - use explicit encoding casts if needed}];
  let example = [{
    // All operands must share the same encoding
    %result = loom.op %a, %b : !loom.tile<4x8xf32, #enc1>, !loom.tile<4x8xf32, #enc1>
  }];
}

//===----------------------------------------------------------------------===//
// Type Errors (TYPE domain)
//===----------------------------------------------------------------------===//

def ERR_LOOM_TYPE_0001 : Loom_Error<"TYPE", "0001"> {
  let summary = [{Optional operand must be !loom.group type.}];
  let description = [{
    Triggered when an optional operand is provided but does not have the
    expected !loom.group type. Group operands are used for ordering and
    dependency tracking.
  }];
  let message = [{operand '{operandName}' must be !loom.group type}];
  let schema = (ins
    StringRefParameter<>:$operandName,
    "Type":$actualType
  );
  let fixHint = [{'{operandName}' has type {actualType}, expected !loom.group or omit entirely}];
  let example = [{
    // Correct: group operand has !loom.group type
    %result = loom.op %input, %group : !loom.tile<4xf32>, !loom.group
    // Or omit the optional group
    %result = loom.op %input : !loom.tile<4xf32>
  }];
}

def ERR_LOOM_TYPE_0002 : Loom_Error<"TYPE", "0002"> {
  let summary = [{Operands must have the same element type.}];
  let description = [{
    Triggered when operands or results are required to have matching element
    types but differ. Element types are the scalar types within tile types.
  }];
  let message = [{element type mismatch: {operandNames} have different element types}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$operandNames,
    ArrayRefParameter<"Type">:$elementTypes
  );
  let fixHint = [{Operands '{operandNames}' have element types {elementTypes} - use explicit type casts if needed}];
  let example = [{
    // All operands must share the same element type
    %result = loom.op %a, %b : !loom.tile<4x8xf32>, !loom.tile<4x8xf32>
    // Invalid: mixing f32 and f16
    %result = loom.op %a, %b : !loom.tile<4x8xf32>, !loom.tile<4x8xf16>
  }];
}

//===----------------------------------------------------------------------===//
// Fill Errors (FILL domain)
//===----------------------------------------------------------------------===//

def ERR_LOOM_FILL_0001 : Loom_Error<"FILL", "0001"> {
  let summary = [{Fill value type mismatch.}];
  let description = [{
    The fill value scalar type must match the target tile's element type.
  }];
  let message = [{
    fill value has type '{valueType}', but tile element type is '{elementType}'
  }];
  let schema = (ins
    "Type":$valueType,
    "Type":$elementType
  );
  let fixHint = [{Cast or convert fill value to type '{elementType}'}];
  let example = [{
    // Correct: f32 value matches f32 tile element type
    %result = loom.tile.fill %value, %tile : f32 -> <4x4xf32>
    // Invalid: i32 value with f32 tile
    %result = loom.tile.fill %i32val, %tile : i32 -> <4x4xf32>
  }];
}

//===----------------------------------------------------------------------===//
// Constant Errors (CONSTANT domain)
//===----------------------------------------------------------------------===//

def ERR_LOOM_CONSTANT_0001 : Loom_Error<"CONSTANT", "0001"> {
  let summary = [{Dense attribute shape must match result tile shape.}];
  let description = [{
    Triggered when a dense constant attribute has a shape that does not match
    the result tile's shape. For loom.tile.constant, the dense elements must
    form the exact shape of the result tile.
  }];
  let message = [{
    constant attribute has type '{attrType}' but result tile has type '{resultType}'
  }];
  let schema = (ins
    "Type":$attrType,
    "Type":$resultType
  );
  let fixHint = [{Ensure dense<...> elements match the shape of '{resultType}'}];
  let example = [{
    // Correct: shapes match
    %t = loom.tile.constant dense<[[1.0, 2.0]]> : !loom.tile<1x2xf32>
    // Invalid: shape mismatch (1x2 vs 2x2)
    %t = loom.tile.constant dense<[[1.0, 2.0]]> : !loom.tile<2x2xf32>
  }];
}

def ERR_LOOM_CONSTANT_0002 : Loom_Error<"CONSTANT", "0002"> {
  let summary = [{Constant attribute element type must match result tile element type.}];
  let description = [{
    Triggered when a constant attribute's element type does not match the
    result tile's element type. For dense attributes, the element type is
    extracted from the shaped type. For scalar splats, it's the scalar type.
  }];
  let message = [{
    constant has element type '{attrType}', expected '{resultType}'
  }];
  let schema = (ins
    "Type":$attrType,
    "Type":$resultType
  );
  let fixHint = [{Use values compatible with element type '{resultType}'}];
  let example = [{
    // Correct: f32 matches f32
    %t = loom.tile.constant 1.0 : !loom.tile<4xf32>
    // Invalid: i32 value with f32 tile
    %t = loom.tile.constant 1 : !loom.tile<4xf32>
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise Errors (ELEMENTWISE domain)
//===----------------------------------------------------------------------===//

def ERR_LOOM_ELEMENTWISE_0001 : Loom_Error<"ELEMENTWISE", "0001"> {
  let summary = [{Elementwise operation requires at least one input tile.}];
  let description = [{
    The loom.tile.elementwise operation must have at least one input tile
    to iterate over. An empty with() clause is not allowed.
  }];
  let message = [{elementwise operation requires at least one input tile}];
  let schema = (ins);
  let fixHint = [{Provide at least one input tile in the with() clause}];
  let example = [{
    // Correct: at least one input
    %r = loom.tile.elementwise(%e = %t : !loom.tile<4xf32>) {
      %neg = arith.negf %e : f32
      loom.tile.yield %neg : f32
    } -> !loom.tile<4xf32>
  }];
}

def ERR_LOOM_ELEMENTWISE_0002 : Loom_Error<"ELEMENTWISE", "0002"> {
  let summary = [{Input tiles must have identical shapes.}];
  let description = [{
    All input tiles to loom.tile.elementwise must have the same shape
    (rank and dimension sizes). Broadcasting is not supported.
  }];
  let message = [{
    input tile shapes do not match: '{firstType}' vs '{otherType}'
  }];
  let schema = (ins
    "Type":$firstType,
    "Type":$otherType
  );
  let fixHint = [{Ensure all input tiles have identical shapes, or use loom.tile.broadcast first}];
  let example = [{
    // Correct: same shapes
    %r = loom.tile.elementwise(%a = %x : !loom.tile<4x8xf32>,
                               %b = %y : !loom.tile<4x8xf32>) {
      %sum = arith.addf %a, %b : f32
      loom.tile.yield %sum : f32
    } -> !loom.tile<4x8xf32>
  }];
}

def ERR_LOOM_ELEMENTWISE_0003 : Loom_Error<"ELEMENTWISE", "0003"> {
  let summary = [{Result tile shape must match input tile shapes.}];
  let description = [{
    The result tile of loom.tile.elementwise must have the same shape as
    all input tiles (same rank and dimension sizes).
  }];
  let message = [{
    result tile shape '{resultType}' does not match input tile shape '{inputType}'
  }];
  let schema = (ins
    "Type":$resultType,
    "Type":$inputType
  );
  let fixHint = [{Ensure result type has the same shape as the input tiles}];
  let example = [{
    // Correct: result matches inputs
    %r = loom.tile.elementwise(%e = %t : !loom.tile<4x8xf32>) {
      %neg = arith.negf %e : f32
      loom.tile.yield %neg : f32
    } -> !loom.tile<4x8xf32>
  }];
}

def ERR_LOOM_ELEMENTWISE_0004 : Loom_Error<"ELEMENTWISE", "0004"> {
  let summary = [{Region block argument count mismatch.}];
  let description = [{
    The region block must have exactly one argument per input tile.
    Each block argument represents the scalar element being iterated.
  }];
  let message = [{
    region has {actualCount} block arguments, expected {expectedCount} (one per input tile)
  }];
  let schema = (ins
    "unsigned":$actualCount,
    "unsigned":$expectedCount
  );
  let fixHint = [{Ensure the region has exactly one block argument per input tile}];
  let example = [{
    // Correct: 2 inputs = 2 block args
    %r = loom.tile.elementwise(%a = %x : !loom.tile<4xf32>,
                               %b = %y : !loom.tile<4xf32>) {
      %sum = arith.addf %a, %b : f32
      loom.tile.yield %sum : f32
    } -> !loom.tile<4xf32>
  }];
}

def ERR_LOOM_ELEMENTWISE_0005 : Loom_Error<"ELEMENTWISE", "0005"> {
  let summary = [{Block argument type does not match tile element type.}];
  let description = [{
    Each block argument must have the element type of its corresponding
    input tile. The block arguments represent scalar elements during iteration.
  }];
  let message = [{
    block argument {argIndex} has type '{argType}', expected element type '{elementType}'
  }];
  let schema = (ins
    "unsigned":$argIndex,
    "Type":$argType,
    "Type":$elementType
  );
  let fixHint = [{Block argument should have type '{elementType}' to match tile element type}];
  let example = [{
    // Correct: f32 element type -> f32 block arg
    %r = loom.tile.elementwise(%e = %t : !loom.tile<4xf32>) {
      %neg = arith.negf %e : f32
      loom.tile.yield %neg : f32
    } -> !loom.tile<4xf32>
  }];
}

def ERR_LOOM_ELEMENTWISE_0006 : Loom_Error<"ELEMENTWISE", "0006"> {
  let summary = [{Yield must produce exactly one value.}];
  let description = [{
    The loom.tile.yield terminator must yield exactly one scalar value,
    which becomes the output element at the current iteration position.
  }];
  let message = [{loom.tile.yield has {actualCount} operands, expected 1}];
  let schema = (ins
    "unsigned":$actualCount
  );
  let fixHint = [{Yield exactly one scalar value}];
  let example = [{
    // Correct: yield one value
    loom.tile.yield %result : f32
  }];
}

def ERR_LOOM_ELEMENTWISE_0007 : Loom_Error<"ELEMENTWISE", "0007"> {
  let summary = [{Yielded value type must match result tile element type.}];
  let description = [{
    The type of the value yielded by loom.tile.yield must match the
    element type of the result tile.
  }];
  let message = [{
    yielded value has type '{yieldType}', but result tile element type is '{resultElementType}'
  }];
  let schema = (ins
    "Type":$yieldType,
    "Type":$resultElementType
  );
  let fixHint = [{Convert yielded value to type '{resultElementType}'}];
  let example = [{
    // Correct: f32 yield matches f32 result element type
    %r = loom.tile.elementwise(%e = %t : !loom.tile<4xf32>) {
      %neg = arith.negf %e : f32
      loom.tile.yield %neg : f32
    } -> !loom.tile<4xf32>
  }];
}

//===----------------------------------------------------------------------===//
// Dimension Errors (DIM domain)
//===----------------------------------------------------------------------===//

def ERR_LOOM_DIM_0001 : Loom_Error<"DIM", "0001"> {
  let summary = [{Dimension index out of bounds.}];
  let description = [{
    The dimension index must be in the range [0, rank). For a tile or tensor of
    rank N, valid indices are 0, 1, ..., N-1.
  }];
  let message = [{
    dimension index {dimIndex} is out of bounds for type with rank {rank}
  }];
  let schema = (ins
    "int64_t":$dimIndex,
    "int64_t":$rank
  );
  let fixHint = [{Use an index in range [0, {rank})}];
  let example = [{
    // Valid: index 0 and 1 for rank-2 tile
    %d0 = loom.tile.dim %tile, 0 : !loom.tile<4x8xf32>
    %d1 = loom.tile.dim %tile, 1 : !loom.tile<4x8xf32>
    // Invalid: index 2 for rank-2 tile
    %d2 = loom.tile.dim %tile, 2 : !loom.tile<4x8xf32>  // ERROR
  }];
}

//===----------------------------------------------------------------------===//
// Subrange Errors (SUBRANGE domain)
//===----------------------------------------------------------------------===//
// Generic errors for subrange operations (slice, update, copy).
// These use '{operandName}' templating so the error messages describe which
// operand has the problem, making them reusable across all subrange ops.

def ERR_LOOM_SUBRANGE_0001 : Loom_Error<"SUBRANGE", "0001"> {
  let summary = [{Offset count must match operand rank.}];
  let description = [{
    The number of offsets provided must exactly match the rank of the
    corresponding operand. Each dimension needs exactly one offset.
    This applies to source offsets, target offsets, or any other offset array.
  }];
  let message = [{
    '{operandName}' offset count ({offsetCount}) does not match rank ({rank})
  }];
  let schema = (ins
    StringRefParameter<>:$operandName,
    "unsigned":$offsetCount,
    "int64_t":$rank
  );
  let fixHint = [{Provide exactly {rank} offsets for '{operandName}'}];
  let example = [{
    // Correct: 2 offsets for rank-2 operand
    %tile = loom.tile.slice %src[0, 0] : !loom.tile<256x256xf16> -> !loom.tile<64x64xf16>
    // Invalid: 1 offset for rank-2 operand
    %tile = loom.tile.slice %src[0] : !loom.tile<256x256xf16> -> !loom.tile<64x64xf16>
  }];
}

def ERR_LOOM_SUBRANGE_0002 : Loom_Error<"SUBRANGE", "0002"> {
  let summary = [{Operand ranks must match.}];
  let description = [{
    Subrange operations do not support rank reduction or expansion.
    The source and target (or result) must have the same rank.
  }];
  let message = [{
    '{operand1Name}' rank ({rank1}) does not match '{operand2Name}' rank ({rank2})
  }];
  let schema = (ins
    StringRefParameter<>:$operand1Name,
    "int64_t":$rank1,
    StringRefParameter<>:$operand2Name,
    "int64_t":$rank2
  );
  let fixHint = [{'{operand1Name}' and '{operand2Name}' must have the same rank}];
  let example = [{
    // Correct: both rank-2
    %tile = loom.tile.slice %src[0, 0] : !loom.tile<256x256xf16> -> !loom.tile<64x64xf16>
    // Invalid: rank-1 result from rank-2 source
    %tile = loom.tile.slice %src[0, 0] : !loom.tile<256x256xf16> -> !loom.tile<64xf16>
  }];
}

def ERR_LOOM_SUBRANGE_0003 : Loom_Error<"SUBRANGE", "0003"> {
  let summary = [{Size count must match operand rank.}];
  let description = [{
    The number of sizes provided must exactly match the rank of the
    corresponding operand. Each dimension needs exactly one size.
  }];
  let message = [{
    '{operandName}' size count ({sizeCount}) does not match rank ({rank})
  }];
  let schema = (ins
    StringRefParameter<>:$operandName,
    "unsigned":$sizeCount,
    "int64_t":$rank
  );
  let fixHint = [{Provide exactly {rank} sizes for '{operandName}'}];
  let example = [{
    // Correct: 2 sizes for rank-2 copy
    %r = loom.tile.copy %src[0, 0], %tgt[0, 0], [8, 8] : ...
    // Invalid: 1 size for rank-2 copy
    %r = loom.tile.copy %src[0, 0], %tgt[0, 0], [8] : ...
  }];
}

//===----------------------------------------------------------------------===//
// Slice Errors (SLICE domain) - DEPRECATED
//===----------------------------------------------------------------------===//
// These are kept for backward compatibility with existing constraints.
// New code should use ERR_LOOM_SUBRANGE_* instead.

def ERR_LOOM_SLICE_0001 : Loom_Error<"SLICE", "0001"> {
  let summary = [{Offset count must match source rank.}];
  let description = [{
    The number of offsets provided must exactly match the rank of the source
    tensor or tile. Each dimension needs exactly one offset.
  }];
  let message = [{
    offset count ({offsetCount}) does not match source rank ({sourceRank})
  }];
  let schema = (ins
    "unsigned":$offsetCount,
    "int64_t":$sourceRank
  );
  let fixHint = [{Provide exactly {sourceRank} offsets}];
  let example = [{
    // Correct: 2 offsets for rank-2 source
    %tile = loom.tensor.slice %src[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
    // Invalid: 1 offset for rank-2 source
    %tile = loom.tensor.slice %src[0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  }];
}

def ERR_LOOM_SLICE_0002 : Loom_Error<"SLICE", "0002"> {
  let summary = [{Result rank must match source rank.}];
  let description = [{
    Slice operations do not support rank reduction or expansion.
    The result must have the same rank as the source.
  }];
  let message = [{
    result rank ({resultRank}) does not match source rank ({sourceRank})
  }];
  let schema = (ins
    "int64_t":$resultRank,
    "int64_t":$sourceRank
  );
  let fixHint = [{Result must have rank {sourceRank}}];
  let example = [{
    // Correct: both rank-2
    %tile = loom.tensor.slice %src[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
    // Invalid: rank-1 result from rank-2 source
    %tile = loom.tensor.slice %src[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64xf16>
  }];
}

//===----------------------------------------------------------------------===//
// Update Errors (UPDATE domain) - DEPRECATED
//===----------------------------------------------------------------------===//
// These are kept for backward compatibility with existing constraints.
// New code should use ERR_LOOM_SUBRANGE_* instead.

def ERR_LOOM_UPDATE_0001 : Loom_Error<"UPDATE", "0001"> {
  let summary = [{Offset count must match target rank.}];
  let description = [{
    The number of offsets provided must exactly match the rank of the target
    tensor or tile. Each dimension needs exactly one offset.
  }];
  let message = [{
    offset count ({offsetCount}) does not match target rank ({targetRank})
  }];
  let schema = (ins
    "unsigned":$offsetCount,
    "int64_t":$targetRank
  );
  let fixHint = [{Provide exactly {targetRank} offsets}];
  let example = [{
    // Correct: 2 offsets for rank-2 target
    %r = loom.tensor.update %tile, %tensor[0, 0]
        : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
    // Invalid: 1 offset for rank-2 target
    %r = loom.tensor.update %tile, %tensor[0]
        : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  }];
}

def ERR_LOOM_UPDATE_0002 : Loom_Error<"UPDATE", "0002"> {
  let summary = [{Update rank must match target rank.}];
  let description = [{
    Update operations do not support rank reduction or expansion.
    The update source must have the same rank as the target.
  }];
  let message = [{
    update rank ({updateRank}) does not match target rank ({targetRank})
  }];
  let schema = (ins
    "int64_t":$updateRank,
    "int64_t":$targetRank
  );
  let fixHint = [{Update must have rank {targetRank}}];
  let example = [{
    // Correct: both rank-2
    %r = loom.tensor.update %tile, %tensor[0, 0]
        : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
    // Invalid: rank-1 update into rank-2 target
    %r = loom.tensor.update %tile, %tensor[0, 0]
        : !loom.tile<64xf16> -> !loom.tensor<256x256xf16>
  }];
}

//===----------------------------------------------------------------------===//
// Fold Remarks (FOLD domain)
//===----------------------------------------------------------------------===//
// Informational remarks emitted during canonicalization when operations
// are folded to poison due to provably invalid bounds or undefined behavior.

def ERR_LOOM_FOLD_0001 : Loom_Error<"FOLD", "0001"> {
  let severity = "remark";
  let summary = [{Operation folded to poison (generic).}];
  let description = [{
    Generic fallback remark for canonicalization folds to poison that don't fit
    a more specific error code. Prefer using specific codes (0002-0005) when
    the reason is well-categorized.

    Use this for complex or unusual UB cases, or when the reason requires
    a custom string explanation.
  }];
  let message = [{
    '{opName}' folded to poison: {reason}
  }];
  let schema = (ins
    StringRefParameter<>:$opName,
    StringRefParameter<>:$reason
  );
  let fixHint = [{
    Check the operation's constraints. This indicates undefined behavior
    in the source program.
  }];
  let example = [{
    // Complex constraint violation requiring custom explanation
    // %t = loom.some.op %src : ... -> ...
    // Remark: 'loom.some.op' folded to poison: specific reason here
  }];
}

def ERR_LOOM_FOLD_0002 : Loom_Error<"FOLD", "0002"> {
  let severity = "remark";
  let summary = [{Slice source operand is poison.}];
  let description = [{
    Emitted when a slice operation reads from a source operand that is poison.
    Reading from poison storage produces a poison result. This applies to
    loom.tensor.slice and loom.tile.slice operations.

    Note: For update operations, source poison does NOT propagate to the entire
    result - only the subregion being written would be affected.
  }];
  let message = [{
    '{opName}' folded to poison: source operand is poison
  }];
  let schema = (ins
    StringRefParameter<>:$opName,
    "Type":$sourceType
  );
  let fixHint = [{
    The source operand was determined to be poison at compile time.
    Check upstream operations that produce this value.
  }];
  let example = [{
    // %poisoned_src is ub.poison
    %tile = loom.tensor.slice %poisoned_src[0, 0]
        : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
    // Result: ub.poison : !loom.tile<64x64xf16>
  }];
}

def ERR_LOOM_FOLD_0003 : Loom_Error<"FOLD", "0003"> {
  let severity = "remark";
  let summary = [{Update target operand is poison.}];
  let description = [{
    Emitted when an update operation writes into a target operand that is poison.
    Writing INTO poison storage produces a poison result, regardless of the
    update tile's contents. This applies to loom.tensor.update and
    loom.tile.update operations.

    Semantics: Like LLVM, updating a poison buffer yields poison. The storage
    location itself being poison means the entire result is poison.
  }];
  let message = [{
    '{opName}' folded to poison: target operand is poison
  }];
  let schema = (ins
    StringRefParameter<>:$opName,
    "Type":$targetType
  );
  let fixHint = [{
    The target operand was determined to be poison at compile time.
    Check upstream operations that produce this value.
  }];
  let example = [{
    // %poisoned_target is ub.poison
    %result = loom.tensor.update %tile, %poisoned_target[0, 0]
        : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
    // Result: ub.poison : !loom.tensor<256x256xf16>
  }];
}

def ERR_LOOM_FOLD_0004 : Loom_Error<"FOLD", "0004"> {
  let severity = "remark";
  let summary = [{Subrange access extends beyond bounds.}];
  let description = [{
    Emitted when a slice or update operation is detected to have an out-of-bounds
    access at compile time. Specifically, when offset + size > dimension bound
    for any dimension, the access is undefined behavior.

    This is a remark because the operation is folded to poison, indicating UB
    in the source program that should be investigated.
  }];
  let message = [{
    '{opName}' folded to poison: dimension {dimIndex} access at offset {offset} + size {size} = {total} exceeds bound {bound}
  }];
  let schema = (ins
    StringRefParameter<>:$opName,
    "int64_t":$dimIndex,
    "int64_t":$offset,
    "int64_t":$size,
    "int64_t":$total,
    "int64_t":$bound
  );
  let fixHint = [{
    Ensure offset + size <= dimension bound for all dimensions.
    Dimension {dimIndex}: {offset} + {size} = {total}, but bound is {bound}.
  }];
  let example = [{
    // Invalid: offset 192 + size 128 = 320 > 256
    %tile = loom.tensor.slice %src[192, 0]
        : !loom.tensor<256x256xf16> -> !loom.tile<128x64xf16>
    // Result: ub.poison : !loom.tile<128x64xf16>
  }];
}

def ERR_LOOM_FOLD_0005 : Loom_Error<"FOLD", "0005"> {
  let severity = "remark";
  let summary = [{Subrange access has negative offset.}];
  let description = [{
    Emitted when a slice or update operation has a negative offset at compile
    time. Offsets must be non-negative indices into the source/target.

    This is a remark because the operation is folded to poison, indicating UB
    in the source program that should be investigated.
  }];
  let message = [{
    '{opName}' folded to poison: dimension {dimIndex} has negative offset {offset}
  }];
  let schema = (ins
    StringRefParameter<>:$opName,
    "int64_t":$dimIndex,
    "int64_t":$offset
  );
  let fixHint = [{
    Ensure offset >= 0 for all dimensions.
    Dimension {dimIndex}: offset {offset} is negative.
  }];
  let example = [{
    // Invalid: negative offset -1
    // (In practice, negative offsets are rare in MLIR since indices
    // are typically unsigned, but can occur through arithmetic)
  }];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_LOOM_ERRORS_TD
