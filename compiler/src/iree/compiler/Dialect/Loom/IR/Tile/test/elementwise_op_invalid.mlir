// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

// Note: Some error cases (0004, 0005, 0006) are difficult to trigger through
// the custom parser which automatically enforces block argument counts/types
// and yield operand count. These would only occur with generic MLIR format
// or programmatically constructed IR.

// ERR_LOOM_ELEMENTWISE_0002: Input tile shape mismatch.
func.func @error_input_shape_mismatch(%a: !loom.tile<4x8xf32>, %b: !loom.tile<8x4xf32>) -> !loom.tile<4x8xf32> {
  // Input tiles have different shapes.
  // expected-error @+3 {{ERR_LOOM_ELEMENTWISE_0002: input tile shapes do not match: '!loom.tile<4x8xf32>' vs '!loom.tile<8x4xf32>'}}
  // expected-note @+2 {{Fix: Ensure all input tiles have identical shapes, or use loom.tile.broadcast first}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.elementwise(%ea = %a : !loom.tile<4x8xf32>, %eb = %b : !loom.tile<8x4xf32>) {
    %sum = arith.addf %ea, %eb : f32
    loom.tile.yield %sum : f32
  } -> !loom.tile<4x8xf32>
  return %result : !loom.tile<4x8xf32>
}

// -----

// ERR_LOOM_ELEMENTWISE_0002: Input tile rank mismatch.
func.func @error_input_rank_mismatch(%a: !loom.tile<4x8xf32>, %b: !loom.tile<8xf32>) -> !loom.tile<4x8xf32> {
  // Input tiles have different ranks.
  // expected-error @+3 {{ERR_LOOM_ELEMENTWISE_0002: input tile shapes do not match: '!loom.tile<4x8xf32>' vs '!loom.tile<8xf32>'}}
  // expected-note @+2 {{Fix: Ensure all input tiles have identical shapes, or use loom.tile.broadcast first}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.elementwise(%ea = %a : !loom.tile<4x8xf32>, %eb = %b : !loom.tile<8xf32>) {
    %sum = arith.addf %ea, %eb : f32
    loom.tile.yield %sum : f32
  } -> !loom.tile<4x8xf32>
  return %result : !loom.tile<4x8xf32>
}

// -----

// ERR_LOOM_ELEMENTWISE_0003: Result shape mismatch.
func.func @error_result_shape_mismatch(%input: !loom.tile<4x8xf32>) -> !loom.tile<8x4xf32> {
  // Result tile has different shape than input.
  // expected-error @+3 {{ERR_LOOM_ELEMENTWISE_0003: result tile shape '!loom.tile<8x4xf32>' does not match input tile shape '!loom.tile<4x8xf32>'}}
  // expected-note @+2 {{Fix: Ensure result type has the same shape as the input tiles}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x8xf32>) {
    %neg = arith.negf %element : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<8x4xf32>
  return %result : !loom.tile<8x4xf32>
}

// -----

// ERR_LOOM_ELEMENTWISE_0003: Result rank mismatch.
func.func @error_result_rank_mismatch(%input: !loom.tile<4x8xf32>) -> !loom.tile<32xf32> {
  // Result tile has different rank than input.
  // expected-error @+3 {{ERR_LOOM_ELEMENTWISE_0003: result tile shape '!loom.tile<32xf32>' does not match input tile shape '!loom.tile<4x8xf32>'}}
  // expected-note @+2 {{Fix: Ensure result type has the same shape as the input tiles}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x8xf32>) {
    %neg = arith.negf %element : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<32xf32>
  return %result : !loom.tile<32xf32>
}

// -----

// ERR_LOOM_ELEMENTWISE_0007: Yielded type mismatch - float to int.
func.func @error_yield_type_mismatch_float_to_int(%input: !loom.tile<4x4xf32>) -> !loom.tile<4x4xi32> {
  // Yielded f32 value doesn't match result i32 element type.
  // expected-error @+3 {{ERR_LOOM_ELEMENTWISE_0007: yielded value has type 'f32', but result tile element type is 'i32'}}
  // expected-note @+2 {{Fix: Convert yielded value to type 'i32'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xf32>) {
    %neg = arith.negf %element : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<4x4xi32>
  return %result : !loom.tile<4x4xi32>
}

// -----

// ERR_LOOM_ELEMENTWISE_0007: Yielded type mismatch - int to float.
func.func @error_yield_type_mismatch_int_to_float(%input: !loom.tile<4x4xi32>) -> !loom.tile<4x4xf32> {
  // Yielded i32 value doesn't match result f32 element type.
  // expected-error @+3 {{ERR_LOOM_ELEMENTWISE_0007: yielded value has type 'i32', but result tile element type is 'f32'}}
  // expected-note @+2 {{Fix: Convert yielded value to type 'f32'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xi32>) {
    %neg = arith.muli %element, %element : i32
    loom.tile.yield %neg : i32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

// ERR_LOOM_ELEMENTWISE_0007: Yielded type mismatch - f32 to bf16.
func.func @error_yield_type_mismatch_f32_to_bf16(%input: !loom.tile<4x4xf32>) -> !loom.tile<4x4xbf16> {
  // Yielded f32 value doesn't match result bf16 element type.
  // expected-error @+3 {{ERR_LOOM_ELEMENTWISE_0007: yielded value has type 'f32', but result tile element type is 'bf16'}}
  // expected-note @+2 {{Fix: Convert yielded value to type 'bf16'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xf32>) {
    %neg = arith.negf %element : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<4x4xbf16>
  return %result : !loom.tile<4x4xbf16>
}
