// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

func.func @error_type_mismatch_int_to_float(%value: i32, %target: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Integer value cannot fill a float tile.
  // expected-error @+3 {{ERR_LOOM_FILL_0001: fill value has type 'i32', but tile element type is 'f32'}}
  // expected-note @+2 {{Fix: Cast or convert fill value to type 'f32'}}
  // expected-note @+1 {{Example:}}
  %0 = loom.tile.fill %value, %target : i32 -> !loom.tile<4x4xf32>
  return %0 : !loom.tile<4x4xf32>
}

// -----

func.func @error_type_mismatch_float_to_int(%value: f32, %target: !loom.tile<8xi32>) -> !loom.tile<8xi32> {
  // Float value cannot fill an integer tile.
  // expected-error @+3 {{ERR_LOOM_FILL_0001: fill value has type 'f32', but tile element type is 'i32'}}
  // expected-note @+2 {{Fix: Cast or convert fill value to type 'i32'}}
  // expected-note @+1 {{Example:}}
  %0 = loom.tile.fill %value, %target : f32 -> !loom.tile<8xi32>
  return %0 : !loom.tile<8xi32>
}

// -----

func.func @error_type_mismatch_f32_to_bf16(%value: f32, %target: !loom.tile<4x4xbf16>) -> !loom.tile<4x4xbf16> {
  // f32 value cannot fill a bf16 tile - types must match exactly.
  // expected-error @+3 {{ERR_LOOM_FILL_0001: fill value has type 'f32', but tile element type is 'bf16'}}
  // expected-note @+2 {{Fix: Cast or convert fill value to type 'bf16'}}
  // expected-note @+1 {{Example:}}
  %0 = loom.tile.fill %value, %target : f32 -> !loom.tile<4x4xbf16>
  return %0 : !loom.tile<4x4xbf16>
}
