// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

// Tests that source offset count mismatch is rejected (too few offsets).
func.func @error_source_offset_count_too_few(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Only 1 source offset for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SUBRANGE_0001: 'source' offset count (1) does not match rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets for 'source'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0], %target[0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that source offset count mismatch is rejected (too many offsets).
func.func @error_source_offset_count_too_many(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // 3 source offsets for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SUBRANGE_0001: 'source' offset count (3) does not match rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets for 'source'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0, 0, 0], %target[0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that target offset count mismatch is rejected (too few offsets).
func.func @error_target_offset_count_too_few(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Only 1 target offset for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SUBRANGE_0001: 'target' offset count (1) does not match rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets for 'target'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0, 0], %target[0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that target offset count mismatch is rejected (too many offsets).
func.func @error_target_offset_count_too_many(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // 3 target offsets for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SUBRANGE_0001: 'target' offset count (3) does not match rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets for 'target'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0, 0], %target[0, 0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that size count mismatch is rejected (too few sizes).
func.func @error_size_count_mismatch(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Only 1 size for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SUBRANGE_0003: 'source' size count (1) does not match rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 sizes for 'source'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0, 0], %target[0, 0], [8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that size count mismatch is rejected (too many sizes).
func.func @error_size_count_too_many(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // 3 sizes for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SUBRANGE_0003: 'source' size count (3) does not match rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 sizes for 'source'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0, 0], %target[0, 0], [8, 8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that rank mismatch between source and target is detected.
// The size count error fires first because sizes [8, 8] don't match source rank 1.
func.func @error_rank_mismatch(%source: !loom.tile<64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Source is rank-1, target is rank-2, sizes [8, 8] don't match source rank.
  // expected-error @+3 {{ERR_LOOM_SUBRANGE_0003: 'source' size count (2) does not match rank (1)}}
  // expected-note @+2 {{Fix: Provide exactly 1 sizes for 'source'}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0], %target[0, 0], [8, 8] : !loom.tile<64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that element type mismatch between source and target is rejected.
func.func @error_element_type_mismatch(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf32>) -> !loom.tile<64x64xf32> {
  // Source is f16, target is f32.
  // expected-error @+3 {{ERR_LOOM_TYPE_0002: element type mismatch: source, target have different element types}}
  // expected-note @+2 {{Fix: Operands 'source, target' have element types f16, f32 - use explicit type casts if needed}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0, 0], %target[0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf32>
  return %result : !loom.tile<64x64xf32>
}

// -----

// Tests that element type mismatch between integer and float is rejected.
func.func @error_element_type_mismatch_int_float(%source: !loom.tile<64x64xi32>, %target: !loom.tile<64x64xf32>) -> !loom.tile<64x64xf32> {
  // Source is i32, target is f32.
  // expected-error @+3 {{ERR_LOOM_TYPE_0002: element type mismatch: source, target have different element types}}
  // expected-note @+2 {{Fix: Operands 'source, target' have element types i32, f32 - use explicit type casts if needed}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.copy %source[0, 0], %target[0, 0], [8, 8] : !loom.tile<64x64xi32> -> !loom.tile<64x64xf32>
  return %result : !loom.tile<64x64xf32>
}
