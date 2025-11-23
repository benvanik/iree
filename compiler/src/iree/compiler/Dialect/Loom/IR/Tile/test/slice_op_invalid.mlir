// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

func.func @error_offset_count_too_few(%src: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // Only 1 offset for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SLICE_0001: offset count (1) does not match source rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets}}
  // expected-note @+1 {{Example:}}
  %subtile = loom.tile.slice %src[0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

func.func @error_offset_count_too_many(%src: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // 3 offsets for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_SLICE_0001: offset count (3) does not match source rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets}}
  // expected-note @+1 {{Example:}}
  %subtile = loom.tile.slice %src[0, 0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

func.func @error_rank_mismatch(%src: !loom.tile<64x64xf16>) -> !loom.tile<16xf16> {
  // Result tile has rank 1, source tile has rank 2.
  // expected-error @+2 {{ERR_LOOM_SLICE_0002: result rank (1) does not match source rank (2)}}
  // expected-note @+1 {{Fix: Result must have rank 2}}
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16xf16>
  return %subtile : !loom.tile<16xf16>
}

// -----

func.func @error_rank_mismatch_3d_to_2d(%src: !loom.tile<8x64x64xf16>) -> !loom.tile<16x16xf16> {
  // Result tile has rank 2, source tile has rank 3.
  // expected-error @+2 {{ERR_LOOM_SLICE_0002: result rank (2) does not match source rank (3)}}
  // expected-note @+1 {{Fix: Result must have rank 3}}
  %subtile = loom.tile.slice %src[0, 0, 0] : !loom.tile<8x64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

func.func @error_element_type_mismatch(%src: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf32> {
  // Source is f16, result is f32.
  // expected-error @+3 {{ERR_LOOM_TYPE_0002: element type mismatch: source, result have different element types}}
  // expected-note @+2 {{Fix: Operands 'source, result' have element types f16, f32 - use explicit type casts if needed}}
  // expected-note @+1 {{Example:}}
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf32>
  return %subtile : !loom.tile<16x16xf32>
}

// -----

func.func @error_element_type_mismatch_int_float(%src: !loom.tile<64x64xi32>) -> !loom.tile<16x16xf32> {
  // Source is i32, result is f32.
  // expected-error @+3 {{ERR_LOOM_TYPE_0002: element type mismatch: source, result have different element types}}
  // expected-note @+2 {{Fix: Operands 'source, result' have element types i32, f32 - use explicit type casts if needed}}
  // expected-note @+1 {{Example:}}
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<64x64xi32> -> !loom.tile<16x16xf32>
  return %subtile : !loom.tile<16x16xf32>
}
