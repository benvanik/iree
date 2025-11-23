// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

func.func @error_offset_count_too_few(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Only 1 offset for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_UPDATE_0001: offset count (1) does not match target rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets}}
  // expected-note @+1 {{Example:}}
  %updated = loom.tile.update %subtile, %tile[0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

func.func @error_offset_count_too_many(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // 3 offsets for rank-2 tile.
  // expected-error @+3 {{ERR_LOOM_UPDATE_0001: offset count (3) does not match target rank (2)}}
  // expected-note @+2 {{Fix: Provide exactly 2 offsets}}
  // expected-note @+1 {{Example:}}
  %updated = loom.tile.update %subtile, %tile[0, 0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

func.func @error_rank_mismatch(%subtile: !loom.tile<16xf16>, %tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Update tile has rank 1, target tile has rank 2.
  // expected-error @+2 {{ERR_LOOM_UPDATE_0002: update rank (1) does not match target rank (2)}}
  // expected-note @+1 {{Fix: Update must have rank 2}}
  %updated = loom.tile.update %subtile, %tile[0, 0] : !loom.tile<16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

func.func @error_rank_mismatch_2d_to_3d(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<8x64x64xf16>) -> !loom.tile<8x64x64xf16> {
  // Update tile has rank 2, target tile has rank 3.
  // expected-error @+2 {{ERR_LOOM_UPDATE_0002: update rank (2) does not match target rank (3)}}
  // expected-note @+1 {{Fix: Update must have rank 3}}
  %updated = loom.tile.update %subtile, %tile[0, 0, 0] : !loom.tile<16x16xf16> -> !loom.tile<8x64x64xf16>
  return %updated : !loom.tile<8x64x64xf16>
}

// -----

func.func @error_element_type_mismatch(%subtile: !loom.tile<16x16xf32>, %tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Update tile is f32, target tile is f16.
  // expected-error @+3 {{ERR_LOOM_TYPE_0002: element type mismatch: update, target have different element types}}
  // expected-note @+2 {{Fix: Operands 'update, target' have element types f32, f16 - use explicit type casts if needed}}
  // expected-note @+1 {{Example:}}
  %updated = loom.tile.update %subtile, %tile[0, 0] : !loom.tile<16x16xf32> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

func.func @error_element_type_mismatch_int_float(%subtile: !loom.tile<16x16xi32>, %tile: !loom.tile<64x64xf32>) -> !loom.tile<64x64xf32> {
  // Update tile is i32, target tile is f32.
  // expected-error @+3 {{ERR_LOOM_TYPE_0002: element type mismatch: update, target have different element types}}
  // expected-note @+2 {{Fix: Operands 'update, target' have element types i32, f32 - use explicit type casts if needed}}
  // expected-note @+1 {{Example:}}
  %updated = loom.tile.update %subtile, %tile[0, 0] : !loom.tile<16x16xi32> -> !loom.tile<64x64xf32>
  return %updated : !loom.tile<64x64xf32>
}
