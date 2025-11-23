// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

func.func @error_shape_mismatch_nested() -> !loom.tile<2x2xf32> {
  // Dense attribute has 1 row but tile expects 2 rows - detected as jagged array.
  // expected-error @+1 {{jagged array: inconsistent dimension sizes}}
  %t = loom.tile.constant #loom.dense<[[1.0, 2.0]]> : !loom.tile<2x2xf32>
  return %t : !loom.tile<2x2xf32>
}

// -----

func.func @error_shape_mismatch_flat() -> !loom.tile<4xf32> {
  // Dense attribute has 2 elements but tile expects 4.
  // expected-error @+1 {{dense values shape [2] doesn't match tile shape [4]}}
  %t = loom.tile.constant #loom.dense<[1.0, 2.0]> : !loom.tile<4xf32>
  return %t : !loom.tile<4xf32>
}

// -----

func.func @error_rank_mismatch() -> !loom.tile<2x2xf32> {
  // Dense attribute is 1D [4] but tile is 2D [2,2].
  // expected-error @+1 {{dense values shape [4] doesn't match tile shape [2, 2]}}
  %t = loom.tile.constant #loom.dense<[1.0, 2.0, 3.0, 4.0]> : !loom.tile<2x2xf32>
  return %t : !loom.tile<2x2xf32>
}
