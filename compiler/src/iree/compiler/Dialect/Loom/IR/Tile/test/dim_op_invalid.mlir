// RUN: iree-opt --split-input-file --verify-diagnostics %s

func.func @tile_dim_out_of_bounds(%tile: !loom.tile<64x128xf32>) -> index {
  %c2 = arith.constant 2 : index
  // expected-error @+3 {{dimension index 2 is out of bounds for type with rank 2}}
  // expected-note @+2 {{Fix: Use an index in range [0, 2)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tile.dim %tile, %c2 : !loom.tile<64x128xf32>
  return %dim : index
}

// -----

func.func @tile_dim_negative_index(%tile: !loom.tile<64x128xf32>) -> index {
  %cn1 = arith.constant -1 : index
  // expected-error @+3 {{dimension index -1 is out of bounds for type with rank 2}}
  // expected-note @+2 {{Fix: Use an index in range [0, 2)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tile.dim %tile, %cn1 : !loom.tile<64x128xf32>
  return %dim : index
}

// -----

func.func @tile_dim_scalar_out_of_bounds(%tile: !loom.tile<f32>) -> index {
  %c0 = arith.constant 0 : index
  // expected-error @+3 {{dimension index 0 is out of bounds for type with rank 0}}
  // expected-note @+2 {{Fix: Use an index in range [0, 0)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tile.dim %tile, %c0 : !loom.tile<f32>
  return %dim : index
}

// -----

func.func @tile_dim_large_index(%tile: !loom.tile<4x8x16xf32>) -> index {
  %c10 = arith.constant 10 : index
  // expected-error @+3 {{dimension index 10 is out of bounds for type with rank 3}}
  // expected-note @+2 {{Fix: Use an index in range [0, 3)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tile.dim %tile, %c10 : !loom.tile<4x8x16xf32>
  return %dim : index
}
