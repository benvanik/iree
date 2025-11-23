// RUN: iree-opt %s --split-input-file --verify-diagnostics

func.func @tensor_dim_index_negative(%tensor: !loom.tensor<64x128xf32>) -> index {
  %cn1 = arith.constant -1 : index
  // expected-error @+3 {{dimension index -1 is out of bounds for type with rank 2}}
  // expected-note @+2 {{Fix: Use an index in range [0, 2)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tensor.dim %tensor, %cn1 : !loom.tensor<64x128xf32>
  return %dim : index
}

// -----

func.func @tensor_dim_index_equal_rank(%tensor: !loom.tensor<64x128xf32>) -> index {
  %c2 = arith.constant 2 : index
  // expected-error @+3 {{dimension index 2 is out of bounds for type with rank 2}}
  // expected-note @+2 {{Fix: Use an index in range [0, 2)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tensor.dim %tensor, %c2 : !loom.tensor<64x128xf32>
  return %dim : index
}

// -----

func.func @tensor_dim_index_greater_than_rank(%tensor: !loom.tensor<64x128xf32>) -> index {
  %c10 = arith.constant 10 : index
  // expected-error @+3 {{dimension index 10 is out of bounds for type with rank 2}}
  // expected-note @+2 {{Fix: Use an index in range [0, 2)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tensor.dim %tensor, %c10 : !loom.tensor<64x128xf32>
  return %dim : index
}

// -----

func.func @tensor_dim_index_out_of_bounds_rank1(%tensor: !loom.tensor<64xf32>) -> index {
  %c1 = arith.constant 1 : index
  // expected-error @+3 {{dimension index 1 is out of bounds for type with rank 1}}
  // expected-note @+2 {{Fix: Use an index in range [0, 1)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tensor.dim %tensor, %c1 : !loom.tensor<64xf32>
  return %dim : index
}

// -----

func.func @tensor_dim_index_out_of_bounds_high_rank(%tensor: !loom.tensor<2x3x4x5xf32>) -> index {
  %c4 = arith.constant 4 : index
  // expected-error @+3 {{dimension index 4 is out of bounds for type with rank 4}}
  // expected-note @+2 {{Fix: Use an index in range [0, 4)}}
  // expected-note @+1 {{Example:}}
  %dim = loom.tensor.dim %tensor, %c4 : !loom.tensor<2x3x4x5xf32>
  return %dim : index
}
