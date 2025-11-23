// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_constant_dense
func.func @tile_constant_dense() -> !loom.tile<4xi32> {
  // Dense attribute with explicit values matching tile shape.
  //
  // Semantic meaning: Creating a constant tile with specific values,
  // useful for weights, masks, or initialization patterns.
  //
  // CHECK: loom.tile.constant #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
  %vector = loom.tile.constant #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
  return %vector : !loom.tile<4xi32>
}

// -----

// CHECK-LABEL: @tile_constant_splat
func.func @tile_constant_splat() -> !loom.tile<64x64xi32> {
  // Splat: a single value broadcast to all tile positions.
  // This is efficient storage and common for initialization.
  //
  // Semantic meaning: Creating a tile filled with zeros, often used
  // as an accumulator initialization for reductions or matmul.
  //
  // CHECK: loom.tile.constant #loom.dense<0> : !loom.tile<64x64xi32>
  %zeros = loom.tile.constant #loom.dense<0> : !loom.tile<64x64xi32>
  return %zeros : !loom.tile<64x64xi32>
}

// -----

// CHECK-LABEL: @tile_constant_scalar
func.func @tile_constant_scalar() -> !loom.tile<i32> {
  // Scalar (rank-0) tile constant.
  // A single value with no dimensions.
  //
  // Semantic meaning: Representing a scalar value as a tile type
  // for uniform handling in tile-based computations.
  //
  // CHECK: loom.tile.constant #loom.dense<42> : !loom.tile<i32>
  %scalar = loom.tile.constant #loom.dense<42> : !loom.tile<i32>
  return %scalar : !loom.tile<i32>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_constant_integer
func.func @tile_constant_integer() -> !loom.tile<4xi32> {
  // Integer element type with dense values.
  //
  // CHECK: loom.tile.constant #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
  %vector = loom.tile.constant #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
  return %vector : !loom.tile<4xi32>
}

// -----

// CHECK-LABEL: @tile_constant_integer_splat
func.func @tile_constant_integer_splat() -> !loom.tile<8x8xi64> {
  // Integer splat value.
  //
  // CHECK: loom.tile.constant #loom.dense<1> : !loom.tile<8x8xi64>
  %ones = loom.tile.constant #loom.dense<1> : !loom.tile<8x8xi64>
  return %ones : !loom.tile<8x8xi64>
}

// -----

// CHECK-LABEL: @tile_constant_1d
func.func @tile_constant_1d() -> !loom.tile<4xi32> {
  // 1D vector constant.
  //
  // CHECK: loom.tile.constant #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
  %vec = loom.tile.constant #loom.dense<[1, 2, 3, 4]> : !loom.tile<4xi32>
  return %vec : !loom.tile<4xi32>
}

// -----

// CHECK-LABEL: @tile_constant_2d
func.func @tile_constant_2d() -> !loom.tile<2x4xi32> {
  // 2D matrix constant with splat value.
  //
  // CHECK: loom.tile.constant #loom.dense<7> : !loom.tile<2x4xi32>
  %matrix = loom.tile.constant #loom.dense<7> : !loom.tile<2x4xi32>
  return %matrix : !loom.tile<2x4xi32>
}

// -----

// CHECK-LABEL: @tile_constant_dense_splat
func.func @tile_constant_dense_splat() -> !loom.tile<4x4xi32> {
  // Dense attribute with single splat value.
  //
  // CHECK: loom.tile.constant #loom.dense<1> : !loom.tile<4x4xi32>
  %ones = loom.tile.constant #loom.dense<1> : !loom.tile<4x4xi32>
  return %ones : !loom.tile<4x4xi32>
}

// -----

// CHECK-LABEL: @tile_constant_i8
func.func @tile_constant_i8() -> !loom.tile<4xi8> {
  // i8 constant for compact storage.
  //
  // CHECK: loom.tile.constant #loom.dense<0> : !loom.tile<4xi8>
  %zeros = loom.tile.constant #loom.dense<0> : !loom.tile<4xi8>
  return %zeros : !loom.tile<4xi8>
}
