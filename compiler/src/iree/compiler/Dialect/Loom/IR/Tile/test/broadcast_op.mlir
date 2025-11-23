// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_broadcast_right_align
func.func @tile_broadcast_right_align(%input: !loom.tile<3x4xf32>) -> !loom.tile<2x3x4xf32> {
  // Right-aligned broadcast (default): source dimensions occupy trailing positions.
  // Input shape: <3x4>
  // Result shape: <2x3x4>
  // Source dims align to positions [1, 2], new dimension added at position [0].
  // This prepends a new leading dimension of size 2.
  //
  // Semantic meaning: Adding a batch dimension to a 3x4 tile, creating 2 copies.
  // Memory layout: Encoding preserved, allowing efficient implementation.
  //
  // CHECK: loom.tile.broadcast<right>
  %batch_tile = loom.tile.broadcast<right> %input : !loom.tile<3x4xf32> -> !loom.tile<2x3x4xf32>
  return %batch_tile : !loom.tile<2x3x4xf32>
}

// -----

// CHECK-LABEL: @tile_broadcast_left_align
func.func @tile_broadcast_left_align(%input: !loom.tile<2x3xf32>) -> !loom.tile<2x3x4xf32> {
  // Left-aligned broadcast: source dimensions occupy leading positions.
  // Input shape: <2x3>
  // Result shape: <2x3x4>
  // Source dims align to positions [0, 1], new dimension added at position [2].
  // This appends a new trailing dimension of size 4.
  //
  // Semantic meaning: Extending a 2x3 tile with an additional dimension,
  // useful for broadcasting across inner dimensions.
  //
  // CHECK: loom.tile.broadcast<left>
  %extended_tile = loom.tile.broadcast<left> %input : !loom.tile<2x3xf32> -> !loom.tile<2x3x4xf32>
  return %extended_tile : !loom.tile<2x3x4xf32>
}

// -----

// CHECK-LABEL: @tile_broadcast_unit_dim
func.func @tile_broadcast_unit_dim(%row_vector: !loom.tile<1x4xf32>) -> !loom.tile<3x4xf32> {
  // Broadcasting with unit dimension: dimension of size 1 is replicated.
  // Input shape: <1x4>
  // Result shape: <3x4>
  // First dimension (size 1) broadcasts to size 3.
  //
  // Semantic meaning: Replicating a row vector across multiple rows,
  // commonly used for broadcasting scalars or vectors in element-wise operations.
  // The implementation may optimize this by reusing memory.
  //
  // CHECK: loom.tile.broadcast<left>
  %matrix = loom.tile.broadcast<left> %row_vector : !loom.tile<1x4xf32> -> !loom.tile<3x4xf32>
  return %matrix : !loom.tile<3x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_broadcast_scalar_to_matrix
func.func @tile_broadcast_scalar_to_matrix(%scalar: !loom.tile<f32>) -> !loom.tile<3x4xf32> {
  // Broadcasting a rank-0 (scalar) tile to rank-2.
  // Demonstrates maximum rank increase and unit dimension broadcasting.
  //
  // CHECK: loom.tile.broadcast<left>
  %matrix = loom.tile.broadcast<left> %scalar : !loom.tile<f32> -> !loom.tile<3x4xf32>
  return %matrix : !loom.tile<3x4xf32>
}
