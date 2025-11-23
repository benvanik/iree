// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_dim_static
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x128xf32>
func.func @tile_dim_static(%tile: !loom.tile<64x128xf32>) -> index {
  // Query dimension of a statically-shaped tile.
  // The dimension index is a constant SSA value.
  //
  // CHECK: loom.tile.dim %[[TILE]], %c0 : !loom.tile<64x128xf32>
  %c0 = arith.constant 0 : index
  %dim = loom.tile.dim %tile, %c0 : !loom.tile<64x128xf32>
  return %dim : index
}

// -----

// CHECK-LABEL: @tile_dim_dynamic
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<?x?xf32>
func.func @tile_dim_dynamic(%tile: !loom.tile<?x?xf32>) -> index {
  // Query dimension of a dynamically-shaped tile.
  //
  // CHECK: loom.tile.dim %[[TILE]], %c1 : !loom.tile<?x?xf32>
  %c1 = arith.constant 1 : index
  %dim = loom.tile.dim %tile, %c1 : !loom.tile<?x?xf32>
  return %dim : index
}

// -----

// CHECK-LABEL: @tile_dim_fold
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<32x64xf32>
func.func @tile_dim_fold(%tile: !loom.tile<32x64xf32>) -> (index, index) {
  // Both dimensions are static, so these can fold to constants.
  //
  // CHECK: loom.tile.dim %[[TILE]], %c0 : !loom.tile<32x64xf32>
  // CHECK: loom.tile.dim %[[TILE]], %c1 : !loom.tile<32x64xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = loom.tile.dim %tile, %c0 : !loom.tile<32x64xf32>
  %dim1 = loom.tile.dim %tile, %c1 : !loom.tile<32x64xf32>
  return %dim0, %dim1 : index, index
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_dim_dynamic_index
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x128xf32>
// CHECK-SAME: %[[INDEX:[^:]+]]: index
func.func @tile_dim_dynamic_index(%tile: !loom.tile<64x128xf32>, %index: index) -> index {
  // Query with a dynamic index value.
  //
  // CHECK: loom.tile.dim %[[TILE]], %[[INDEX]] : !loom.tile<64x128xf32>
  %dim = loom.tile.dim %tile, %index : !loom.tile<64x128xf32>
  return %dim : index
}

// -----

// CHECK-LABEL: @tile_dim_rank0
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<f32>
// CHECK-SAME: %[[INDEX:[^:]+]]: index
func.func @tile_dim_rank0(%tile: !loom.tile<f32>, %index: index) -> index {
  // Rank-0 (scalar) tile - dynamic index allows runtime error.
  //
  // CHECK: loom.tile.dim %[[TILE]], %[[INDEX]] : !loom.tile<f32>
  %dim = loom.tile.dim %tile, %index : !loom.tile<f32>
  return %dim : index
}

// -----

// CHECK-LABEL: @tile_dim_mixed_shape
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x?x128xf32>
func.func @tile_dim_mixed_shape(%tile: !loom.tile<64x?x128xf32>) -> (index, index) {
  // Tile with mixed static/dynamic dimensions.
  //
  // CHECK: loom.tile.dim %[[TILE]], %c0 : !loom.tile<64x?x128xf32>
  // CHECK: loom.tile.dim %[[TILE]], %c1 : !loom.tile<64x?x128xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = loom.tile.dim %tile, %c0 : !loom.tile<64x?x128xf32>
  %dim1 = loom.tile.dim %tile, %c1 : !loom.tile<64x?x128xf32>
  return %dim0, %dim1 : index, index
}
