// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @fold_static_dim
func.func @fold_static_dim(%tile: !loom.tile<64x128xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = loom.tile.dim %tile, %c0 : !loom.tile<64x128xf32>
  %dim1 = loom.tile.dim %tile, %c1 : !loom.tile<64x128xf32>
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 64 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 128 : index
  // CHECK: return %[[DIM0]], %[[DIM1]]
  return %dim0, %dim1 : index, index
}

// -----

// CHECK-LABEL: @fold_static_dim_rank3
func.func @fold_static_dim_rank3(%tile: !loom.tile<4x8x16xf32>) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = loom.tile.dim %tile, %c0 : !loom.tile<4x8x16xf32>
  %dim1 = loom.tile.dim %tile, %c1 : !loom.tile<4x8x16xf32>
  %dim2 = loom.tile.dim %tile, %c2 : !loom.tile<4x8x16xf32>
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 8 : index
  // CHECK-DAG: %[[DIM2:.+]] = arith.constant 16 : index
  // CHECK: return %[[DIM0]], %[[DIM1]], %[[DIM2]]
  return %dim0, %dim1, %dim2 : index, index, index
}

// -----

// CHECK-LABEL: @fold_mixed_static_dynamic
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x?xf32>
func.func @fold_mixed_static_dynamic(%tile: !loom.tile<64x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = loom.tile.dim %tile, %c0 : !loom.tile<64x?xf32>
  %dim1 = loom.tile.dim %tile, %c1 : !loom.tile<64x?xf32>
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 64 : index
  // Static dim folds, dynamic dim remains
  // CHECK: loom.tile.dim %[[TILE]], %c1 : !loom.tile<64x?xf32>
  // CHECK: return %[[DIM0]],
  return %dim0, %dim1 : index, index
}

// -----

// CHECK-LABEL: @no_fold_dynamic_index
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x128xf32>
// CHECK-SAME: %[[INDEX:[^:]+]]: index
func.func @no_fold_dynamic_index(%tile: !loom.tile<64x128xf32>, %index: index) -> index {
  // Cannot fold when index is not constant.
  // CHECK: loom.tile.dim %[[TILE]], %[[INDEX]]
  %dim = loom.tile.dim %tile, %index : !loom.tile<64x128xf32>
  return %dim : index
}

// -----

// CHECK-LABEL: @no_fold_all_dynamic
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<?x?xf32>
func.func @no_fold_all_dynamic(%tile: !loom.tile<?x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // Dynamic dims cannot fold without producer information.
  // CHECK: loom.tile.dim %[[TILE]], %c0 : !loom.tile<?x?xf32>
  // CHECK: loom.tile.dim %[[TILE]], %c1 : !loom.tile<?x?xf32>
  %dim0 = loom.tile.dim %tile, %c0 : !loom.tile<?x?xf32>
  %dim1 = loom.tile.dim %tile, %c1 : !loom.tile<?x?xf32>
  return %dim0, %dim1 : index, index
}

// -----

// CHECK-LABEL: @fold_dim_through_fill
// CHECK-SAME: %[[SIZE0:[^:]+]]: index
// CHECK-SAME: %[[SIZE1:[^:]+]]: index
func.func @fold_dim_through_fill(%size0: index, %size1: index) -> (index, index) {
  %cst = arith.constant 0.0 : f32
  %alloca = loom.tile.alloca : !loom.tile<?x?xf32>{%size0, %size1}
  %filled = loom.tile.fill %cst, %alloca : f32 -> !loom.tile<?x?xf32>{%size0, %size1}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = loom.tile.dim %filled, %c0 : !loom.tile<?x?xf32>
  %dim1 = loom.tile.dim %filled, %c1 : !loom.tile<?x?xf32>

  // Dynamic dims should resolve through ShapeAwareOpInterface.
  // CHECK: loom.tile.alloca
  // CHECK-NOT: loom.tile.dim
  // CHECK: return %[[SIZE0]], %[[SIZE1]]
  return %dim0, %dim1 : index, index
}

// -----

// CHECK-LABEL: @fold_dim_through_broadcast
func.func @fold_dim_through_broadcast(%in: !loom.tile<1x64xf32>) -> (index, index) {
  // Broadcast adds a leading dimension of size 8.
  %bcast = loom.tile.broadcast<right> %in : !loom.tile<1x64xf32> -> !loom.tile<8x1x64xf32>

  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index

  // First dim (8) and third dim (64) are both static - should fold.
  %dim0 = loom.tile.dim %bcast, %c0 : !loom.tile<8x1x64xf32>
  %dim2 = loom.tile.dim %bcast, %c2 : !loom.tile<8x1x64xf32>

  // CHECK: %c8 = arith.constant 8 : index
  // CHECK: %c64 = arith.constant 64 : index
  // CHECK: return %c8, %c64
  return %dim0, %dim2 : index, index
}
