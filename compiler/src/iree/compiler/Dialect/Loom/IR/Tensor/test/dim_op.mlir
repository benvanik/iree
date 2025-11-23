// RUN: iree-opt %s | FileCheck %s

// CHECK-LABEL: @tensor_dim_static
// CHECK-SAME: (%[[TENSOR:.+]]: !loom.tensor<64x128xf32>)
func.func @tensor_dim_static(%tensor: !loom.tensor<64x128xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[DIM0:.+]] = loom.tensor.dim %[[TENSOR]], %c0 : !loom.tensor<64x128xf32>
  %dim0 = loom.tensor.dim %tensor, %c0 : !loom.tensor<64x128xf32>
  // CHECK: %[[DIM1:.+]] = loom.tensor.dim %[[TENSOR]], %c1 : !loom.tensor<64x128xf32>
  %dim1 = loom.tensor.dim %tensor, %c1 : !loom.tensor<64x128xf32>
  // CHECK: return %[[DIM0]], %[[DIM1]]
  return %dim0, %dim1 : index, index
}

// CHECK-LABEL: @tensor_dim_dynamic
// CHECK-SAME: (%[[TENSOR:.+]]: !loom.tensor<?x?xf32>)
func.func @tensor_dim_dynamic(%tensor: !loom.tensor<?x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[DIM0:.+]] = loom.tensor.dim %[[TENSOR]], %c0 : !loom.tensor<?x?xf32>
  %dim0 = loom.tensor.dim %tensor, %c0 : !loom.tensor<?x?xf32>
  // CHECK: %[[DIM1:.+]] = loom.tensor.dim %[[TENSOR]], %c1 : !loom.tensor<?x?xf32>
  %dim1 = loom.tensor.dim %tensor, %c1 : !loom.tensor<?x?xf32>
  // CHECK: return %[[DIM0]], %[[DIM1]]
  return %dim0, %dim1 : index, index
}

// CHECK-LABEL: @tensor_dim_mixed
// CHECK-SAME: (%[[TENSOR:.+]]: !loom.tensor<64x?x128xf32>)
func.func @tensor_dim_mixed(%tensor: !loom.tensor<64x?x128xf32>) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: loom.tensor.dim %[[TENSOR]], %c0 : !loom.tensor<64x?x128xf32>
  %dim0 = loom.tensor.dim %tensor, %c0 : !loom.tensor<64x?x128xf32>
  // CHECK: loom.tensor.dim %[[TENSOR]], %c1 : !loom.tensor<64x?x128xf32>
  %dim1 = loom.tensor.dim %tensor, %c1 : !loom.tensor<64x?x128xf32>
  // CHECK: loom.tensor.dim %[[TENSOR]], %c2 : !loom.tensor<64x?x128xf32>
  %dim2 = loom.tensor.dim %tensor, %c2 : !loom.tensor<64x?x128xf32>
  return %dim0, %dim1, %dim2 : index, index, index
}

// CHECK-LABEL: @tensor_dim_ssa_index
// CHECK-SAME: (%[[TENSOR:.+]]: !loom.tensor<?x?xf32>, %[[IDX:.+]]: index)
func.func @tensor_dim_ssa_index(%tensor: !loom.tensor<?x?xf32>, %index: index) -> index {
  // CHECK: %[[DIM:.+]] = loom.tensor.dim %[[TENSOR]], %[[IDX]] : !loom.tensor<?x?xf32>
  %dim = loom.tensor.dim %tensor, %index : !loom.tensor<?x?xf32>
  // CHECK: return %[[DIM]]
  return %dim : index
}

// CHECK-LABEL: @tensor_dim_high_rank
// CHECK-SAME: (%[[TENSOR:.+]]: !loom.tensor<2x3x4x5x6xf32>)
func.func @tensor_dim_high_rank(%tensor: !loom.tensor<2x3x4x5x6xf32>) -> index {
  %c4 = arith.constant 4 : index
  // CHECK: loom.tensor.dim %[[TENSOR]], %c4 : !loom.tensor<2x3x4x5x6xf32>
  %dim4 = loom.tensor.dim %tensor, %c4 : !loom.tensor<2x3x4x5x6xf32>
  return %dim4 : index
}
