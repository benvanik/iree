// RUN: iree-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @tensor_dim_fold_static
func.func @tensor_dim_fold_static(%tensor: !loom.tensor<64x128xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = loom.tensor.dim %tensor, %c0 : !loom.tensor<64x128xf32>
  %dim1 = loom.tensor.dim %tensor, %c1 : !loom.tensor<64x128xf32>
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 64 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 128 : index
  // CHECK: return %[[DIM0]], %[[DIM1]]
  return %dim0, %dim1 : index, index
}

// CHECK-LABEL: @tensor_dim_fold_mixed
// CHECK-SAME: (%[[TENSOR:.+]]: !loom.tensor<64x?xf32>)
func.func @tensor_dim_fold_mixed(%tensor: !loom.tensor<64x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = loom.tensor.dim %tensor, %c0 : !loom.tensor<64x?xf32>
  %dim1 = loom.tensor.dim %tensor, %c1 : !loom.tensor<64x?xf32>
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 64 : index
  // Static dimension folds to constant.
  // Dynamic dimension remains as dim op.
  // CHECK: %[[DIM1:.+]] = loom.tensor.dim %[[TENSOR]], %c1 : !loom.tensor<64x?xf32>
  // CHECK: return %[[DIM0]], %[[DIM1]]
  return %dim0, %dim1 : index, index
}

// CHECK-LABEL: @tensor_dim_no_fold_dynamic_index
// CHECK-SAME: (%[[TENSOR:.+]]: !loom.tensor<64x128xf32>, %[[IDX:.+]]: index)
func.func @tensor_dim_no_fold_dynamic_index(%tensor: !loom.tensor<64x128xf32>, %index: index) -> index {
  // Dynamic index prevents folding even with static shape.
  // CHECK: %[[DIM:.+]] = loom.tensor.dim %[[TENSOR]], %[[IDX]] : !loom.tensor<64x128xf32>
  %dim = loom.tensor.dim %tensor, %index : !loom.tensor<64x128xf32>
  // CHECK: return %[[DIM]]
  return %dim : index
}

// CHECK-LABEL: @tensor_dim_fold_all_static
func.func @tensor_dim_fold_all_static(%tensor: !loom.tensor<2x3x4x5xf32>) -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %dim0 = loom.tensor.dim %tensor, %c0 : !loom.tensor<2x3x4x5xf32>
  %dim1 = loom.tensor.dim %tensor, %c1 : !loom.tensor<2x3x4x5xf32>
  %dim2 = loom.tensor.dim %tensor, %c2 : !loom.tensor<2x3x4x5xf32>
  %dim3 = loom.tensor.dim %tensor, %c3 : !loom.tensor<2x3x4x5xf32>
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 3 : index
  // CHECK-DAG: %[[DIM2:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[DIM3:.+]] = arith.constant 5 : index
  // CHECK: return %[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]
  return %dim0, %dim1, %dim2, %dim3 : index, index, index, index
}
