// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Constant Offsets
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_all_constant_offsets
func.func @fold_all_constant_offsets(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[10, 20] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[%c10, %c20] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @fold_partial_constant_offsets
func.func @fold_partial_constant_offsets(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>, %offset: index) -> !loom.tensor<256x256xf16> {
  %c10 = arith.constant 10 : index
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[10, %{{.+}}] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[%c10, %offset] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @no_fold_already_static
func.func @no_fold_already_static(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @fold_with_dynamic_dims
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<?x?xf16>
// CHECK-SAME: %[[TENSOR:[^:]+]]: !loom.tensor<?x?xf16>
// CHECK-SAME: %[[DIM0:[^:]+]]: index
// CHECK-SAME: %[[DIM1:[^:]+]]: index
// CHECK-SAME: %[[TSZ0:[^:]+]]: index
// CHECK-SAME: %[[TSZ1:[^:]+]]: index
func.func @fold_with_dynamic_dims(%tile: !loom.tile<?x?xf16>, %tensor: !loom.tensor<?x?xf16>, %dim0: index, %dim1: index, %tsz0: index, %tsz1: index) -> !loom.tensor<?x?xf16> {
  %c5 = arith.constant 5 : index
  %c15 = arith.constant 15 : index
  // CHECK: loom.tensor.update %[[TILE]], %[[TENSOR]][5, 15] : !loom.tile<?x?xf16>{%[[TSZ0]], %[[TSZ1]]} -> !loom.tensor<?x?xf16>{%[[DIM0]], %[[DIM1]]}
  %updated = loom.tensor.update %tile, %tensor[%c5, %c15] : !loom.tile<?x?xf16>{%tsz0, %tsz1} -> !loom.tensor<?x?xf16>{%dim0, %dim1}
  return %updated : !loom.tensor<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Update of Poison
//===----------------------------------------------------------------------===//

// Poison tile does NOT poison the entire tensor - only the updated subregion.
// This matches LLVM semantics where storing poison only affects stored bytes.
// CHECK-LABEL: @no_fold_update_of_poison_tile
func.func @no_fold_update_of_poison_tile(%tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  %poison_tile = ub.poison : !loom.tile<64x64xf16>
  // CHECK: %[[POISON:.+]] = ub.poison : !loom.tile<64x64xf16>
  // CHECK: loom.tensor.update %[[POISON]], %{{.+}}[0, 0]
  %updated = loom.tensor.update %poison_tile, %tensor[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// Poison target DOES fold - the entire tensor was already poison.
// CHECK-LABEL: @fold_update_of_poison_target
func.func @fold_update_of_poison_target(%tile: !loom.tile<64x64xf16>) -> !loom.tensor<256x256xf16> {
  %poison_tensor = ub.poison : !loom.tensor<256x256xf16>
  // CHECK-NOT: loom.tensor.update
  // CHECK: ub.poison : !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %poison_tensor[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Out-of-Bounds Update to Poison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_out_of_bounds_update
func.func @fold_out_of_bounds_update(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  // offset + size > dim: 240 + 64 = 304 > 256
  // CHECK-NOT: loom.tensor.update
  // CHECK: ub.poison : !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[240, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @no_fold_in_bounds_update
func.func @no_fold_in_bounds_update(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  // offset + size <= dim: 192 + 64 = 256 <= 256
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[192, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[192, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Round-Trip Identity (Update of Slice)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_update_of_slice_roundtrip
// CHECK-SAME: %[[TENSOR:[^:]+]]: !loom.tensor<256x256xf16>
func.func @fold_update_of_slice_roundtrip(%tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  %tile = loom.tensor.slice %tensor[32, 32] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  // CHECK-NOT: loom.tensor.slice
  // CHECK-NOT: loom.tensor.update
  // CHECK: return %[[TENSOR]] : !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @no_fold_update_of_slice_different_offset
func.func @no_fold_update_of_slice_different_offset(%tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  %tile = loom.tensor.slice %tensor[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  // Different offset - not a round-trip
  // CHECK: loom.tensor.slice
  // CHECK: loom.tensor.update
  %updated = loom.tensor.update %tile, %tensor[32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @no_fold_update_of_slice_different_tensor
func.func @no_fold_update_of_slice_different_tensor(%tensor1: !loom.tensor<256x256xf16>, %tensor2: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  %tile = loom.tensor.slice %tensor1[32, 32] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  // Different tensor - not a round-trip
  // CHECK: loom.tensor.slice
  // CHECK: loom.tensor.update
  %updated = loom.tensor.update %tile, %tensor2[32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Dead Store Elimination (Update over Update)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_update_over_update_same_offset
// CHECK-SAME: %[[TILE1:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TILE2:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TENSOR:[^:]+]]: !loom.tensor<256x256xf16>
func.func @fold_update_over_update_same_offset(%tile1: !loom.tile<64x64xf16>, %tile2: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  %u1 = loom.tensor.update %tile1, %tensor[32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  // CHECK-NOT: %[[TILE1]]
  // CHECK: loom.tensor.update %[[TILE2]], %[[TENSOR]][32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %u2 = loom.tensor.update %tile2, %u1[32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %u2 : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @no_fold_update_over_update_different_offset
// CHECK-SAME: %[[TILE1:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TILE2:[^:]+]]: !loom.tile<64x64xf16>
func.func @no_fold_update_over_update_different_offset(%tile1: !loom.tile<64x64xf16>, %tile2: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  %u1 = loom.tensor.update %tile1, %tensor[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  // Different offsets - both updates needed
  // CHECK: loom.tensor.update %[[TILE1]]
  // CHECK: loom.tensor.update %[[TILE2]]
  %u2 = loom.tensor.update %tile2, %u1[128, 128] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %u2 : !loom.tensor<256x256xf16>
}
