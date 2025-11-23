// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Constant Offsets
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_all_constant_offsets
func.func @fold_all_constant_offsets(%src: !loom.tensor<256x256xf16>) -> !loom.tile<64x64xf16> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: loom.tensor.slice %{{.+}}[10, 20] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[%c10, %c20] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @fold_partial_constant_offsets
func.func @fold_partial_constant_offsets(%src: !loom.tensor<256x256xf16>, %offset: index) -> !loom.tile<64x64xf16> {
  %c10 = arith.constant 10 : index
  // CHECK: loom.tensor.slice %{{.+}}[10, %{{.+}}] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[%c10, %offset] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_already_static
func.func @no_fold_already_static(%src: !loom.tensor<256x256xf16>) -> !loom.tile<64x64xf16> {
  // CHECK: loom.tensor.slice %{{.+}}[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @fold_with_dynamic_dims
func.func @fold_with_dynamic_dims(%src: !loom.tensor<?x?xf16>, %dim0: index, %dim1: index, %size0: index, %size1: index) -> !loom.tile<?x?xf16> {
  %c5 = arith.constant 5 : index
  %c15 = arith.constant 15 : index
  // CHECK: loom.tensor.slice %[[src:.+]][5, 15] : !loom.tensor<?x?xf16>{%[[dim0:.+]], %[[dim1:.+]]} -> !loom.tile<?x?xf16>{%[[size0:.+]], %[[size1:.+]]}
  %tile = loom.tensor.slice %src[%c5, %c15] : !loom.tensor<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%size0, %size1}
  return %tile : !loom.tile<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Slice of Poison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_poison
func.func @fold_slice_of_poison() -> !loom.tile<64x64xf16> {
  %poison = ub.poison : !loom.tensor<256x256xf16>
  // CHECK-NOT: loom.tensor.slice
  // CHECK: ub.poison : !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %poison[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Out-of-Bounds Slice to Poison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_out_of_bounds_slice
func.func @fold_out_of_bounds_slice(%src: !loom.tensor<256x256xf16>) -> !loom.tile<64x64xf16> {
  // offset + size > dim: 240 + 64 = 304 > 256
  // CHECK-NOT: loom.tensor.slice
  // CHECK: ub.poison : !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[240, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_in_bounds_slice
func.func @no_fold_in_bounds_slice(%src: !loom.tensor<256x256xf16>) -> !loom.tile<64x64xf16> {
  // offset + size <= dim: 192 + 64 = 256 <= 256
  // CHECK: loom.tensor.slice %{{.+}}[192, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[192, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Store-to-Load Forwarding (Slice of Update)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_update_same_offset
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_slice_of_update_same_offset(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tile<64x64xf16> {
  %updated = loom.tensor.update %tile, %tensor[32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  // CHECK-NOT: loom.tensor.slice
  // CHECK-NOT: loom.tensor.update
  // CHECK: return %[[TILE]] : !loom.tile<64x64xf16>
  %result = loom.tensor.slice %updated[32, 32] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_update_different_offset
func.func @no_fold_slice_of_update_different_offset(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tile<64x64xf16> {
  %updated = loom.tensor.update %tile, %tensor[32, 32] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  // Different offset - cannot forward
  // CHECK: loom.tensor.update
  // CHECK: loom.tensor.slice
  %result = loom.tensor.slice %updated[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Disjoint Slice of Update
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_update_disjoint
// CHECK-SAME: %[[TENSOR:[^:]+]]: !loom.tensor<256x256xf16>
func.func @fold_slice_of_update_disjoint(%tile: !loom.tile<32x32xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tile<32x32xf16> {
  // Update at [0,0] with 32x32, slice at [64,64] - disjoint
  %updated = loom.tensor.update %tile, %tensor[0, 0] : !loom.tile<32x32xf16> -> !loom.tensor<256x256xf16>
  // CHECK-NOT: loom.tensor.update
  // CHECK: loom.tensor.slice %[[TENSOR]][64, 64] : !loom.tensor<256x256xf16> -> !loom.tile<32x32xf16>
  %result = loom.tensor.slice %updated[64, 64] : !loom.tensor<256x256xf16> -> !loom.tile<32x32xf16>
  return %result : !loom.tile<32x32xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_update_overlapping
func.func @no_fold_slice_of_update_overlapping(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tile<32x32xf16> {
  // Update at [0,0] with 64x64, slice at [32,32] with 32x32 - overlapping
  %updated = loom.tensor.update %tile, %tensor[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  // Regions overlap, cannot bypass update
  // CHECK: loom.tensor.update
  // CHECK: loom.tensor.slice
  %result = loom.tensor.slice %updated[32, 32] : !loom.tensor<256x256xf16> -> !loom.tile<32x32xf16>
  return %result : !loom.tile<32x32xf16>
}
