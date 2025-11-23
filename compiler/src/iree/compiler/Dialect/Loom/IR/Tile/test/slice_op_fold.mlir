// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Constant Offsets
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_all_constant_offsets
func.func @fold_all_constant_offsets(%src: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: loom.tile.slice %{{.+}}[10, 20] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  %subtile = loom.tile.slice %src[%c10, %c20] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @fold_partial_constant_offsets
func.func @fold_partial_constant_offsets(%src: !loom.tile<64x64xf16>, %offset: index) -> !loom.tile<16x16xf16> {
  %c10 = arith.constant 10 : index
  // CHECK: loom.tile.slice %{{.+}}[10, %{{.+}}] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  %subtile = loom.tile.slice %src[%c10, %offset] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @no_fold_already_static
func.func @no_fold_already_static(%src: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // CHECK: loom.tile.slice %{{.+}}[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @fold_with_dynamic_dims
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<?x?xf16>
// CHECK-SAME: %[[DIM0:[^:]+]]: index
// CHECK-SAME: %[[DIM1:[^:]+]]: index
// CHECK-SAME: %[[SIZE0:[^:]+]]: index
// CHECK-SAME: %[[SIZE1:[^:]+]]: index
func.func @fold_with_dynamic_dims(%src: !loom.tile<?x?xf16>, %dim0: index, %dim1: index, %size0: index, %size1: index) -> !loom.tile<?x?xf16> {
  %c5 = arith.constant 5 : index
  %c15 = arith.constant 15 : index
  // CHECK: loom.tile.slice %[[SRC]][5, 15] : !loom.tile<?x?xf16>{%[[DIM0]], %[[DIM1]]} -> !loom.tile<?x?xf16>{%[[SIZE0]], %[[SIZE1]]}
  %subtile = loom.tile.slice %src[%c5, %c15] : !loom.tile<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%size0, %size1}
  return %subtile : !loom.tile<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Folding: Identity Slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_identity_slice
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_identity_slice(%src: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Identity slice (same shape, zero offset) folds away.
  // CHECK-NOT: loom.tile.slice
  // CHECK: return %[[SRC]] : !loom.tile<64x64xf16>
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %subtile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @fold_identity_slice_dynamic
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<?x?xf16>
func.func @fold_identity_slice_dynamic(%src: !loom.tile<?x?xf16>, %dim0: index, %dim1: index) -> !loom.tile<?x?xf16> {
  // Identity slice with dynamic dims (same shape, zero offset) folds away.
  // CHECK-NOT: loom.tile.slice
  // CHECK: return %[[SRC]] : !loom.tile<?x?xf16>
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%dim0, %dim1}
  return %subtile : !loom.tile<?x?xf16>
}

// -----

// CHECK-LABEL: @no_fold_identity_dynamic_offset
func.func @no_fold_identity_dynamic_offset(%src: !loom.tile<64x64xf16>, %offset: index) -> !loom.tile<64x64xf16> {
  // Can't fold - offset is dynamic, might not be zero.
  // CHECK: loom.tile.slice %{{.+}}[%{{.+}}, 0] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %subtile = loom.tile.slice %src[%offset, 0] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %subtile : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Compose Slice of Slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @compose_slice_of_slice_static
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<256x256xf16>
func.func @compose_slice_of_slice_static(%src: !loom.tile<256x256xf16>) -> !loom.tile<16x16xf16> {
  // slice(slice(x)[10,20])[5,8] -> slice(x)[15,28]
  // CHECK: loom.tile.slice %[[SRC]][15, 28] : !loom.tile<256x256xf16> -> !loom.tile<16x16xf16>
  %s1 = loom.tile.slice %src[10, 20] : !loom.tile<256x256xf16> -> !loom.tile<64x64xf16>
  %s2 = loom.tile.slice %s1[5, 8] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %s2 : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @compose_slice_of_slice_dynamic
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<256x256xf16>
// CHECK-SAME: %[[OFF0:[^:]+]]: index
// CHECK-SAME: %[[OFF1:[^:]+]]: index
func.func @compose_slice_of_slice_dynamic(%src: !loom.tile<256x256xf16>, %offset0: index, %offset1: index) -> !loom.tile<16x16xf16> {
  // slice(slice(x)[off0,off1])[5,8] -> slice(x)[off0+5,off1+8]
  // CHECK-DAG: %[[OFF0_PLUS_5:.*]] = arith.addi %[[OFF0]]
  // CHECK-DAG: %[[OFF1_PLUS_8:.*]] = arith.addi %[[OFF1]]
  // CHECK: loom.tile.slice %[[SRC]][%[[OFF0_PLUS_5]], %[[OFF1_PLUS_8]]] : !loom.tile<256x256xf16> -> !loom.tile<16x16xf16>
  %s1 = loom.tile.slice %src[%offset0, %offset1] : !loom.tile<256x256xf16> -> !loom.tile<64x64xf16>
  %s2 = loom.tile.slice %s1[5, 8] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %s2 : !loom.tile<16x16xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Poison Fold (UB)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_impossible_slice_to_poison
func.func @fold_impossible_slice_to_poison(%src: !loom.tile<32x32xf16>) -> !loom.tile<32x32xf16> {
  // Same shape but non-zero offset is UB - folds to poison.
  // Emits remark: ERR_LOOM_FOLD_0001 (verified separately with --verify-diagnostics)
  // CHECK: %[[POISON:.*]] = ub.poison : !loom.tile<32x32xf16>
  // CHECK: return %[[POISON]]
  %subtile = loom.tile.slice %src[1, 0] : !loom.tile<32x32xf16> -> !loom.tile<32x32xf16>
  return %subtile : !loom.tile<32x32xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Slice of Poison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_poison
func.func @fold_slice_of_poison() -> !loom.tile<16x16xf16> {
  // Slicing poison produces poison.
  // Emits remark: ERR_LOOM_FOLD_0001
  // CHECK: %[[POISON:.*]] = ub.poison : !loom.tile<16x16xf16>
  // CHECK: return %[[POISON]]
  %poison = ub.poison : !loom.tile<64x64xf16>
  %subtile = loom.tile.slice %poison[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Out-of-Bounds Slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_oob_slice_to_poison
func.func @fold_oob_slice_to_poison(%src: !loom.tile<64x64xf16>) -> !loom.tile<32x32xf16> {
  // offset 48 + size 32 = 80 > 64 - out of bounds
  // Emits remark: ERR_LOOM_FOLD_0001
  // CHECK: %[[POISON:.*]] = ub.poison : !loom.tile<32x32xf16>
  // CHECK: return %[[POISON]]
  %subtile = loom.tile.slice %src[48, 0] : !loom.tile<64x64xf16> -> !loom.tile<32x32xf16>
  return %subtile : !loom.tile<32x32xf16>
}

// -----

// CHECK-LABEL: @no_fold_in_bounds_slice
func.func @no_fold_in_bounds_slice(%src: !loom.tile<64x64xf16>) -> !loom.tile<32x32xf16> {
  // offset 32 + size 32 = 64 == 64 - exactly in bounds, should NOT fold
  // CHECK: loom.tile.slice %{{.+}}[32, 32] : !loom.tile<64x64xf16> -> !loom.tile<32x32xf16>
  %subtile = loom.tile.slice %src[32, 32] : !loom.tile<64x64xf16> -> !loom.tile<32x32xf16>
  return %subtile : !loom.tile<32x32xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Slice of Fill
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_fill_static
// CHECK-SAME: %[[CST:[^:]+]]: f32
func.func @fold_slice_of_fill_static(%cst: f32) -> !loom.tile<16x16xf32> {
  // slice(fill(val, target)) -> fill(val, smaller_alloca)
  // The slice op is replaced with a fill to a smaller alloca.
  // Note: the original alloca/fill may remain until DCE removes them.
  // CHECK: %[[SMALL_ALLOCA:.*]] = loom.tile.alloca : !loom.tile<16x16xf32>
  // CHECK: %[[SMALL_FILL:.*]] = loom.tile.fill %[[CST]], %[[SMALL_ALLOCA]] : f32 -> !loom.tile<16x16xf32>
  // CHECK: return %[[SMALL_FILL]]
  %alloca = loom.tile.alloca : !loom.tile<64x64xf32>
  %fill = loom.tile.fill %cst, %alloca : f32 -> !loom.tile<64x64xf32>
  %slice = loom.tile.slice %fill[8, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  return %slice : !loom.tile<16x16xf32>
}

// -----

// CHECK-LABEL: @fold_slice_of_fill_dynamic
// CHECK-SAME: %[[CST:[^:]+]]: f16
// CHECK-SAME: %[[DIM0:[^:]+]]: index
// CHECK-SAME: %[[DIM1:[^:]+]]: index
// CHECK-SAME: %[[SIZE0:[^:]+]]: index
// CHECK-SAME: %[[SIZE1:[^:]+]]: index
func.func @fold_slice_of_fill_dynamic(%cst: f16, %dim0: index, %dim1: index, %size0: index, %size1: index) -> !loom.tile<?x?xf16> {
  // slice(fill(val, target)) with dynamic shapes
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<?x?xf16>{%[[SIZE0]], %[[SIZE1]]}
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[CST]], %[[ALLOCA]] : f16 -> !loom.tile<?x?xf16>{%[[SIZE0]], %[[SIZE1]]}
  // CHECK: return %[[FILL]]
  %alloca = loom.tile.alloca : !loom.tile<?x?xf16>{%dim0, %dim1}
  %fill = loom.tile.fill %cst, %alloca : f16 -> !loom.tile<?x?xf16>{%dim0, %dim1}
  %slice = loom.tile.slice %fill[4, 4] : !loom.tile<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%size0, %size1}
  return %slice : !loom.tile<?x?xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_fill_multiple_uses
func.func @no_fold_slice_of_fill_multiple_uses(%cst: f32) -> (!loom.tile<64x64xf32>, !loom.tile<16x16xf32>) {
  // Don't fold when fill has multiple uses - would duplicate fill work.
  // CHECK: loom.tile.alloca : !loom.tile<64x64xf32>
  // CHECK: loom.tile.fill
  // CHECK: loom.tile.slice
  %alloca = loom.tile.alloca : !loom.tile<64x64xf32>
  %fill = loom.tile.fill %cst, %alloca : f32 -> !loom.tile<64x64xf32>
  %slice = loom.tile.slice %fill[8, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  return %fill, %slice : !loom.tile<64x64xf32>, !loom.tile<16x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Slice of Update (exact match)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_update_exact_match
// CHECK-SAME: %[[DATA:[^:]+]]: !loom.tile<16x16xf16>
func.func @fold_slice_of_update_exact_match(%data: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // slice(update(data, target[8,8])[8,8]) -> data
  // Slice extracts exactly what was updated.
  // CHECK-NOT: loom.tile.update
  // CHECK-NOT: loom.tile.slice
  // CHECK: return %[[DATA]] : !loom.tile<16x16xf16>
  %updated = loom.tile.update %data, %target[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.slice %updated[8, 8] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %result : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @fold_slice_of_update_exact_match_zero_offset
// CHECK-SAME: %[[DATA:[^:]+]]: !loom.tile<16x16xf16>
func.func @fold_slice_of_update_exact_match_zero_offset(%data: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // slice(update(data, target[0,0])[0,0]) -> data
  // CHECK-NOT: loom.tile.update
  // CHECK-NOT: loom.tile.slice
  // CHECK: return %[[DATA]] : !loom.tile<16x16xf16>
  %updated = loom.tile.update %data, %target[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.slice %updated[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %result : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_update_different_offset
func.func @no_fold_slice_of_update_different_offset(%data: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // Slice offset (0,0) != update offset (8,8) - can't fold directly.
  // CHECK: loom.tile.update
  // CHECK: loom.tile.slice
  %updated = loom.tile.update %data, %target[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.slice %updated[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %result : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_update_different_size
func.func @no_fold_slice_of_update_different_size(%data: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<8x8xf16> {
  // Slice size (8x8) != update size (16x16) - can't fold.
  // CHECK: loom.tile.update
  // CHECK: loom.tile.slice
  %updated = loom.tile.update %data, %target[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.slice %updated[8, 8] : !loom.tile<64x64xf16> -> !loom.tile<8x8xf16>
  return %result : !loom.tile<8x8xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Slice of Update (disjoint regions)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_update_disjoint
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_slice_of_update_disjoint(%data: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // Update at [0,0], slice at [32,32] - completely disjoint.
  // slice(update(data, target[0,0])[32,32]) -> slice(target[32,32])
  // CHECK-NOT: loom.tile.update
  // CHECK: loom.tile.slice %[[TARGET]][32, 32] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  %updated = loom.tile.update %data, %target[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.slice %updated[32, 32] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %result : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @fold_slice_of_update_disjoint_dim0
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_slice_of_update_disjoint_dim0(%data: !loom.tile<16x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<16x64xf16> {
  // Update at [0,0], slice at [32,0] - disjoint in dim 0 only.
  // CHECK-NOT: loom.tile.update
  // CHECK: loom.tile.slice %[[TARGET]][32, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x64xf16>
  %updated = loom.tile.update %data, %target[0, 0] : !loom.tile<16x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.slice %updated[32, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x64xf16>
  return %result : !loom.tile<16x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_update_overlapping
func.func @no_fold_slice_of_update_overlapping(%data: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // Update at [8,8] size 16x16, slice at [16,16] size 16x16.
  // Update region: [8,24) x [8,24), Slice region: [16,32) x [16,32)
  // These regions overlap - can't fold.
  // CHECK: loom.tile.update
  // CHECK: loom.tile.slice
  %updated = loom.tile.update %data, %target[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.slice %updated[16, 16] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %result : !loom.tile<16x16xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Slice of Broadcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_broadcast_right_align
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<4x8xf16>
func.func @fold_slice_of_broadcast_right_align(%src: !loom.tile<4x8xf16>) -> !loom.tile<2x4x8xf16> {
  // broadcast<right> adds leading dims. Slicing the leading (broadcast) dims
  // keeps source dims unchanged -> broadcast directly to smaller result.
  // CHECK-NOT: loom.tile.slice
  // CHECK: loom.tile.broadcast<right> %[[SRC]] : !loom.tile<4x8xf16> -> !loom.tile<2x4x8xf16>
  %bc = loom.tile.broadcast<right> %src : !loom.tile<4x8xf16> -> !loom.tile<8x4x8xf16>
  %result = loom.tile.slice %bc[2, 0, 0] : !loom.tile<8x4x8xf16> -> !loom.tile<2x4x8xf16>
  return %result : !loom.tile<2x4x8xf16>
}

// -----

// CHECK-LABEL: @fold_slice_of_broadcast_left_align
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<4x8xf16>
func.func @fold_slice_of_broadcast_left_align(%src: !loom.tile<4x8xf16>) -> !loom.tile<4x8x2xf16> {
  // broadcast<left> adds trailing dims. Slicing the trailing (broadcast) dims
  // keeps source dims unchanged -> broadcast directly to smaller result.
  // CHECK-NOT: loom.tile.slice
  // CHECK: loom.tile.broadcast<left> %[[SRC]] : !loom.tile<4x8xf16> -> !loom.tile<4x8x2xf16>
  %bc = loom.tile.broadcast<left> %src : !loom.tile<4x8xf16> -> !loom.tile<4x8x16xf16>
  %result = loom.tile.slice %bc[0, 0, 4] : !loom.tile<4x8x16xf16> -> !loom.tile<4x8x2xf16>
  return %result : !loom.tile<4x8x2xf16>
}

// -----

// CHECK-LABEL: @fold_slice_of_broadcast_multiple_broadcast_dims
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<8xf16>
func.func @fold_slice_of_broadcast_multiple_broadcast_dims(%src: !loom.tile<8xf16>) -> !loom.tile<2x4x8xf16> {
  // Source is 1D, broadcast adds two leading dims.
  // Slicing both leading dims -> broadcast to smaller result.
  // CHECK-NOT: loom.tile.slice
  // CHECK: loom.tile.broadcast<right> %[[SRC]] : !loom.tile<8xf16> -> !loom.tile<2x4x8xf16>
  %bc = loom.tile.broadcast<right> %src : !loom.tile<8xf16> -> !loom.tile<16x32x8xf16>
  %result = loom.tile.slice %bc[4, 8, 0] : !loom.tile<16x32x8xf16> -> !loom.tile<2x4x8xf16>
  return %result : !loom.tile<2x4x8xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_broadcast_slices_source_dim
func.func @no_fold_slice_of_broadcast_slices_source_dim(%src: !loom.tile<4x8xf16>) -> !loom.tile<8x2x8xf16> {
  // Slice affects source dim (dim 1: 4 -> 2) - can't fold.
  // CHECK: loom.tile.broadcast
  // CHECK: loom.tile.slice
  %bc = loom.tile.broadcast<right> %src : !loom.tile<4x8xf16> -> !loom.tile<8x4x8xf16>
  %result = loom.tile.slice %bc[0, 1, 0] : !loom.tile<8x4x8xf16> -> !loom.tile<8x2x8xf16>
  return %result : !loom.tile<8x2x8xf16>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_broadcast_nonzero_source_offset
func.func @no_fold_slice_of_broadcast_nonzero_source_offset(%src: !loom.tile<4x8xf16>) -> !loom.tile<8x2x4xf16> {
  // Source dim has non-zero offset (dim 1: offset 1, size 2 from 4) - can't fold.
  // CHECK: loom.tile.broadcast
  // CHECK: loom.tile.slice
  %bc = loom.tile.broadcast<right> %src : !loom.tile<4x8xf16> -> !loom.tile<16x4x8xf16>
  %result = loom.tile.slice %bc[0, 1, 2] : !loom.tile<16x4x8xf16> -> !loom.tile<8x2x4xf16>
  return %result : !loom.tile<8x2x4xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Cross-Boundary Composition (Tile.Slice of Tensor.Slice)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @compose_tile_slice_of_tensor_slice_static
// CHECK-SAME: %[[TENSOR:[^:]+]]: !loom.tensor<256x256xf16>
func.func @compose_tile_slice_of_tensor_slice_static(%tensor: !loom.tensor<256x256xf16>) -> !loom.tile<16x16xf16> {
  %t = loom.tensor.slice %tensor[8, 8] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  // CHECK-NOT: loom.tile.slice
  // CHECK: loom.tensor.slice %[[TENSOR]][12, 12] : !loom.tensor<256x256xf16> -> !loom.tile<16x16xf16>
  %result = loom.tile.slice %t[4, 4] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %result : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @compose_tile_slice_of_tensor_slice_dynamic
// CHECK-SAME: %[[TENSOR:[^:]+]]: !loom.tensor<256x256xf16>
func.func @compose_tile_slice_of_tensor_slice_dynamic(%tensor: !loom.tensor<256x256xf16>, %offset1: index, %offset2: index) -> !loom.tile<16x16xf16> {
  %t = loom.tensor.slice %tensor[%offset1, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  // CHECK: arith.addi
  // CHECK: loom.tensor.slice %[[TENSOR]][{{.+}}, 8] : !loom.tensor<256x256xf16> -> !loom.tile<16x16xf16>
  %result = loom.tile.slice %t[%offset2, 8] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %result : !loom.tile<16x16xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Slice of Elementwise
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_slice_of_elementwise_single_input
// CHECK-SAME: %[[INPUT:[^:]+]]: !loom.tile<64x64xf32>
func.func @fold_slice_of_elementwise_single_input(%input: !loom.tile<64x64xf32>) -> !loom.tile<16x16xf32> {
  // Slice of elementwise -> elementwise of slices.
  // CHECK: %[[RESULT:.+]] = loom.tile.slice %[[INPUT]][8, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  // CHECK: %[[EW:.+]] = loom.tile.elementwise(%[[A:.+]] = %[[RESULT]] : !loom.tile<16x16xf32>)
  // CHECK:   %[[NEG:.+]] = arith.negf %[[A]] : f32
  // CHECK:   loom.tile.yield %[[NEG]] : f32
  // CHECK: return %[[EW]]
  %ew = loom.tile.elementwise(%a = %input : !loom.tile<64x64xf32>) {
    %neg = arith.negf %a : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<64x64xf32>
  %result = loom.tile.slice %ew[8, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  return %result : !loom.tile<16x16xf32>
}

// -----

// CHECK-LABEL: @fold_slice_of_elementwise_binary
// CHECK-SAME: %[[A:[^:]+]]: !loom.tile<64x64xf32>
// CHECK-SAME: %[[B:[^:]+]]: !loom.tile<64x64xf32>
func.func @fold_slice_of_elementwise_binary(%a: !loom.tile<64x64xf32>, %b: !loom.tile<64x64xf32>) -> !loom.tile<16x16xf32> {
  // Binary elementwise: slice both inputs with same offsets.
  // CHECK-DAG: %[[A_SLICED:.+]] = loom.tile.slice %[[A]][4, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  // CHECK-DAG: %[[B_SLICED:.+]] = loom.tile.slice %[[B]][4, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  // CHECK: loom.tile.elementwise(%{{.+}} = %[[A_SLICED]] : !loom.tile<16x16xf32>, %{{.+}} = %[[B_SLICED]] : !loom.tile<16x16xf32>)
  // CHECK:   arith.addf
  %ew = loom.tile.elementwise(%x = %a : !loom.tile<64x64xf32>, %y = %b : !loom.tile<64x64xf32>) {
    %sum = arith.addf %x, %y : f32
    loom.tile.yield %sum : f32
  } -> !loom.tile<64x64xf32>
  %result = loom.tile.slice %ew[4, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  return %result : !loom.tile<16x16xf32>
}

// -----

// CHECK-LABEL: @fold_slice_of_elementwise_dynamic_offset
// CHECK-SAME: %[[INPUT:[^:]+]]: !loom.tile<64x64xf32>
// CHECK-SAME: %[[OFF:[^:]+]]: index
func.func @fold_slice_of_elementwise_dynamic_offset(%input: !loom.tile<64x64xf32>, %off: index) -> !loom.tile<16x16xf32> {
  // Dynamic offsets are propagated through.
  // CHECK: %[[RESULT:.+]] = loom.tile.slice %[[INPUT]][%[[OFF]], 0] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  // CHECK: loom.tile.elementwise(%{{.+}} = %[[RESULT]] : !loom.tile<16x16xf32>)
  %ew = loom.tile.elementwise(%a = %input : !loom.tile<64x64xf32>) {
    %neg = arith.negf %a : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<64x64xf32>
  %result = loom.tile.slice %ew[%off, 0] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  return %result : !loom.tile<16x16xf32>
}

// -----

// CHECK-LABEL: @no_fold_slice_of_elementwise_multiple_uses
func.func @no_fold_slice_of_elementwise_multiple_uses(%input: !loom.tile<64x64xf32>) -> (!loom.tile<64x64xf32>, !loom.tile<16x16xf32>) {
  // Don't fold when elementwise has multiple uses.
  // CHECK: %[[EW:.+]] = loom.tile.elementwise
  // CHECK: %[[SLICE:.+]] = loom.tile.slice %[[EW]]
  // CHECK: return %[[EW]], %[[SLICE]]
  %ew = loom.tile.elementwise(%a = %input : !loom.tile<64x64xf32>) {
    %neg = arith.negf %a : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<64x64xf32>
  %slice = loom.tile.slice %ew[8, 8] : !loom.tile<64x64xf32> -> !loom.tile<16x16xf32>
  return %ew, %slice : !loom.tile<64x64xf32>, !loom.tile<16x16xf32>
}
