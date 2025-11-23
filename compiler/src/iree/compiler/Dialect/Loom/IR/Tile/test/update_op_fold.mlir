// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Constant Offsets
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_all_constant_offsets
func.func @fold_all_constant_offsets(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: loom.tile.update %{{.+}}, %{{.+}}[10, 20] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %updated = loom.tile.update %subtile, %tile[%c10, %c20] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @fold_partial_constant_offsets
func.func @fold_partial_constant_offsets(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>, %offset: index) -> !loom.tile<64x64xf16> {
  %c10 = arith.constant 10 : index
  // CHECK: loom.tile.update {{.+}}[10, {{.+}}] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %updated = loom.tile.update %subtile, %tile[%c10, %offset] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_already_static
func.func @no_fold_already_static(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // CHECK: loom.tile.update %{{.+}}, %{{.+}}[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %updated = loom.tile.update %subtile, %tile[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @fold_with_dynamic_dims
func.func @fold_with_dynamic_dims(%subtile: !loom.tile<?x?xf16>, %tile: !loom.tile<?x?xf16>, %dim0: index, %dim1: index, %ssz0: index, %ssz1: index) -> !loom.tile<?x?xf16> {
  %c5 = arith.constant 5 : index
  %c15 = arith.constant 15 : index
  // CHECK: loom.tile.update {{.+}}[5, 15] : !loom.tile<?x?xf16>{{.+}}
  %updated = loom.tile.update %subtile, %tile[%c5, %c15] : !loom.tile<?x?xf16>{%ssz0, %ssz1} -> !loom.tile<?x?xf16>{%dim0, %dim1}
  return %updated : !loom.tile<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Folding: Identity Update
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_identity_update
// CHECK-SAME: %[[UPDATE:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_identity_update(%update: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Identity update (same shape, zero offset) folds to just the update value.
  // CHECK-NOT: loom.tile.update
  // CHECK: return %[[UPDATE]] : !loom.tile<64x64xf16>
  %result = loom.tile.update %update, %target[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @fold_identity_update_dynamic
// CHECK-SAME: %[[UPDATE:[^:]+]]: !loom.tile<?x?xf16>
func.func @fold_identity_update_dynamic(%update: !loom.tile<?x?xf16>, %target: !loom.tile<?x?xf16>, %dim0: index, %dim1: index) -> !loom.tile<?x?xf16> {
  // Identity update with dynamic dims (same shape, zero offset) folds away.
  // CHECK-NOT: loom.tile.update
  // CHECK: return %[[UPDATE]] : !loom.tile<?x?xf16>
  %result = loom.tile.update %update, %target[0, 0] : !loom.tile<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%dim0, %dim1}
  return %result : !loom.tile<?x?xf16>
}

// -----

// CHECK-LABEL: @no_fold_identity_dynamic_offset
func.func @no_fold_identity_dynamic_offset(%update: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>, %offset: index) -> !loom.tile<64x64xf16> {
  // Can't fold - offset is dynamic, might not be zero.
  // CHECK: loom.tile.update {{.+}}[{{.+}}, 0] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.update %update, %target[%offset, 0] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Poison Fold (UB)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_impossible_update_to_poison
func.func @fold_impossible_update_to_poison(%update: !loom.tile<32x32xf16>, %target: !loom.tile<32x32xf16>) -> !loom.tile<32x32xf16> {
  // Same shape but non-zero offset is UB - folds to poison.
  // Emits remark: ERR_LOOM_FOLD_0001 (verified separately with --verify-diagnostics)
  // CHECK: %[[POISON:.*]] = ub.poison : !loom.tile<32x32xf16>
  // CHECK: return %[[POISON]]
  %result = loom.tile.update %update, %target[1, 0] : !loom.tile<32x32xf16> -> !loom.tile<32x32xf16>
  return %result : !loom.tile<32x32xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Update of Poison Target
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_update_of_poison_target
func.func @fold_update_of_poison_target(%update: !loom.tile<16x16xf16>) -> !loom.tile<64x64xf16> {
  // Updating a poison target produces poison.
  // Emits remark: ERR_LOOM_FOLD_0001
  // CHECK: %[[POISON:.*]] = ub.poison : !loom.tile<64x64xf16>
  // CHECK: return %[[POISON]]
  %poison = ub.poison : !loom.tile<64x64xf16>
  %result = loom.tile.update %update, %poison[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Out-of-Bounds Update
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_oob_update_to_poison
func.func @fold_oob_update_to_poison(%update: !loom.tile<32x32xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // offset 48 + size 32 = 80 > 64 - out of bounds
  // Emits remark: ERR_LOOM_FOLD_0001
  // CHECK: %[[POISON:.*]] = ub.poison : !loom.tile<64x64xf16>
  // CHECK: return %[[POISON]]
  %result = loom.tile.update %update, %target[48, 0] : !loom.tile<32x32xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_in_bounds_update
func.func @no_fold_in_bounds_update(%update: !loom.tile<32x32xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // offset 32 + size 32 = 64 == 64 - exactly in bounds, should NOT fold
  // CHECK: loom.tile.update %{{.+}}, %{{.+}}[32, 32] : !loom.tile<32x32xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.update %update, %target[32, 32] : !loom.tile<32x32xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Idempotent Fill Update
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_idempotent_fill_update
// CHECK-SAME: %[[VALUE:[^:]+]]: f32
func.func @fold_idempotent_fill_update(%value: f32) -> !loom.tile<64x64xf32> {
  // update(fill(v), fill(v)) -> fill(v) since filling with same value is no-op.
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<64x64xf32>
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[VALUE]], %[[ALLOCA]] : f32 -> !loom.tile<64x64xf32>
  // CHECK-NOT: loom.tile.update
  // CHECK: return %[[FILL]]
  %alloca_target = loom.tile.alloca : !loom.tile<64x64xf32>
  %target_fill = loom.tile.fill %value, %alloca_target : f32 -> !loom.tile<64x64xf32>
  %alloca_update = loom.tile.alloca : !loom.tile<16x16xf32>
  %update_fill = loom.tile.fill %value, %alloca_update : f32 -> !loom.tile<16x16xf32>
  %result = loom.tile.update %update_fill, %target_fill[8, 8] : !loom.tile<16x16xf32> -> !loom.tile<64x64xf32>
  return %result : !loom.tile<64x64xf32>
}

// -----

// CHECK-LABEL: @no_fold_different_fill_values
// CHECK-SAME: %[[VALUE1:[^:]+]]: f32
// CHECK-SAME: %[[VALUE2:[^:]+]]: f32
func.func @no_fold_different_fill_values(%value1: f32, %value2: f32) -> !loom.tile<64x64xf32> {
  // Don't fold when fill values are different.
  // CHECK: loom.tile.fill %[[VALUE1]]
  // CHECK: loom.tile.fill %[[VALUE2]]
  // CHECK: loom.tile.update
  %alloca_target = loom.tile.alloca : !loom.tile<64x64xf32>
  %target_fill = loom.tile.fill %value1, %alloca_target : f32 -> !loom.tile<64x64xf32>
  %alloca_update = loom.tile.alloca : !loom.tile<16x16xf32>
  %update_fill = loom.tile.fill %value2, %alloca_update : f32 -> !loom.tile<16x16xf32>
  %result = loom.tile.update %update_fill, %target_fill[8, 8] : !loom.tile<16x16xf32> -> !loom.tile<64x64xf32>
  return %result : !loom.tile<64x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Update Over Update (overwrite elimination)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_update_over_update
// CHECK-SAME: %[[DATA1:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[DATA2:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_update_over_update(%data1: !loom.tile<16x16xf16>, %data2: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // update(data2, update(data1, target[8,8])[8,8]) -> update(data2, target[8,8])
  // Second update completely overwrites first at same location.
  // CHECK-NOT: loom.tile.update %[[DATA1]]
  // CHECK: loom.tile.update %[[DATA2]], %[[TARGET]][8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %u1 = loom.tile.update %data1, %target[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %u2 = loom.tile.update %data2, %u1[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %u2 : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @fold_update_over_update_zero_offset
// CHECK-SAME: %[[DATA1:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[DATA2:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_update_over_update_zero_offset(%data1: !loom.tile<16x16xf16>, %data2: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // update(data2, update(data1, target[0,0])[0,0]) -> update(data2, target[0,0])
  // CHECK-NOT: loom.tile.update %[[DATA1]]
  // CHECK: loom.tile.update %[[DATA2]], %[[TARGET]][0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %u1 = loom.tile.update %data1, %target[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %u2 = loom.tile.update %data2, %u1[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %u2 : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_update_over_update_different_offset
// CHECK-SAME: %[[DATA1:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[DATA2:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @no_fold_update_over_update_different_offset(%data1: !loom.tile<16x16xf16>, %data2: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Different offsets - can't fold.
  // CHECK: loom.tile.update %[[DATA1]], %[[TARGET]][0, 0]
  // CHECK: loom.tile.update %[[DATA2]], %{{.+}}[8, 8]
  %u1 = loom.tile.update %data1, %target[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %u2 = loom.tile.update %data2, %u1[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %u2 : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_update_over_update_different_size
// CHECK-SAME: %[[DATA1:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[DATA2:[^:]+]]: !loom.tile<8x8xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @no_fold_update_over_update_different_size(%data1: !loom.tile<16x16xf16>, %data2: !loom.tile<8x8xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Different sizes - can't fold (second update is smaller, doesn't fully overwrite).
  // CHECK: loom.tile.update %[[DATA1]], %[[TARGET]][8, 8]
  // CHECK: loom.tile.update %[[DATA2]], %{{.+}}[8, 8]
  %u1 = loom.tile.update %data1, %target[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %u2 = loom.tile.update %data2, %u1[8, 8] : !loom.tile<8x8xf16> -> !loom.tile<64x64xf16>
  return %u2 : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @no_fold_update_over_update_multiple_uses
// CHECK-SAME: %[[DATA1:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[DATA2:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @no_fold_update_over_update_multiple_uses(%data1: !loom.tile<16x16xf16>, %data2: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> (!loom.tile<64x64xf16>, !loom.tile<64x64xf16>) {
  // Inner update has multiple uses - can't skip it.
  // CHECK: loom.tile.update %[[DATA1]], %[[TARGET]][8, 8]
  // CHECK: loom.tile.update %[[DATA2]], %{{.+}}[8, 8]
  %u1 = loom.tile.update %data1, %target[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %u2 = loom.tile.update %data2, %u1[8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %u1, %u2 : !loom.tile<64x64xf16>, !loom.tile<64x64xf16>
}
