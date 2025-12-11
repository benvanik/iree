// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Constant Source Offsets
//===----------------------------------------------------------------------===//

// Tests that constant source offset operands fold into static attributes.
// CHECK-LABEL: @fold_constant_source_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_constant_source_offsets(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: loom.tile.copy %[[SOURCE]][10, 20], %[[TARGET]][0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[%c10, %c20], %target[0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that partial constant source offsets fold correctly.
// CHECK-LABEL: @fold_partial_constant_source_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[OFFSET:[^:]+]]: index
func.func @fold_partial_constant_source_offsets(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>, %offset: index) -> !loom.tile<64x64xf16> {
  %c10 = arith.constant 10 : index
  // CHECK: loom.tile.copy %[[SOURCE]][10, %[[OFFSET]]], %[[TARGET]][0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[%c10, %offset], %target[0, 0], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Constant Target Offsets
//===----------------------------------------------------------------------===//

// Tests that constant target offset operands fold into static attributes.
// CHECK-LABEL: @fold_constant_target_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_constant_target_offsets(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][16, 32], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[%c16, %c32], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that partial constant target offsets fold correctly.
// CHECK-LABEL: @fold_partial_constant_target_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[OFFSET:[^:]+]]: index
func.func @fold_partial_constant_target_offsets(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>, %offset: index) -> !loom.tile<64x64xf16> {
  %c32 = arith.constant 32 : index
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][%[[OFFSET]], 32], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[%offset, %c32], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Fold Constant Sizes
//===----------------------------------------------------------------------===//

// Tests that constant size operands fold into static attributes.
// CHECK-LABEL: @fold_constant_sizes
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_constant_sizes(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][0, 0], [8, 16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[0, 0], [%c8, %c16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Simplify Copy to Slice
//===----------------------------------------------------------------------===//

// Tests that copy with target offsets [0,0] and sizes matching target simplifies to slice.
// The copy writes to the entire target, so target content is irrelevant.
// CHECK-LABEL: @simplify_copy_to_slice
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<128x128xf16>
func.func @simplify_copy_to_slice(%source: !loom.tile<128x128xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Source is large enough: 8 + 64 = 72 < 128 for both dimensions.
  // CHECK: loom.tile.slice %[[SOURCE]][8, 8] : !loom.tile<128x128xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[8, 8], %target[0, 0], [64, 64] : !loom.tile<128x128xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Simplify Copy to Update
//===----------------------------------------------------------------------===//

// Tests that copy with source offsets [0,0] and sizes matching source simplifies to update.
// CHECK-LABEL: @simplify_copy_to_update
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @simplify_copy_to_update(%source: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // CHECK: loom.tile.update %[[SOURCE]], %[[TARGET]][16, 32] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[16, 32], [16, 16] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Folding: Self-Copy Fold
//===----------------------------------------------------------------------===//

// Tests that copy from tile to itself at the same offset folds to identity.
// CHECK-LABEL: @fold_self_copy_same_offsets
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x64xf16>
func.func @fold_self_copy_same_offsets(%tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Copy from tile to itself at the same offset is a no-op.
  // CHECK-NOT: loom.tile.copy
  // CHECK: return %[[TILE]] : !loom.tile<64x64xf16>
  %result = loom.tile.copy %tile[8, 8], %tile[8, 8], [16, 16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that copy from tile to itself at different offsets is NOT folded.
// CHECK-LABEL: @no_fold_self_copy_different_offsets
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<64x64xf16>
func.func @no_fold_self_copy_different_offsets(%tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Copy from tile to itself at different offsets is NOT a no-op.
  // (This could be used to shift data within a tile.)
  // CHECK: loom.tile.copy %[[TILE]][0, 0], %[[TILE]][16, 16], [16, 16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %tile[0, 0], %tile[16, 16], [16, 16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

//===----------------------------------------------------------------------===//
// No Folding Cases
//===----------------------------------------------------------------------===//

// Tests that already-static copy is unchanged.
// CHECK-LABEL: @no_fold_already_static
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @no_fold_already_static(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Already fully static - nothing to fold.
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][16, 32], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[16, 32], [8, 8] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that copy with dynamic source offsets cannot simplify to update.
// CHECK-LABEL: @no_fold_simplify_dynamic_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[OFFSET:[^:]+]]: index
func.func @no_fold_simplify_dynamic_offsets(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>, %offset: index) -> !loom.tile<64x64xf16> {
  // Can't simplify to update - source offsets are dynamic.
  // CHECK: loom.tile.copy %[[SOURCE]][%[[OFFSET]], 0], %[[TARGET]][16, 32], [16, 16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[%offset, 0], %target[16, 32], [16, 16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}
