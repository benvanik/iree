// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization: Broadcast of Poison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_broadcast_of_poison
func.func @fold_broadcast_of_poison() -> !loom.tile<4x4xf32> {
  // Broadcasting poison produces poison.
  // Emits remark: ERR_LOOM_FOLD_0001
  // CHECK: %[[POISON:.*]] = ub.poison : !loom.tile<4x4xf32>
  // CHECK: return %[[POISON]]
  %poison = ub.poison : !loom.tile<1x1xf32>
  %broadcast = loom.tile.broadcast<right> %poison : !loom.tile<1x1xf32> -> !loom.tile<4x4xf32>
  return %broadcast : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Broadcast of Fill
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_broadcast_of_fill_static
// CHECK-SAME: %[[VALUE:[^:]+]]: f32
func.func @fold_broadcast_of_fill_static(%value: f32) -> !loom.tile<16x16xf32> {
  // broadcast(fill(val, target)) -> fill(val, larger_alloca)
  // The broadcast op is replaced with a fill to a larger alloca.
  // CHECK: %[[SMALL_ALLOCA:.*]] = loom.tile.alloca : !loom.tile<16x16xf32>
  // CHECK: %[[LARGE_FILL:.*]] = loom.tile.fill %[[VALUE]], %[[SMALL_ALLOCA]] : f32 -> !loom.tile<16x16xf32>
  // CHECK: return %[[LARGE_FILL]]
  %small_alloca = loom.tile.alloca : !loom.tile<1x1xf32>
  %fill = loom.tile.fill %value, %small_alloca : f32 -> !loom.tile<1x1xf32>
  %broadcast = loom.tile.broadcast<right> %fill : !loom.tile<1x1xf32> -> !loom.tile<16x16xf32>
  return %broadcast : !loom.tile<16x16xf32>
}

// -----

// CHECK-LABEL: @fold_broadcast_of_fill_add_rank
// CHECK-SAME: %[[VALUE:[^:]+]]: f16
func.func @fold_broadcast_of_fill_add_rank(%value: f16) -> !loom.tile<8x4x4xf16> {
  // broadcast adds a dimension to a filled tile.
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<8x4x4xf16>
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[VALUE]], %[[ALLOCA]] : f16 -> !loom.tile<8x4x4xf16>
  // CHECK: return %[[FILL]]
  %alloca = loom.tile.alloca : !loom.tile<4x4xf16>
  %fill = loom.tile.fill %value, %alloca : f16 -> !loom.tile<4x4xf16>
  %broadcast = loom.tile.broadcast<right> %fill : !loom.tile<4x4xf16> -> !loom.tile<8x4x4xf16>
  return %broadcast : !loom.tile<8x4x4xf16>
}

// -----

// CHECK-LABEL: @no_fold_broadcast_of_fill_multiple_uses
func.func @no_fold_broadcast_of_fill_multiple_uses(%value: f32) -> (!loom.tile<1x1xf32>, !loom.tile<16x16xf32>) {
  // Don't fold when fill has multiple uses - would duplicate fill work.
  // CHECK: loom.tile.alloca : !loom.tile<1x1xf32>
  // CHECK: loom.tile.fill
  // CHECK: loom.tile.broadcast
  %alloca = loom.tile.alloca : !loom.tile<1x1xf32>
  %fill = loom.tile.fill %value, %alloca : f32 -> !loom.tile<1x1xf32>
  %broadcast = loom.tile.broadcast<right> %fill : !loom.tile<1x1xf32> -> !loom.tile<16x16xf32>
  return %fill, %broadcast : !loom.tile<1x1xf32>, !loom.tile<16x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Broadcast of Splat Constant
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_broadcast_of_splat_constant
func.func @fold_broadcast_of_splat_constant() -> !loom.tile<8x4x4xi32> {
  // broadcast(constant<splat>) -> constant<splat> with larger shape.
  // A splat constant has the same value everywhere, so broadcasting it
  // just creates a larger splat constant with the same element value.
  // CHECK: %[[C:.+]] = loom.tile.constant #loom.dense<1> : !loom.tile<8x4x4xi32>
  // CHECK: return %[[C]]
  %c = loom.tile.constant #loom.dense<1> : !loom.tile<4x4xi32>
  %bc = loom.tile.broadcast<right> %c : !loom.tile<4x4xi32> -> !loom.tile<8x4x4xi32>
  return %bc : !loom.tile<8x4x4xi32>
}

// -----

// CHECK-LABEL: @fold_broadcast_of_splat_constant_integer
func.func @fold_broadcast_of_splat_constant_integer() -> !loom.tile<16x8xi32> {
  // Integer splat constant also folds.
  // CHECK: %[[C:.+]] = loom.tile.constant #loom.dense<42> : !loom.tile<16x8xi32>
  // CHECK: return %[[C]]
  %c = loom.tile.constant #loom.dense<42> : !loom.tile<8xi32>
  %bc = loom.tile.broadcast<right> %c : !loom.tile<8xi32> -> !loom.tile<16x8xi32>
  return %bc : !loom.tile<16x8xi32>
}

// -----

// CHECK-LABEL: @no_fold_broadcast_of_non_splat_constant
func.func @no_fold_broadcast_of_non_splat_constant() -> !loom.tile<4x2xi32> {
  // Non-splat constant - cannot fold since broadcast would need to
  // replicate different values.
  // CHECK: loom.tile.constant #loom.dense<[1, 2]>
  // CHECK: loom.tile.broadcast
  %c = loom.tile.constant #loom.dense<[1, 2]> : !loom.tile<2xi32>
  %bc = loom.tile.broadcast<right> %c : !loom.tile<2xi32> -> !loom.tile<4x2xi32>
  return %bc : !loom.tile<4x2xi32>
}
