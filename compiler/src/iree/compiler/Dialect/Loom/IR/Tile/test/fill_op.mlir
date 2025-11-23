// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_fill_static
// CHECK-SAME: %[[VALUE:.+]]: f32, %[[TARGET:.+]]: !loom.tile
func.func @tile_fill_static(%value: f32, %target: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Fill a static-shaped tile with a scalar value.
  // The result is tied to the target operand (in-place semantics).
  //
  // CHECK: loom.tile.fill %[[VALUE]], %[[TARGET]] : f32 -> !loom.tile<4x4xf32>
  %0 = loom.tile.fill %value, %target : f32 -> !loom.tile<4x4xf32>
  return %0 : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @tile_fill_dynamic
// CHECK-SAME: %[[VALUE:.+]]: f32, %[[TARGET:.+]]: !loom.tile<?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index
func.func @tile_fill_dynamic(%value: f32, %target: !loom.tile<?x?xf32>, %dim0: index, %dim1: index) -> !loom.tile<?x?xf32> {
  // Fill a dynamic-shaped tile. Dynamic dimensions are specified in braces.
  //
  // CHECK: loom.tile.fill %[[VALUE]], %[[TARGET]] : f32 -> !loom.tile<?x?xf32>{%[[D0]], %[[D1]]}
  %0 = loom.tile.fill %value, %target : f32 -> !loom.tile<?x?xf32>{%dim0, %dim1}
  return %0 : !loom.tile<?x?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_fill_integer
// CHECK-SAME: %[[VALUE:.+]]: i32, %[[TARGET:.+]]: !loom.tile
func.func @tile_fill_integer(%value: i32, %target: !loom.tile<8xi32>) -> !loom.tile<8xi32> {
  // Integer element type.
  //
  // CHECK: loom.tile.fill %[[VALUE]], %[[TARGET]] : i32 -> !loom.tile<8xi32>
  %0 = loom.tile.fill %value, %target : i32 -> !loom.tile<8xi32>
  return %0 : !loom.tile<8xi32>
}

// -----

// CHECK-LABEL: @tile_fill_1d
// CHECK-SAME: %[[VALUE:.+]]: f32, %[[TARGET:.+]]: !loom.tile
func.func @tile_fill_1d(%value: f32, %target: !loom.tile<16xf32>) -> !loom.tile<16xf32> {
  // 1D vector fill.
  //
  // CHECK: loom.tile.fill %[[VALUE]], %[[TARGET]] : f32 -> !loom.tile<16xf32>
  %0 = loom.tile.fill %value, %target : f32 -> !loom.tile<16xf32>
  return %0 : !loom.tile<16xf32>
}

// -----

// CHECK-LABEL: @tile_fill_3d
// CHECK-SAME: %[[VALUE:.+]]: f32, %[[TARGET:.+]]: !loom.tile
func.func @tile_fill_3d(%value: f32, %target: !loom.tile<2x4x8xf32>) -> !loom.tile<2x4x8xf32> {
  // 3D tensor fill.
  //
  // CHECK: loom.tile.fill %[[VALUE]], %[[TARGET]] : f32 -> !loom.tile<2x4x8xf32>
  %0 = loom.tile.fill %value, %target : f32 -> !loom.tile<2x4x8xf32>
  return %0 : !loom.tile<2x4x8xf32>
}

// -----

// CHECK-LABEL: @tile_fill_bf16
// CHECK-SAME: %[[VALUE:.+]]: bf16, %[[TARGET:.+]]: !loom.tile
func.func @tile_fill_bf16(%value: bf16, %target: !loom.tile<4x4xbf16>) -> !loom.tile<4x4xbf16> {
  // BFloat16 element type.
  //
  // CHECK: loom.tile.fill %[[VALUE]], %[[TARGET]] : bf16 -> !loom.tile<4x4xbf16>
  %0 = loom.tile.fill %value, %target : bf16 -> !loom.tile<4x4xbf16>
  return %0 : !loom.tile<4x4xbf16>
}

// -----

// CHECK-LABEL: @tile_fill_i64
// CHECK-SAME: %[[VALUE:.+]]: i64, %[[TARGET:.+]]: !loom.tile
func.func @tile_fill_i64(%value: i64, %target: !loom.tile<8x8xi64>) -> !loom.tile<8x8xi64> {
  // 64-bit integer fill.
  //
  // CHECK: loom.tile.fill %[[VALUE]], %[[TARGET]] : i64 -> !loom.tile<8x8xi64>
  %0 = loom.tile.fill %value, %target : i64 -> !loom.tile<8x8xi64>
  return %0 : !loom.tile<8x8xi64>
}
