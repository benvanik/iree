// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_alloca_static
func.func @tile_alloca_static() -> !loom.tile<4x4xf32> {
  // Allocate a static-shaped tile with undefined contents.
  // The tile must be initialized before reading.
  //
  // CHECK: loom.tile.alloca : !loom.tile<4x4xf32>
  %0 = loom.tile.alloca : !loom.tile<4x4xf32>
  return %0 : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @tile_alloca_dynamic
// CHECK-SAME: %[[D0:.+]]: index, %[[D1:.+]]: index
func.func @tile_alloca_dynamic(%dim0: index, %dim1: index) -> !loom.tile<?x?xf32> {
  // Allocate a dynamic-shaped tile. Dimensions are provided as operands.
  //
  // CHECK: loom.tile.alloca : !loom.tile<?x?xf32>{%[[D0]], %[[D1]]}
  %0 = loom.tile.alloca : !loom.tile<?x?xf32>{%dim0, %dim1}
  return %0 : !loom.tile<?x?xf32>
}

// -----

// CHECK-LABEL: @tile_alloca_then_fill
// CHECK-SAME: %[[D0:.+]]: index, %[[D1:.+]]: index
func.func @tile_alloca_then_fill(%dim0: index, %dim1: index) -> !loom.tile<?x?xi32> {
  // Common pattern: allocate then fill with a value.
  //
  // CHECK: %[[CST:.+]] = arith.constant 0 : i32
  // CHECK: %[[UNINIT:.+]] = loom.tile.alloca : !loom.tile<?x?xi32>{%[[D0]], %[[D1]]}
  // CHECK: %[[ZEROS:.+]] = loom.tile.fill %[[CST]], %[[UNINIT]] : i32 -> !loom.tile<?x?xi32>{%[[D0]], %[[D1]]}
  // CHECK: return %[[ZEROS]]
  %cst = arith.constant 0 : i32
  %uninit = loom.tile.alloca : !loom.tile<?x?xi32>{%dim0, %dim1}
  %zeros = loom.tile.fill %cst, %uninit : i32 -> !loom.tile<?x?xi32>{%dim0, %dim1}
  return %zeros : !loom.tile<?x?xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tile_alloca_integer
func.func @tile_alloca_integer() -> !loom.tile<8xi32> {
  // Integer element type allocation.
  //
  // CHECK: loom.tile.alloca : !loom.tile<8xi32>
  %0 = loom.tile.alloca : !loom.tile<8xi32>
  return %0 : !loom.tile<8xi32>
}

// -----

// CHECK-LABEL: @tile_alloca_1d
// CHECK-SAME: %[[D:.+]]: index
func.func @tile_alloca_1d(%d: index) -> !loom.tile<?xf32> {
  // 1D dynamic allocation.
  //
  // CHECK: loom.tile.alloca : !loom.tile<?xf32>{%[[D]]}
  %0 = loom.tile.alloca : !loom.tile<?xf32>{%d}
  return %0 : !loom.tile<?xf32>
}

// -----

// CHECK-LABEL: @tile_alloca_3d
func.func @tile_alloca_3d() -> !loom.tile<2x4x8xf32> {
  // 3D static allocation.
  //
  // CHECK: loom.tile.alloca : !loom.tile<2x4x8xf32>
  %0 = loom.tile.alloca : !loom.tile<2x4x8xf32>
  return %0 : !loom.tile<2x4x8xf32>
}

// -----

// CHECK-LABEL: @tile_alloca_bf16
func.func @tile_alloca_bf16() -> !loom.tile<4x4xbf16> {
  // BFloat16 element type allocation.
  //
  // CHECK: loom.tile.alloca : !loom.tile<4x4xbf16>
  %0 = loom.tile.alloca : !loom.tile<4x4xbf16>
  return %0 : !loom.tile<4x4xbf16>
}

// -----

// CHECK-LABEL: @tile_alloca_mixed_dims
// CHECK-SAME: %[[D1:.+]]: index
func.func @tile_alloca_mixed_dims(%dim1: index) -> !loom.tile<4x?x8xf32> {
  // Mixed static and dynamic dimensions.
  //
  // CHECK: loom.tile.alloca : !loom.tile<4x?x8xf32>{%[[D1]]}
  %0 = loom.tile.alloca : !loom.tile<4x?x8xf32>{%dim1}
  return %0 : !loom.tile<4x?x8xf32>
}
