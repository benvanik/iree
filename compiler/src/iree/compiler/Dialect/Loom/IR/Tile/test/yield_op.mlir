// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @yield_basic
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<4x4xf32>
func.func @yield_basic(%input: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Basic yield usage - terminates elementwise region.
  //
  // CHECK: %[[RESULT:.+]] = loom.tile.elementwise(%[[ELEMENT:.+]] = %[[INPUT]] : !loom.tile<4x4xf32>)
  // CHECK:   %[[NEG:.+]] = arith.negf %[[ELEMENT]] : f32
  // CHECK:   loom.tile.yield %[[NEG]] : f32
  // CHECK: }
  // CHECK: return %[[RESULT]] : !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xf32>) {
    %neg = arith.negf %element : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @yield_integer
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<8xi32>
func.func @yield_integer(%input: !loom.tile<8xi32>) -> !loom.tile<8xi32> {
  // Yield with integer type.
  //
  // CHECK: %[[RESULT:.+]] = loom.tile.elementwise(%[[ELEMENT:.+]] = %[[INPUT]] : !loom.tile<8xi32>)
  // CHECK:   %[[SQ:.+]] = arith.muli %[[ELEMENT]], %[[ELEMENT]] : i32
  // CHECK:   loom.tile.yield %[[SQ]] : i32
  // CHECK: }
  // CHECK: return %[[RESULT]] : !loom.tile<8xi32>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<8xi32>) {
    %sq = arith.muli %element, %element : i32
    loom.tile.yield %sq : i32
  } -> !loom.tile<8xi32>
  return %result : !loom.tile<8xi32>
}

// -----

// CHECK-LABEL: @yield_bf16
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<4x4xbf16>
func.func @yield_bf16(%input: !loom.tile<4x4xbf16>) -> !loom.tile<4x4xbf16> {
  // Yield with bf16 type.
  //
  // CHECK: %[[RESULT:.+]] = loom.tile.elementwise(%[[ELEMENT:.+]] = %[[INPUT]] : !loom.tile<4x4xbf16>)
  // CHECK:   %[[NEG:.+]] = arith.negf %[[ELEMENT]] : bf16
  // CHECK:   loom.tile.yield %[[NEG]] : bf16
  // CHECK: }
  // CHECK: return %[[RESULT]] : !loom.tile<4x4xbf16>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xbf16>) {
    %neg = arith.negf %element : bf16
    loom.tile.yield %neg : bf16
  } -> !loom.tile<4x4xbf16>
  return %result : !loom.tile<4x4xbf16>
}

// -----

// CHECK-LABEL: @yield_after_computation
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<16xf32>, %[[SCALE:.+]]: f32
func.func @yield_after_computation(%input: !loom.tile<16xf32>, %scale: f32) -> !loom.tile<16xf32> {
  // Yield after multiple computation steps.
  //
  // CHECK: %[[RESULT:.+]] = loom.tile.elementwise(%[[ELEMENT:.+]] = %[[INPUT]] : !loom.tile<16xf32>)
  // CHECK:   %[[ABS:.+]] = math.absf %[[ELEMENT]] : f32
  // CHECK:   %[[SCALED:.+]] = arith.mulf %[[ABS]], %[[SCALE]] : f32
  // CHECK:   %[[CLAMP:.+]] = arith.maximumf %[[SCALED]]
  // CHECK:   loom.tile.yield %[[CLAMP]] : f32
  // CHECK: }
  // CHECK: return %[[RESULT]] : !loom.tile<16xf32>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<16xf32>) {
    %abs = math.absf %element : f32
    %scaled = arith.mulf %abs, %scale : f32
    %cst = arith.constant 0.0 : f32
    %clamp = arith.maximumf %scaled, %cst : f32
    loom.tile.yield %clamp : f32
  } -> !loom.tile<16xf32>
  return %result : !loom.tile<16xf32>
}
