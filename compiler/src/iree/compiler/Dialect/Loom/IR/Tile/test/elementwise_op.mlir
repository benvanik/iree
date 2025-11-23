// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @elementwise_unary
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<4x4xf32>
func.func @elementwise_unary(%input: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Unary operation - negation of all elements.
  //
  // CHECK: loom.tile.elementwise(%[[element:.+]] = %[[INPUT]] : !loom.tile<4x4xf32>)
  // CHECK:   %[[NEG:.+]] = arith.negf %[[element]] : f32
  // CHECK:   loom.tile.yield %[[NEG]] : f32
  // CHECK: } -> !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xf32>) {
    %neg = arith.negf %element : f32
    loom.tile.yield %neg : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @elementwise_binary
// CHECK-SAME: %[[A:.+]]: !loom.tile<8xf16>, %[[B:.+]]: !loom.tile<8xf16>
func.func @elementwise_binary(%a: !loom.tile<8xf16>, %b: !loom.tile<8xf16>) -> !loom.tile<8xf16> {
  // Binary operation - elementwise addition.
  //
  // CHECK: loom.tile.elementwise(%[[EA:.+]] = %[[A]] : !loom.tile<8xf16>, %[[EB:.+]] = %[[B]] : !loom.tile<8xf16>)
  // CHECK:   %[[SUM:.+]] = arith.addf %[[EA]], %[[EB]] : f16
  // CHECK:   loom.tile.yield %[[SUM]] : f16
  // CHECK: } -> !loom.tile<8xf16>
  %result = loom.tile.elementwise(%ea = %a : !loom.tile<8xf16>, %eb = %b : !loom.tile<8xf16>) {
    %sum = arith.addf %ea, %eb : f16
    loom.tile.yield %sum : f16
  } -> !loom.tile<8xf16>
  return %result : !loom.tile<8xf16>
}

// -----

// CHECK-LABEL: @elementwise_with_capture
// CHECK-SAME: %[[A:.+]]: !loom.tile<8xf16>, %[[B:.+]]: !loom.tile<8xf16>, %[[SCALE:.+]]: f16
func.func @elementwise_with_capture(%a: !loom.tile<8xf16>, %b: !loom.tile<8xf16>, %scale: f16) -> !loom.tile<8xf16> {
  // Binary with implicit capture - %scale is captured from outside.
  //
  // CHECK: loom.tile.elementwise(%[[EA:.+]] = %[[A]] : !loom.tile<8xf16>, %[[EB:.+]] = %[[B]] : !loom.tile<8xf16>)
  // CHECK:   %[[SUM:.+]] = arith.addf %[[EA]], %[[EB]] : f16
  // CHECK:   %[[SCALED:.+]] = arith.mulf %[[SUM]], %[[SCALE]] : f16
  // CHECK:   loom.tile.yield %[[SCALED]] : f16
  // CHECK: } -> !loom.tile<8xf16>
  %result = loom.tile.elementwise(%ea = %a : !loom.tile<8xf16>, %eb = %b : !loom.tile<8xf16>) {
    %sum = arith.addf %ea, %eb : f16
    %scaled = arith.mulf %sum, %scale : f16
    loom.tile.yield %scaled : f16
  } -> !loom.tile<8xf16>
  return %result : !loom.tile<8xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @elementwise_dynamic
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index
func.func @elementwise_dynamic(%input: !loom.tile<?x?xf32>, %dim0: index, %dim1: index) -> !loom.tile<?x?xf32> {
  // Dynamic dimensions - dimensions passed via braces.
  //
  // CHECK: loom.tile.elementwise(%[[element:.+]] = %[[INPUT]] : !loom.tile<?x?xf32>{%[[D0]], %[[D1]]})
  // CHECK:   %[[ABS:.+]] = math.absf %[[element]] : f32
  // CHECK:   loom.tile.yield %[[ABS]] : f32
  // CHECK: } -> !loom.tile<?x?xf32>{%[[D0]], %[[D1]]}
  %result = loom.tile.elementwise(%element = %input : !loom.tile<?x?xf32>{%dim0, %dim1}) {
    %abs = math.absf %element : f32
    loom.tile.yield %abs : f32
  } -> !loom.tile<?x?xf32>{%dim0, %dim1}
  return %result : !loom.tile<?x?xf32>
}

// -----

// CHECK-LABEL: @elementwise_integer
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<16xi32>, %[[BIAS:.+]]: i32
func.func @elementwise_integer(%input: !loom.tile<16xi32>, %bias: i32) -> !loom.tile<16xi32> {
  // Integer element type with captured value.
  //
  // CHECK: loom.tile.elementwise(%[[element:.+]] = %[[INPUT]] : !loom.tile<16xi32>)
  // CHECK:   %[[SUM:.+]] = arith.addi %[[element]], %[[BIAS]] : i32
  // CHECK:   loom.tile.yield %[[SUM]] : i32
  // CHECK: } -> !loom.tile<16xi32>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<16xi32>) {
    %sum = arith.addi %element, %bias : i32
    loom.tile.yield %sum : i32
  } -> !loom.tile<16xi32>
  return %result : !loom.tile<16xi32>
}

// -----

// CHECK-LABEL: @elementwise_ternary
// CHECK-SAME: %[[A:.+]]: !loom.tile<8x8xf32>, %[[B:.+]]: !loom.tile<8x8xf32>, %[[C:.+]]: !loom.tile<8x8xf32>
func.func @elementwise_ternary(%a: !loom.tile<8x8xf32>, %b: !loom.tile<8x8xf32>, %c: !loom.tile<8x8xf32>) -> !loom.tile<8x8xf32> {
  // Ternary - fused multiply-add.
  //
  // CHECK: loom.tile.elementwise(%[[EA:.+]] = %[[A]] : !loom.tile<8x8xf32>, %[[EB:.+]] = %[[B]] : !loom.tile<8x8xf32>, %[[EC:.+]] = %[[C]] : !loom.tile<8x8xf32>)
  // CHECK:   %[[FMA:.+]] = math.fma %[[EA]], %[[EB]], %[[EC]] : f32
  // CHECK:   loom.tile.yield %[[FMA]] : f32
  // CHECK: } -> !loom.tile<8x8xf32>
  %result = loom.tile.elementwise(%ea = %a : !loom.tile<8x8xf32>, %eb = %b : !loom.tile<8x8xf32>, %ec = %c : !loom.tile<8x8xf32>) {
    %fma = math.fma %ea, %eb, %ec : f32
    loom.tile.yield %fma : f32
  } -> !loom.tile<8x8xf32>
  return %result : !loom.tile<8x8xf32>
}

// -----

// CHECK-LABEL: @elementwise_1d
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<128xf32>
func.func @elementwise_1d(%input: !loom.tile<128xf32>) -> !loom.tile<128xf32> {
  // 1D tile.
  //
  // CHECK: loom.tile.elementwise(%[[element:.+]] = %[[INPUT]] : !loom.tile<128xf32>)
  // CHECK:   %[[EXP:.+]] = math.exp %[[element]] : f32
  // CHECK:   loom.tile.yield %[[EXP]] : f32
  // CHECK: } -> !loom.tile<128xf32>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<128xf32>) {
    %exp = math.exp %element : f32
    loom.tile.yield %exp : f32
  } -> !loom.tile<128xf32>
  return %result : !loom.tile<128xf32>
}

// -----

// CHECK-LABEL: @elementwise_3d
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<2x4x8xf32>
func.func @elementwise_3d(%input: !loom.tile<2x4x8xf32>) -> !loom.tile<2x4x8xf32> {
  // 3D tile.
  //
  // CHECK: loom.tile.elementwise(%[[element:.+]] = %[[INPUT]] : !loom.tile<2x4x8xf32>)
  // CHECK:   %[[SQRT:.+]] = math.sqrt %[[element]] : f32
  // CHECK:   loom.tile.yield %[[SQRT]] : f32
  // CHECK: } -> !loom.tile<2x4x8xf32>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<2x4x8xf32>) {
    %sqrt = math.sqrt %element : f32
    loom.tile.yield %sqrt : f32
  } -> !loom.tile<2x4x8xf32>
  return %result : !loom.tile<2x4x8xf32>
}

// -----

// CHECK-LABEL: @elementwise_bf16
// CHECK-SAME: %[[INPUT:.+]]: !loom.tile<4x4xbf16>
func.func @elementwise_bf16(%input: !loom.tile<4x4xbf16>) -> !loom.tile<4x4xbf16> {
  // BFloat16 element type.
  //
  // CHECK: loom.tile.elementwise(%[[element:.+]] = %[[INPUT]] : !loom.tile<4x4xbf16>)
  // CHECK:   %[[NEG:.+]] = arith.negf %[[element]] : bf16
  // CHECK:   loom.tile.yield %[[NEG]] : bf16
  // CHECK: } -> !loom.tile<4x4xbf16>
  %result = loom.tile.elementwise(%element = %input : !loom.tile<4x4xbf16>) {
    %neg = arith.negf %element : bf16
    loom.tile.yield %neg : bf16
  } -> !loom.tile<4x4xbf16>
  return %result : !loom.tile<4x4xbf16>
}

// -----

// CHECK-LABEL: @elementwise_multiple_dynamic_inputs
// CHECK-SAME: %[[A:.+]]: !loom.tile<?x?xf32>, %[[B:.+]]: !loom.tile<?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index
func.func @elementwise_multiple_dynamic_inputs(%a: !loom.tile<?x?xf32>, %b: !loom.tile<?x?xf32>, %dim0: index, %dim1: index) -> !loom.tile<?x?xf32> {
  // Multiple dynamic inputs with same shape.
  //
  // CHECK: loom.tile.elementwise(%[[EA:.+]] = %[[A]] : !loom.tile<?x?xf32>{%[[D0]], %[[D1]]}, %[[EB:.+]] = %[[B]] : !loom.tile<?x?xf32>{%[[D0]], %[[D1]]})
  // CHECK:   %[[SUM:.+]] = arith.addf %[[EA]], %[[EB]] : f32
  // CHECK:   loom.tile.yield %[[SUM]] : f32
  // CHECK: } -> !loom.tile<?x?xf32>{%[[D0]], %[[D1]]}
  %result = loom.tile.elementwise(%ea = %a : !loom.tile<?x?xf32>{%dim0, %dim1}, %eb = %b : !loom.tile<?x?xf32>{%dim0, %dim1}) {
    %sum = arith.addf %ea, %eb : f32
    loom.tile.yield %sum : f32
  } -> !loom.tile<?x?xf32>{%dim0, %dim1}
  return %result : !loom.tile<?x?xf32>
}
