// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Canonicalization: Propagate Poison Into Region
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @propagate_poison_into_region
func.func @propagate_poison_into_region(%good: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Poison input is propagated into region as scalar poison.
  // arith propagates poison through addf, yielding poison.
  // YieldsCapturedValue pattern then converts to fill(poison).
  // CHECK: %[[POISON:.*]] = ub.poison : f32
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<4x4xf32>
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[POISON]], %[[ALLOCA]] : f32 -> !loom.tile<4x4xf32>
  // CHECK: return %[[FILL]]
  %poison = ub.poison : !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%a = %poison : !loom.tile<4x4xf32>, %b = %good : !loom.tile<4x4xf32>) {
    %r = arith.addf %a, %b : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Propagate Constants Into Region
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @propagate_constant_into_region
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<4x4xi32>
func.func @propagate_constant_into_region(%tile: !loom.tile<4x4xi32>) -> !loom.tile<4x4xi32> {
  // Splat constant input is propagated into region as scalar constant.
  // CHECK: %[[CST:.+]] = arith.constant 2 : i32
  // CHECK: %[[RESULT:.+]] = loom.tile.elementwise(%[[ARG:.+]] = %[[TILE]] : !loom.tile<4x4xi32>) {
  // CHECK:   %[[ADD:.+]] = arith.addi %[[ARG]], %[[CST]] : i32
  // CHECK:   loom.tile.yield %[[ADD]] : i32
  // CHECK: } -> !loom.tile<4x4xi32>
  // CHECK: return %[[RESULT]]
  %cst = loom.tile.constant #loom.dense<2> : !loom.tile<4x4xi32>
  %result = loom.tile.elementwise(%a = %tile : !loom.tile<4x4xi32>, %b = %cst : !loom.tile<4x4xi32>) {
    %r = arith.addi %a, %b : i32
    loom.tile.yield %r : i32
  } -> !loom.tile<4x4xi32>
  return %result : !loom.tile<4x4xi32>
}

// -----

// CHECK-LABEL: @propagate_all_constants_folds_to_fill
func.func @propagate_all_constants_folds_to_fill() -> !loom.tile<4x4xi32> {
  // When all inputs are splat constants, propagation + YieldsCaptured
  // folds entire elementwise to fill with computed scalar.
  // CHECK: %[[CST:.+]] = arith.constant 5 : i32
  // CHECK: %[[ALLOCA:.+]] = loom.tile.alloca : !loom.tile<4x4xi32>
  // CHECK: %[[FILL:.+]] = loom.tile.fill %[[CST]], %[[ALLOCA]] : i32 -> !loom.tile<4x4xi32>
  // CHECK: return %[[FILL]]
  %c2 = loom.tile.constant #loom.dense<2> : !loom.tile<4x4xi32>
  %c3 = loom.tile.constant #loom.dense<3> : !loom.tile<4x4xi32>
  %result = loom.tile.elementwise(%a = %c2 : !loom.tile<4x4xi32>, %b = %c3 : !loom.tile<4x4xi32>) {
    %r = arith.addi %a, %b : i32
    loom.tile.yield %r : i32
  } -> !loom.tile<4x4xi32>
  return %result : !loom.tile<4x4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Elementwise of Fills
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_elementwise_of_fills_binary
// CHECK-SAME: %[[VALUE1:[^:]+]]: f32
// CHECK-SAME: %[[VALUE2:[^:]+]]: f32
func.func @fold_elementwise_of_fills_binary(%value1: f32, %value2: f32) -> !loom.tile<4x4xf32> {
  // elementwise(fill(v1), fill(v2)) { add } -> fill(v1 + v2)
  // CHECK: %[[SUM:.*]] = arith.addf %[[VALUE1]], %[[VALUE2]] : f32
  // CHECK: %[[ALLOCA1:.*]] = loom.tile.alloca : !loom.tile<4x4xf32>
  // CHECK: %[[FILL1:.*]] = loom.tile.fill %[[SUM]], %[[ALLOCA1]] : f32 -> !loom.tile<4x4xf32>
  // CHECK: return %[[FILL1]]
  %alloca1 = loom.tile.alloca : !loom.tile<4x4xf32>
  %fill1 = loom.tile.fill %value1, %alloca1 : f32 -> !loom.tile<4x4xf32>
  %alloca2 = loom.tile.alloca : !loom.tile<4x4xf32>
  %fill2 = loom.tile.fill %value2, %alloca2 : f32 -> !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%a = %fill1 : !loom.tile<4x4xf32>, %b = %fill2 : !loom.tile<4x4xf32>) {
    %r = arith.addf %a, %b : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @fold_elementwise_of_fills_unary
// CHECK-SAME: %[[VALUE:[^:]+]]: f32
func.func @fold_elementwise_of_fills_unary(%value: f32) -> !loom.tile<8x8xf32> {
  // elementwise(fill(v)) { neg } -> fill(neg(v))
  // CHECK: %[[NEG:.*]] = arith.negf %[[VALUE]] : f32
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<8x8xf32>
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[NEG]], %[[ALLOCA]] : f32 -> !loom.tile<8x8xf32>
  // CHECK: return %[[FILL]]
  %alloca = loom.tile.alloca : !loom.tile<8x8xf32>
  %fill = loom.tile.fill %value, %alloca : f32 -> !loom.tile<8x8xf32>
  %result = loom.tile.elementwise(%e = %fill : !loom.tile<8x8xf32>) {
    %r = arith.negf %e : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<8x8xf32>
  return %result : !loom.tile<8x8xf32>
}

// -----

// CHECK-LABEL: @fold_elementwise_of_fills_with_capture
// CHECK-SAME: %[[VALUE:[^:]+]]: f32
// CHECK-SAME: %[[SCALE:[^:]+]]: f32
func.func @fold_elementwise_of_fills_with_capture(%value: f32, %scale: f32) -> !loom.tile<4x4xf32> {
  // Captured scalars work correctly in the fold.
  // CHECK: %[[SCALED:.*]] = arith.mulf %[[VALUE]], %[[SCALE]] : f32
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<4x4xf32>
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[SCALED]], %[[ALLOCA]] : f32 -> !loom.tile<4x4xf32>
  // CHECK: return %[[FILL]]
  %alloca = loom.tile.alloca : !loom.tile<4x4xf32>
  %fill = loom.tile.fill %value, %alloca : f32 -> !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%e = %fill : !loom.tile<4x4xf32>) {
    %r = arith.mulf %e, %scale : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @no_fold_elementwise_fill_multiple_uses
func.func @no_fold_elementwise_fill_multiple_uses(%value: f32) -> (!loom.tile<4x4xf32>, !loom.tile<4x4xf32>) {
  // Don't fold when fill has multiple uses.
  // CHECK: loom.tile.fill
  // CHECK: loom.tile.elementwise
  %alloca = loom.tile.alloca : !loom.tile<4x4xf32>
  %fill = loom.tile.fill %value, %alloca : f32 -> !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%e = %fill : !loom.tile<4x4xf32>) {
    %r = arith.negf %e : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %fill, %result : !loom.tile<4x4xf32>, !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @no_fold_elementwise_mixed_inputs
func.func @no_fold_elementwise_mixed_inputs(%value: f32, %tile: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Don't fold when not all inputs are fills.
  // CHECK: loom.tile.fill
  // CHECK: loom.tile.elementwise
  %alloca = loom.tile.alloca : !loom.tile<4x4xf32>
  %fill = loom.tile.fill %value, %alloca : f32 -> !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%a = %fill : !loom.tile<4x4xf32>, %b = %tile : !loom.tile<4x4xf32>) {
    %r = arith.addf %a, %b : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Passthrough
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_elementwise_passthrough
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<4x4xf32>
func.func @fold_elementwise_passthrough(%tile: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Region just yields block arg - replace with input.
  // CHECK-NOT: loom.tile.elementwise
  // CHECK: return %[[TILE]]
  %result = loom.tile.elementwise(%a = %tile : !loom.tile<4x4xf32>) {
    loom.tile.yield %a : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @fold_elementwise_passthrough_second_arg
// CHECK-SAME: %[[TILE1:[^:]+]]: !loom.tile<4x4xf32>
// CHECK-SAME: %[[TILE2:[^:]+]]: !loom.tile<4x4xf32>
func.func @fold_elementwise_passthrough_second_arg(%tile1: !loom.tile<4x4xf32>, %tile2: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Yields second block arg - replace with second input.
  // CHECK-NOT: loom.tile.elementwise
  // CHECK: return %[[TILE2]]
  %result = loom.tile.elementwise(%a = %tile1 : !loom.tile<4x4xf32>, %b = %tile2 : !loom.tile<4x4xf32>) {
    loom.tile.yield %b : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Yields Captured Value
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_elementwise_yields_captured
// CHECK-SAME: %[[SCALAR:[^:]+]]: f32
func.func @fold_elementwise_yields_captured(%tile: !loom.tile<4x4xf32>, %scalar: f32) -> !loom.tile<4x4xf32> {
  // Region ignores block args and yields captured scalar - convert to fill.
  // CHECK-NOT: loom.tile.elementwise
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<4x4xf32>
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[SCALAR]], %[[ALLOCA]] : f32 -> !loom.tile<4x4xf32>
  // CHECK: return %[[FILL]]
  %result = loom.tile.elementwise(%a = %tile : !loom.tile<4x4xf32>) {
    loom.tile.yield %scalar : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @fold_elementwise_yields_captured_with_computation
// CHECK-SAME: %[[S1:[^:]+]]: f32
// CHECK-SAME: %[[S2:[^:]+]]: f32
func.func @fold_elementwise_yields_captured_with_computation(%tile: !loom.tile<4x4xf32>, %s1: f32, %s2: f32) -> !loom.tile<4x4xf32> {
  // Region computes from captured scalars only - compute and fill.
  // CHECK-NOT: loom.tile.elementwise
  // CHECK: %[[SUM:.*]] = arith.addf %[[S1]], %[[S2]] : f32
  // CHECK: %[[ALLOCA:.*]] = loom.tile.alloca : !loom.tile<4x4xf32>
  // CHECK: %[[FILL:.*]] = loom.tile.fill %[[SUM]], %[[ALLOCA]] : f32 -> !loom.tile<4x4xf32>
  // CHECK: return %[[FILL]]
  %result = loom.tile.elementwise(%a = %tile : !loom.tile<4x4xf32>) {
    %sum = arith.addf %s1, %s2 : f32
    loom.tile.yield %sum : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Eliminate Unused Inputs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @eliminate_unused_input
// CHECK-SAME: %[[TILE1:[^:]+]]: !loom.tile<4x4xf32>
// CHECK-SAME: %[[TILE2:[^:]+]]: !loom.tile<4x4xf32>
func.func @eliminate_unused_input(%tile1: !loom.tile<4x4xf32>, %tile2: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Second input is unused - eliminate it.
  // CHECK: %[[RESULT:.*]] = loom.tile.elementwise(%[[ARG:.*]] = %[[TILE1]] : !loom.tile<4x4xf32>) {
  // CHECK:   %[[NEG:.*]] = arith.negf %[[ARG]] : f32
  // CHECK:   loom.tile.yield %[[NEG]] : f32
  // CHECK: } -> !loom.tile<4x4xf32>
  // CHECK: return %[[RESULT]]
  %result = loom.tile.elementwise(%a = %tile1 : !loom.tile<4x4xf32>, %b = %tile2 : !loom.tile<4x4xf32>) {
    %r = arith.negf %a : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Fuse Elementwise Chain
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fuse_elementwise_chain
// CHECK-SAME: %[[X:[^:]+]]: !loom.tile<4x4xf32>
// CHECK-SAME: %[[Y:[^:]+]]: !loom.tile<4x4xf32>
// CHECK-SAME: %[[Z:[^:]+]]: !loom.tile<4x4xf32>
func.func @fuse_elementwise_chain(%x: !loom.tile<4x4xf32>, %y: !loom.tile<4x4xf32>, %z: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Fuse chained elementwise ops when intermediate has single use.
  // CHECK: %[[RESULT:.+]] = loom.tile.elementwise(%[[A:.+]] = %[[X]] : !loom.tile<4x4xf32>, %[[B:.+]] = %[[Y]] : !loom.tile<4x4xf32>, %[[D:.+]] = %[[Z]] : !loom.tile<4x4xf32>) {
  // CHECK:   %[[SUM:.+]] = arith.addf %[[A]], %[[B]] : f32
  // CHECK:   %[[MUL:.+]] = arith.mulf %[[SUM]], %[[D]] : f32
  // CHECK:   loom.tile.yield %[[MUL]] : f32
  // CHECK: } -> !loom.tile<4x4xf32>
  // CHECK: return %[[RESULT]]
  %tmp = loom.tile.elementwise(%a = %x : !loom.tile<4x4xf32>, %b = %y : !loom.tile<4x4xf32>) {
    %r = arith.addf %a, %b : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%c = %tmp : !loom.tile<4x4xf32>, %d = %z : !loom.tile<4x4xf32>) {
    %r = arith.mulf %c, %d : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

// CHECK-LABEL: @no_fuse_elementwise_multiple_uses
func.func @no_fuse_elementwise_multiple_uses(%x: !loom.tile<4x4xf32>, %y: !loom.tile<4x4xf32>, %z: !loom.tile<4x4xf32>) -> (!loom.tile<4x4xf32>, !loom.tile<4x4xf32>) {
  // Don't fuse when intermediate has multiple uses.
  // CHECK: loom.tile.elementwise
  // CHECK: loom.tile.elementwise
  %tmp = loom.tile.elementwise(%a = %x : !loom.tile<4x4xf32>, %b = %y : !loom.tile<4x4xf32>) {
    %r = arith.addf %a, %b : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  %result = loom.tile.elementwise(%c = %tmp : !loom.tile<4x4xf32>, %d = %z : !loom.tile<4x4xf32>) {
    %r = arith.mulf %c, %d : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %tmp, %result : !loom.tile<4x4xf32>, !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization: Sink Broadcasts Over Elementwise
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sink_broadcasts_over_elementwise_unary
// CHECK-SAME: %[[SRC:[^:]+]]: !loom.tile<4x4xf32>
func.func @sink_broadcasts_over_elementwise_unary(%src: !loom.tile<4x4xf32>) -> !loom.tile<8x4x4xf32> {
  // broadcast(x) -> elementwise -> should become elementwise(x) -> broadcast
  // CHECK: %[[EW:.+]] = loom.tile.elementwise(%[[A:.+]] = %[[SRC]] : !loom.tile<4x4xf32>) {
  // CHECK:   %[[NEG:.+]] = arith.negf %[[A]] : f32
  // CHECK:   loom.tile.yield %[[NEG]] : f32
  // CHECK: } -> !loom.tile<4x4xf32>
  // CHECK: %[[BC:.+]] = loom.tile.broadcast<right> %[[EW]] : !loom.tile<4x4xf32> -> !loom.tile<8x4x4xf32>
  // CHECK: return %[[BC]]
  %bc = loom.tile.broadcast<right> %src : !loom.tile<4x4xf32> -> !loom.tile<8x4x4xf32>
  %result = loom.tile.elementwise(%a = %bc : !loom.tile<8x4x4xf32>) {
    %r = arith.negf %a : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<8x4x4xf32>
  return %result : !loom.tile<8x4x4xf32>
}

// -----

// CHECK-LABEL: @sink_broadcasts_over_elementwise_binary
// CHECK-SAME: %[[A:[^:]+]]: !loom.tile<4x4xf32>
// CHECK-SAME: %[[B:[^:]+]]: !loom.tile<4x4xf32>
func.func @sink_broadcasts_over_elementwise_binary(%a: !loom.tile<4x4xf32>, %b: !loom.tile<4x4xf32>) -> !loom.tile<8x4x4xf32> {
  // Both inputs are broadcasts from same source shape -> sink broadcasts.
  // CHECK: %[[EW:.+]] = loom.tile.elementwise(%[[ARG_A:.+]] = %[[A]] : !loom.tile<4x4xf32>, %[[ARG_B:.+]] = %[[B]] : !loom.tile<4x4xf32>) {
  // CHECK:   %[[SUM:.+]] = arith.addf %[[ARG_A]], %[[ARG_B]] : f32
  // CHECK:   loom.tile.yield %[[SUM]] : f32
  // CHECK: } -> !loom.tile<4x4xf32>
  // CHECK: %[[BC:.+]] = loom.tile.broadcast<right> %[[EW]] : !loom.tile<4x4xf32> -> !loom.tile<8x4x4xf32>
  // CHECK: return %[[BC]]
  %bc_a = loom.tile.broadcast<right> %a : !loom.tile<4x4xf32> -> !loom.tile<8x4x4xf32>
  %bc_b = loom.tile.broadcast<right> %b : !loom.tile<4x4xf32> -> !loom.tile<8x4x4xf32>
  %result = loom.tile.elementwise(%x = %bc_a : !loom.tile<8x4x4xf32>, %y = %bc_b : !loom.tile<8x4x4xf32>) {
    %r = arith.addf %x, %y : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<8x4x4xf32>
  return %result : !loom.tile<8x4x4xf32>
}

// -----

// CHECK-LABEL: @no_sink_broadcasts_non_broadcast_input
func.func @no_sink_broadcasts_non_broadcast_input(%a: !loom.tile<4x4xf32>, %b: !loom.tile<8x4x4xf32>) -> !loom.tile<8x4x4xf32> {
  // One input is not a broadcast - cannot sink.
  // CHECK: loom.tile.broadcast<right>
  // CHECK: loom.tile.elementwise
  %bc_a = loom.tile.broadcast<right> %a : !loom.tile<4x4xf32> -> !loom.tile<8x4x4xf32>
  %result = loom.tile.elementwise(%x = %bc_a : !loom.tile<8x4x4xf32>, %y = %b : !loom.tile<8x4x4xf32>) {
    %r = arith.addf %x, %y : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<8x4x4xf32>
  return %result : !loom.tile<8x4x4xf32>
}
