// RUN: iree-opt --canonicalize --split-input-file --verify-diagnostics %s

// This test verifies that poison reason chains are properly built and reported
// when poison propagates through multiple operations.

//===----------------------------------------------------------------------===//
// Test: Two-level poison propagation (slice of slice of OOB)
//===----------------------------------------------------------------------===//

func.func @two_level_poison_chain(%src: !loom.tile<32x32xf16>) -> !loom.tile<8x8xf16> {
  // First slice is OOB (offset 30 + size 16 = 46 > 32) -> poison
  // expected-remark @+3 {{ERR_LOOM_FOLD_0001: 'loom.tile.slice' folded to poison: 'slice extends beyond source bounds'}}
  // expected-note @+2 {{Fix:}}
  // expected-note @+1 {{Example:}}
  %oob = loom.tile.slice %src[30, 0] : !loom.tile<32x32xf16> -> !loom.tile<16x16xf16>
  // Second slice consumes poison -> propagates with chained reason
  // expected-remark @+3 {{ERR_LOOM_FOLD_0001: 'loom.tile.slice' folded to poison: 'slice source is poison <- slice extends beyond source bounds'}}
  // expected-note @+2 {{Fix:}}
  // expected-note @+1 {{Example:}}
  %subtile = loom.tile.slice %oob[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<8x8xf16>
  return %subtile : !loom.tile<8x8xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Test: Three-level poison propagation (slice of broadcast of poison)
//===----------------------------------------------------------------------===//

func.func @three_level_poison_chain(%good: !loom.tile<4x4xf32>) -> !loom.tile<4x4xf32> {
  // Start with poison
  %poison = ub.poison : !loom.tile<1x1xf32>
  // Broadcast poison -> poison (level 1)
  // expected-remark @+3 {{ERR_LOOM_FOLD_0001: 'loom.tile.broadcast' folded to poison: 'broadcast operand is poison'}}
  // expected-note @+2 {{Fix:}}
  // expected-note @+1 {{Example:}}
  %bc = loom.tile.broadcast<right> %poison : !loom.tile<1x1xf32> -> !loom.tile<8x8xf32>
  // Slice poison -> poison (level 2, chained)
  // expected-remark @+3 {{ERR_LOOM_FOLD_0001: 'loom.tile.slice' folded to poison: 'slice source is poison <- broadcast operand is poison'}}
  // expected-note @+2 {{Fix:}}
  // expected-note @+1 {{Example:}}
  %slice = loom.tile.slice %bc[0, 0] : !loom.tile<8x8xf32> -> !loom.tile<4x4xf32>
  // Elementwise with poison input propagates poison INTO the region as scalar
  // poison, then arith propagates it, and YieldsCaptured turns it into fill(poison).
  // (No longer emits ERR_LOOM_FOLD_0001 for elementwise - poison propagates through ops)
  %result = loom.tile.elementwise(%a = %slice : !loom.tile<4x4xf32>, %b = %good : !loom.tile<4x4xf32>) {
    %r = arith.addf %a, %b : f32
    loom.tile.yield %r : f32
  } -> !loom.tile<4x4xf32>
  return %result : !loom.tile<4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Test: Update of poison chain
//===----------------------------------------------------------------------===//

func.func @update_poison_chain(%subtile: !loom.tile<16x16xf16>) -> !loom.tile<32x32xf16> {
  // Create poison via impossible slice (same shape, non-zero offset)
  %alloca = loom.tile.alloca : !loom.tile<32x32xf16>
  // expected-remark @+3 {{ERR_LOOM_FOLD_0001: 'loom.tile.slice' folded to poison: 'same-shape slice with non-zero offset is undefined behavior'}}
  // expected-note @+2 {{Fix:}}
  // expected-note @+1 {{Example:}}
  %poison_target = loom.tile.slice %alloca[1, 0] : !loom.tile<32x32xf16> -> !loom.tile<32x32xf16>
  // Update into poison target -> propagates
  // expected-remark @+3 {{ERR_LOOM_FOLD_0001: 'loom.tile.update' folded to poison: 'update target is poison <- same-shape slice with non-zero offset is undefined behavior'}}
  // expected-note @+2 {{Fix:}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.update %subtile, %poison_target[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<32x32xf16>
  return %result : !loom.tile<32x32xf16>
}
