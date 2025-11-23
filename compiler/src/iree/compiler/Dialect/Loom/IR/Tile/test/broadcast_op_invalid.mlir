// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

func.func @error_rank_decrease(%input: !loom.tile<2x3x4xf32>) -> !loom.tile<3x4xf32> {
  // expected-error @+3 {{ERR_UTIL_SHAPE_0001: rank of 'result' must be >= rank of 'operand'}}
  // expected-note @+2 {{Fix: 'result' (rank 2) must have rank >= 'operand' (rank 3)}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.broadcast<left> %input : !loom.tile<2x3x4xf32> -> !loom.tile<3x4xf32>
  return %result : !loom.tile<3x4xf32>
}

// -----

func.func @error_incompatible_dim_left(%input: !loom.tile<3x4xf32>) -> !loom.tile<2x4xf32> {
  // Left-aligned: source dim 0 (size 3) must match result dim 0 (size 2) or be 1.
  // expected-error @+3 {{ERR_LOOM_BROADCAST_0001: incompatible dimension at source axis 0 (size 3) with result axis 0 (size 2)}}
  // expected-note @+2 {{Fix: Source axis 0 (size 3) must be 1 or match result axis 0 (size 2)}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.broadcast<left> %input : !loom.tile<3x4xf32> -> !loom.tile<2x4xf32>
  return %result : !loom.tile<2x4xf32>
}

// -----

func.func @error_incompatible_dim_right(%input: !loom.tile<3x4xf32>) -> !loom.tile<2x2x4xf32> {
  // Right-aligned: source dim 0 (size 3) must match result dim 1 (size 2) or be 1.
  // expected-error @+3 {{ERR_LOOM_BROADCAST_0001: incompatible dimension at source axis 0 (size 3) with result axis 1 (size 2)}}
  // expected-note @+2 {{Fix: Source axis 0 (size 3) must be 1 or match result axis 1 (size 2)}}
  // expected-note @+1 {{Example:}}
  %result = loom.tile.broadcast<right> %input : !loom.tile<3x4xf32> -> !loom.tile<2x2x4xf32>
  return %result : !loom.tile<2x2x4xf32>
}
