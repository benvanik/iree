// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Loom Dialect: Comprehensive Tile Operation Examples
//===----------------------------------------------------------------------===//
//
// This file demonstrates the canonical tiling patterns using Loom operations.
// These examples serve as documentation for the dialect and should be
// referenced when learning how to use Loom for tiled computation.
//
// Key concepts:
// - tensor: Global memory storage (e.g., GPU VRAM, system memory)
// - tile:   Local compute memory (e.g., shared memory, registers)
//
// Typical data flow:
//   tensor --[slice]--> tile --[compute]--> tile --[update]--> tensor
//
// TODO: Add examples with encoding attributes once encoding tests are ready.
// TODO: Apply this documentation style to other ops (broadcast, alloca, etc.).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Basic Tiling Pattern: slice -> compute -> update
//===----------------------------------------------------------------------===//
// This is the fundamental pattern for tiled computation in Loom.
// Data flows: input tensor -> tile -> computation -> tile -> output tensor

// CHECK-LABEL: @basic_tiling_pattern
func.func @basic_tiling_pattern(
    %input: !loom.tensor<256x256xf16>,
    %output: !loom.tensor<256x256xf16>,
    %tile_rows: index,
    %tile_cols: index) -> !loom.tensor<256x256xf16> {
  // Step 1: Extract a tile from the input tensor (tensor -> tile boundary).
  // This conceptually copies data from global memory to local compute memory.
  //
  // CHECK: loom.tensor.slice %{{.+}}[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<?x?xf16>
  %in_tile = loom.tensor.slice %input[0, 0]
      : !loom.tensor<256x256xf16> -> !loom.tile<?x?xf16>{%tile_rows, %tile_cols}

  // Step 2: Perform elementwise computation on the tile.
  // All computation happens in tile-space (local memory).
  //
  // CHECK: loom.tile.elementwise
  %out_tile = loom.tile.elementwise(%element = %in_tile : !loom.tile<?x?xf16>{%tile_rows, %tile_cols}) {
    // Double each element.
    %two = arith.constant 2.0 : f16
    %result = arith.mulf %element, %two : f16
    loom.tile.yield %result : f16
  } -> !loom.tile<?x?xf16>{%tile_rows, %tile_cols}

  // Step 3: Update the output tensor with the computed tile (tile -> tensor boundary)
  // This conceptually copies data from local memory back to global memory.
  // The result is tied to the target, enabling in-place updates.
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[0, 0] : !loom.tile<?x?xf16>{{.*}} -> !loom.tensor<256x256xf16>
  %result = loom.tensor.update %out_tile, %output[0, 0]
      : !loom.tile<?x?xf16>{%tile_rows, %tile_cols} -> !loom.tensor<256x256xf16>

  return %result : !loom.tensor<256x256xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Tiled Loop Pattern: scf.for with slice/update
//===----------------------------------------------------------------------===//
// Real programs iterate over tiles. This shows the typical loop structure.

// CHECK-LABEL: @tiled_loop_pattern
func.func @tiled_loop_pattern(
    %input: !loom.tensor<256x256xf16>,
    %output: !loom.tensor<256x256xf16>,
    %tile_rows: index,
    %tile_cols: index) -> !loom.tensor<256x256xf16> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index

  // Iterate over row tiles.
  %result = scf.for %i = %c0 to %c256 step %tile_rows
      iter_args(%out_iter = %output) -> !loom.tensor<256x256xf16> {

    // Iterate over column tiles.
    %inner_result = scf.for %j = %c0 to %c256 step %tile_cols
        iter_args(%out_inner = %out_iter) -> !loom.tensor<256x256xf16> {

      // Extract input tile at position [i, j].
      // CHECK: loom.tensor.slice %{{.+}}[%{{.+}}, %{{.+}}]
      %in_tile = loom.tensor.slice %input[%i, %j]
          : !loom.tensor<256x256xf16> -> !loom.tile<?x?xf16>{%tile_rows, %tile_cols}

      // Compute on the tile (negation as example).
      // CHECK: loom.tile.elementwise
      %out_tile = loom.tile.elementwise(%element = %in_tile : !loom.tile<?x?xf16>{%tile_rows, %tile_cols}) {
        %neg = arith.negf %element : f16
        loom.tile.yield %neg : f16
      } -> !loom.tile<?x?xf16>{%tile_rows, %tile_cols}

      // Update output tensor at position [i, j].
      // The iter_arg pattern ensures proper SSA form for in-place updates.
      // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}]
      %updated = loom.tensor.update %out_tile, %out_inner[%i, %j]
          : !loom.tile<?x?xf16>{%tile_rows, %tile_cols} -> !loom.tensor<256x256xf16>

      scf.yield %updated : !loom.tensor<256x256xf16>
    }
    scf.yield %inner_result : !loom.tensor<256x256xf16>
  }

  return %result : !loom.tensor<256x256xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Hierarchical Tiling: Two-Level Tile Decomposition
//===----------------------------------------------------------------------===//
// For GPU programming, tiles are often decomposed hierarchically:
//   tensor -> workgroup tile -> subgroup/thread tile
// This example shows nested tile.slice/tile.update operations.

// CHECK-LABEL: @hierarchical_tiling
// CHECK-SAME: %[[INPUT:[^:]+]]: !loom.tensor<256x256xf16>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: !loom.tensor<256x256xf16>
// CHECK-SAME: %[[WG_ROWS:[^:]+]]: index
// CHECK-SAME: %[[WG_COLS:[^:]+]]: index
// CHECK-SAME: %[[SG_ROWS:[^:]+]]: index
// CHECK-SAME: %[[SG_COLS:[^:]+]]: index
func.func @hierarchical_tiling(
    %input: !loom.tensor<256x256xf16>,
    %output: !loom.tensor<256x256xf16>,
    %wg_rows: index,     // workgroup tile size (e.g., 64)
    %wg_cols: index,
    %sg_rows: index,     // subgroup tile size (e.g., 16)
    %sg_cols: index) -> !loom.tensor<256x256xf16> {

  // Level 1: tensor -> workgroup tile.
  // CHECK: %[[WG_TILE:.+]] = loom.tensor.slice %[[INPUT]][0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<?x?xf16>{%[[WG_ROWS]], %[[WG_COLS]]}
  %wg_tile = loom.tensor.slice %input[0, 0]
      : !loom.tensor<256x256xf16> -> !loom.tile<?x?xf16>{%wg_rows, %wg_cols}

  // Level 2: workgroup tile -> subgroup tile.
  // This is where loom.tile.slice is used (tile-to-tile, not tensor-to-tile)
  // CHECK: %[[SG_TILE:.+]] = loom.tile.slice %[[WG_TILE]][0, 0] : !loom.tile<?x?xf16>{%[[WG_ROWS]], %[[WG_COLS]]} -> !loom.tile<?x?xf16>{%[[SG_ROWS]], %[[SG_COLS]]}
  %sg_tile = loom.tile.slice %wg_tile[0, 0]
      : !loom.tile<?x?xf16>{%wg_rows, %wg_cols} -> !loom.tile<?x?xf16>{%sg_rows, %sg_cols}

  // Compute at subgroup level (finest granularity).
  // CHECK: %[[SG_RESULT:.+]] = loom.tile.elementwise(%{{.+}} = %[[SG_TILE]] : !loom.tile<?x?xf16>{%[[SG_ROWS]], %[[SG_COLS]]})
  %sg_result = loom.tile.elementwise(%element = %sg_tile : !loom.tile<?x?xf16>{%sg_rows, %sg_cols}) {
    %one = arith.constant 1.0 : f16
    %result = arith.addf %element, %one : f16
    loom.tile.yield %result : f16
  } -> !loom.tile<?x?xf16>{%sg_rows, %sg_cols}

  // Level 2 reverse: subgroup tile -> workgroup tile.
  // CHECK: %[[WG_RESULT:.+]] = loom.tile.update %[[SG_RESULT]], %[[WG_TILE]][0, 0] : !loom.tile<?x?xf16>{%[[SG_ROWS]], %[[SG_COLS]]} -> !loom.tile<?x?xf16>{%[[WG_ROWS]], %[[WG_COLS]]}
  %wg_result = loom.tile.update %sg_result, %wg_tile[0, 0]
      : !loom.tile<?x?xf16>{%sg_rows, %sg_cols} -> !loom.tile<?x?xf16>{%wg_rows, %wg_cols}

  // Level 1 reverse: workgroup tile -> tensor.
  // CHECK: loom.tensor.update %[[WG_RESULT]], %[[OUTPUT]][0, 0] : !loom.tile<?x?xf16>{%[[WG_ROWS]], %[[WG_COLS]]} -> !loom.tensor<256x256xf16>
  %result = loom.tensor.update %wg_result, %output[0, 0]
      : !loom.tile<?x?xf16>{%wg_rows, %wg_cols} -> !loom.tensor<256x256xf16>

  return %result : !loom.tensor<256x256xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Multiple Input Tiles: Binary Operations
//===----------------------------------------------------------------------===//
// Many operations combine multiple input tiles (add, mul, matmul, etc.).

// CHECK-LABEL: @binary_tile_operation
func.func @binary_tile_operation(
    %lhs_tensor: !loom.tensor<256x256xf16>,
    %rhs_tensor: !loom.tensor<256x256xf16>,
    %output: !loom.tensor<256x256xf16>,
    %tile_rows: index,
    %tile_cols: index) -> !loom.tensor<256x256xf16> {

  // Extract tiles from both input tensors.
  // CHECK: loom.tensor.slice
  %lhs_tile = loom.tensor.slice %lhs_tensor[0, 0]
      : !loom.tensor<256x256xf16> -> !loom.tile<?x?xf16>{%tile_rows, %tile_cols}

  // CHECK: loom.tensor.slice
  %rhs_tile = loom.tensor.slice %rhs_tensor[0, 0]
      : !loom.tensor<256x256xf16> -> !loom.tile<?x?xf16>{%tile_rows, %tile_cols}

  // Binary elementwise operation (addition).
  // CHECK: loom.tile.elementwise
  %result_tile = loom.tile.elementwise(
      %lhs_elem = %lhs_tile : !loom.tile<?x?xf16>{%tile_rows, %tile_cols},
      %rhs_elem = %rhs_tile : !loom.tile<?x?xf16>{%tile_rows, %tile_cols}) {
    %sum = arith.addf %lhs_elem, %rhs_elem : f16
    loom.tile.yield %sum : f16
  } -> !loom.tile<?x?xf16>{%tile_rows, %tile_cols}

  // Write result back to output tensor.
  // CHECK: loom.tensor.update
  %result = loom.tensor.update %result_tile, %output[0, 0]
      : !loom.tile<?x?xf16>{%tile_rows, %tile_cols} -> !loom.tensor<256x256xf16>

  return %result : !loom.tensor<256x256xf16>
}
