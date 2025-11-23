// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @update_static
func.func @update_static(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Static offsets, static sizes.
  // Updates a 16x16 region starting at position [0, 0].
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %updated = loom.tile.update %subtile, %tile[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @update_dynamic_offsets
func.func @update_dynamic_offsets(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>, %offset0: index, %offset1: index) -> !loom.tile<64x64xf16> {
  // Dynamic offsets, static sizes.
  // Update position is determined at runtime.
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %updated = loom.tile.update %subtile, %tile[%offset0, %offset1] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @update_fully_dynamic
func.func @update_fully_dynamic(%subtile: !loom.tile<?x?xf16>, %tile: !loom.tile<?x?xf16>,
                                %dim0: index, %dim1: index, %ssz0: index, %ssz1: index,
                                %offset0: index, %offset1: index) -> !loom.tile<?x?xf16> {
  // Dynamic target tile, dynamic update tile, dynamic offsets.
  // All dimensions resolved at runtime.
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}] : !loom.tile<?x?xf16>{%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}} -> !loom.tile<?x?xf16>{%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}}
  %updated = loom.tile.update %subtile, %tile[%offset0, %offset1]
      : !loom.tile<?x?xf16>{%ssz0, %ssz1} -> !loom.tile<?x?xf16>{%dim0, %dim1}
  return %updated : !loom.tile<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @update_mixed_static_dynamic_offsets
func.func @update_mixed_static_dynamic_offsets(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<64x64xf16>, %offset1: index) -> !loom.tile<64x64xf16> {
  // Mixed static and dynamic offsets.
  // First offset is static (0), second is dynamic.
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[0, %{{[a-z0-9_]+}}] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %updated = loom.tile.update %subtile, %tile[0, %offset1] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @update_dynamic_target_static_sub
func.func @update_dynamic_target_static_sub(%subtile: !loom.tile<16x16xf16>, %tile: !loom.tile<?x?xf16>, %dim0: index, %dim1: index) -> !loom.tile<?x?xf16> {
  // Dynamic target tile, static sub-tile sizes.
  // Target dimensions tracked via {%dim0, %dim1}.
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<?x?xf16>{%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}}
  %updated = loom.tile.update %subtile, %tile[0, 0] : !loom.tile<16x16xf16> -> !loom.tile<?x?xf16>{%dim0, %dim1}
  return %updated : !loom.tile<?x?xf16>
}

// -----

// CHECK-LABEL: @update_static_target_dynamic_sub
func.func @update_static_target_dynamic_sub(%subtile: !loom.tile<?x16xf16>, %tile: !loom.tile<64x64xf16>, %ssz0: index) -> !loom.tile<64x64xf16> {
  // Static target tile, partially dynamic sub-tile sizes.
  // Only the first sub-tile dimension is dynamic.
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[0, 0] : !loom.tile<?x16xf16>{%{{[a-z0-9_]+}}} -> !loom.tile<64x64xf16>
  %updated = loom.tile.update %subtile, %tile[0, 0] : !loom.tile<?x16xf16>{%ssz0} -> !loom.tile<64x64xf16>
  return %updated : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @update_rank1
func.func @update_rank1(%subtile: !loom.tile<128xf32>, %tile: !loom.tile<1024xf32>, %offset: index) -> !loom.tile<1024xf32> {
  // Rank-1 update: simple 1D insertion.
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}] : !loom.tile<128xf32> -> !loom.tile<1024xf32>
  %updated = loom.tile.update %subtile, %tile[%offset] : !loom.tile<128xf32> -> !loom.tile<1024xf32>
  return %updated : !loom.tile<1024xf32>
}

// -----

// CHECK-LABEL: @update_rank3
func.func @update_rank3(%subtile: !loom.tile<1x16x16xf16>, %tile: !loom.tile<8x64x64xf16>, %offset0: index) -> !loom.tile<8x64x64xf16> {
  // Rank-3 update: batch dimension with 2D spatial sub-tile.
  //
  // CHECK: loom.tile.update %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}[%{{[a-z0-9_]+}}, 0, 0] : !loom.tile<1x16x16xf16> -> !loom.tile<8x64x64xf16>
  %updated = loom.tile.update %subtile, %tile[%offset0, 0, 0] : !loom.tile<1x16x16xf16> -> !loom.tile<8x64x64xf16>
  return %updated : !loom.tile<8x64x64xf16>
}
