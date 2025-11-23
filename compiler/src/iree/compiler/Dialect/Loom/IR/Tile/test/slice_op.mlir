// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @slice_static
func.func @slice_static(%src: !loom.tile<64x64xf16>) -> !loom.tile<16x16xf16> {
  // Static offsets, static sizes.
  // Extracts a 16x16 sub-tile starting at position [0, 0].
  //
  // CHECK: loom.tile.slice %{{.+}}[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @slice_dynamic_offsets
func.func @slice_dynamic_offsets(%src: !loom.tile<64x64xf16>, %offset0: index, %offset1: index) -> !loom.tile<16x16xf16> {
  // Dynamic offsets, static sizes.
  // Sub-tile position is determined at runtime.
  //
  // CHECK: loom.tile.slice %{{.+}}[%[[OFF0:.+]], %[[OFF1:.+]]] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  %subtile = loom.tile.slice %src[%offset0, %offset1] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @slice_fully_dynamic
func.func @slice_fully_dynamic(%src: !loom.tile<?x?xf16>, %dim0: index, %dim1: index,
                               %offset0: index, %offset1: index, %size0: index, %size1: index) -> !loom.tile<?x?xf16> {
  // Dynamic source tile, dynamic offsets, dynamic result sizes.
  // All dimensions resolved at runtime.
  //
  // CHECK: loom.tile.slice %{{.+}}[%[[OFF0:.+]], %[[OFF1:.+]]] : !loom.tile<?x?xf16>{%[[DIM0:.+]], %[[DIM1:.+]]} -> !loom.tile<?x?xf16>{%[[SIZE0:.+]], %[[SIZE1:.+]]}
  %subtile = loom.tile.slice %src[%offset0, %offset1]
      : !loom.tile<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%size0, %size1}
  return %subtile : !loom.tile<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @slice_mixed_static_dynamic_offsets
func.func @slice_mixed_static_dynamic_offsets(%src: !loom.tile<64x64xf16>, %offset1: index) -> !loom.tile<16x16xf16> {
  // Mixed static and dynamic offsets.
  // First offset is static (0), second is dynamic.
  //
  // CHECK: loom.tile.slice %{{.+}}[0, %{{.+}}] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  %subtile = loom.tile.slice %src[0, %offset1] : !loom.tile<64x64xf16> -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @slice_dynamic_source_static_result
func.func @slice_dynamic_source_static_result(%src: !loom.tile<?x?xf16>, %dim0: index, %dim1: index) -> !loom.tile<16x16xf16> {
  // Dynamic source tile, static result sizes.
  // Source dimensions tracked via {%dim0, %dim1}.
  //
  // CHECK: loom.tile.slice %{{.+}}[0, 0] : !loom.tile<?x?xf16>{%[[DIM0:.+]], %[[DIM1:.+]]} -> !loom.tile<16x16xf16>
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<?x?xf16>{%dim0, %dim1} -> !loom.tile<16x16xf16>
  return %subtile : !loom.tile<16x16xf16>
}

// -----

// CHECK-LABEL: @slice_static_source_dynamic_result
func.func @slice_static_source_dynamic_result(%src: !loom.tile<64x64xf16>, %size0: index) -> !loom.tile<?x16xf16> {
  // Static source tile, partially dynamic result sizes.
  // Only the first result dimension is dynamic.
  //
  // CHECK: loom.tile.slice %{{.+}}[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<?x16xf16>{%{{.+}}}
  %subtile = loom.tile.slice %src[0, 0] : !loom.tile<64x64xf16> -> !loom.tile<?x16xf16>{%size0}
  return %subtile : !loom.tile<?x16xf16>
}

// -----

// CHECK-LABEL: @slice_rank1
func.func @slice_rank1(%src: !loom.tile<1024xf32>, %offset: index) -> !loom.tile<128xf32> {
  // Rank-1 slice: simple 1D extraction.
  //
  // CHECK: loom.tile.slice %{{.+}}[%{{.+}}] : !loom.tile<1024xf32> -> !loom.tile<128xf32>
  %subtile = loom.tile.slice %src[%offset] : !loom.tile<1024xf32> -> !loom.tile<128xf32>
  return %subtile : !loom.tile<128xf32>
}

// -----

// CHECK-LABEL: @slice_rank3
func.func @slice_rank3(%src: !loom.tile<8x64x64xf16>, %offset0: index) -> !loom.tile<1x16x16xf16> {
  // Rank-3 slice: batch dimension with 2D spatial sub-tile.
  //
  // CHECK: loom.tile.slice %{{.+}}[%{{.+}}, 0, 0] : !loom.tile<8x64x64xf16> -> !loom.tile<1x16x16xf16>
  %subtile = loom.tile.slice %src[%offset0, 0, 0] : !loom.tile<8x64x64xf16> -> !loom.tile<1x16x16xf16>
  return %subtile : !loom.tile<1x16x16xf16>
}
