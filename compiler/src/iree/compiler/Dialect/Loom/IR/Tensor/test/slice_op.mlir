// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @slice_static
func.func @slice_static(%src: !loom.tensor<256x256xf16>) -> !loom.tile<64x64xf16> {
  // Static offsets, static sizes.
  // Extracts a 64x64 tile starting at position [0, 0].
  //
  // CHECK: loom.tensor.slice %{{.+}}[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @slice_dynamic_offsets
func.func @slice_dynamic_offsets(%src: !loom.tensor<256x256xf16>, %offset0: index, %offset1: index) -> !loom.tile<64x64xf16> {
  // Dynamic offsets, static sizes.
  // Tile position is determined at runtime.
  //
  // CHECK: loom.tensor.slice %{{.+}}[%[[OFF0:.+]], %[[OFF1:.+]]] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[%offset0, %offset1] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @slice_fully_dynamic
func.func @slice_fully_dynamic(%src: !loom.tensor<?x?xf16>, %dim0: index, %dim1: index,
                               %offset0: index, %offset1: index, %size0: index, %size1: index) -> !loom.tile<?x?xf16> {
  // Dynamic tensor, dynamic offsets, dynamic tile sizes.
  // All dimensions resolved at runtime.
  //
  // CHECK: loom.tensor.slice %{{.+}}[%[[OFF0:.+]], %[[OFF1:.+]]] : !loom.tensor<?x?xf16>{%[[DIM0:.+]], %[[DIM1:.+]]} -> !loom.tile<?x?xf16>{%[[SIZE0:.+]], %[[SIZE1:.+]]}
  %tile = loom.tensor.slice %src[%offset0, %offset1]
      : !loom.tensor<?x?xf16>{%dim0, %dim1} -> !loom.tile<?x?xf16>{%size0, %size1}
  return %tile : !loom.tile<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @slice_mixed_static_dynamic_offsets
func.func @slice_mixed_static_dynamic_offsets(%src: !loom.tensor<256x256xf16>, %offset1: index) -> !loom.tile<64x64xf16> {
  // Mixed static and dynamic offsets.
  // First offset is static (0), second is dynamic.
  //
  // CHECK: loom.tensor.slice %{{.+}}[0, %{{.+}}] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[0, %offset1] : !loom.tensor<256x256xf16> -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @slice_dynamic_source_static_tile
func.func @slice_dynamic_source_static_tile(%src: !loom.tensor<?x?xf16>, %dim0: index, %dim1: index) -> !loom.tile<64x64xf16> {
  // Dynamic source tensor, static tile sizes.
  // Source dimensions tracked via {%dim0, %dim1}.
  //
  // CHECK: loom.tensor.slice %{{.+}}[0, 0] : !loom.tensor<?x?xf16>{%[[DIM0:.+]], %[[DIM1:.+]]} -> !loom.tile<64x64xf16>
  %tile = loom.tensor.slice %src[0, 0] : !loom.tensor<?x?xf16>{%dim0, %dim1} -> !loom.tile<64x64xf16>
  return %tile : !loom.tile<64x64xf16>
}

// -----

// CHECK-LABEL: @slice_static_source_dynamic_tile
func.func @slice_static_source_dynamic_tile(%src: !loom.tensor<256x256xf16>, %size0: index) -> !loom.tile<?x64xf16> {
  // Static source tensor, partially dynamic tile sizes.
  // Only the first tile dimension is dynamic.
  //
  // CHECK: loom.tensor.slice %{{.+}}[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<?x64xf16>{%{{.+}}}
  %tile = loom.tensor.slice %src[0, 0] : !loom.tensor<256x256xf16> -> !loom.tile<?x64xf16>{%size0}
  return %tile : !loom.tile<?x64xf16>
}

// -----

// CHECK-LABEL: @slice_rank1
func.func @slice_rank1(%src: !loom.tensor<1024xf32>, %offset: index) -> !loom.tile<128xf32> {
  // Rank-1 slice: simple 1D extraction.
  //
  // CHECK: loom.tensor.slice %{{.+}}[%{{.+}}] : !loom.tensor<1024xf32> -> !loom.tile<128xf32>
  %tile = loom.tensor.slice %src[%offset] : !loom.tensor<1024xf32> -> !loom.tile<128xf32>
  return %tile : !loom.tile<128xf32>
}

// -----

// CHECK-LABEL: @slice_rank3
func.func @slice_rank3(%src: !loom.tensor<8x256x256xf16>, %offset0: index) -> !loom.tile<1x64x64xf16> {
  // Rank-3 slice: batch dimension with 2D spatial tile.
  //
  // CHECK: loom.tensor.slice %{{.+}}[%{{.+}}, 0, 0] : !loom.tensor<8x256x256xf16> -> !loom.tile<1x64x64xf16>
  %tile = loom.tensor.slice %src[%offset0, 0, 0] : !loom.tensor<8x256x256xf16> -> !loom.tile<1x64x64xf16>
  return %tile : !loom.tile<1x64x64xf16>
}
