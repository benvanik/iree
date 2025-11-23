// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @update_static
func.func @update_static(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>) -> !loom.tensor<256x256xf16> {
  // Static offsets, static sizes.
  // Updates a 64x64 region starting at position [0, 0].
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @update_dynamic_offsets
func.func @update_dynamic_offsets(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>, %offset0: index, %offset1: index) -> !loom.tensor<256x256xf16> {
  // Dynamic offsets, static sizes.
  // Update position is determined at runtime.
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[%offset0, %offset1] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @update_fully_dynamic
// CHECK-SAME: %[[TILE:[^:]+]]: !loom.tile<?x?xf16>
// CHECK-SAME: %[[TENSOR:[^:]+]]: !loom.tensor<?x?xf16>
// CHECK-SAME: %[[DIM0:[^:]+]]: index
// CHECK-SAME: %[[DIM1:[^:]+]]: index
// CHECK-SAME: %[[TSZ0:[^:]+]]: index
// CHECK-SAME: %[[TSZ1:[^:]+]]: index
// CHECK-SAME: %[[OFF0:[^:]+]]: index
// CHECK-SAME: %[[OFF1:[^:]+]]: index
func.func @update_fully_dynamic(%tile: !loom.tile<?x?xf16>, %tensor: !loom.tensor<?x?xf16>,
                                %dim0: index, %dim1: index, %tsz0: index, %tsz1: index,
                                %offset0: index, %offset1: index) -> !loom.tensor<?x?xf16> {
  // Dynamic tensor, dynamic tile, dynamic offsets.
  // All dimensions resolved at runtime.
  //
  // CHECK: loom.tensor.update %[[TILE]], %[[TENSOR]][%[[OFF0]], %[[OFF1]]] : !loom.tile<?x?xf16>{%[[TSZ0]], %[[TSZ1]]} -> !loom.tensor<?x?xf16>{%[[DIM0]], %[[DIM1]]}
  %updated = loom.tensor.update %tile, %tensor[%offset0, %offset1]
      : !loom.tile<?x?xf16>{%tsz0, %tsz1} -> !loom.tensor<?x?xf16>{%dim0, %dim1}
  return %updated : !loom.tensor<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @update_mixed_static_dynamic_offsets
func.func @update_mixed_static_dynamic_offsets(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<256x256xf16>, %offset1: index) -> !loom.tensor<256x256xf16> {
  // Mixed static and dynamic offsets.
  // First offset is static (0), second is dynamic.
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[0, %{{.+}}] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[0, %offset1] : !loom.tile<64x64xf16> -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @update_dynamic_tensor_static_tile
func.func @update_dynamic_tensor_static_tile(%tile: !loom.tile<64x64xf16>, %tensor: !loom.tensor<?x?xf16>, %dim0: index, %dim1: index) -> !loom.tensor<?x?xf16> {
  // Dynamic tensor, static tile sizes.
  // Tensor dimensions tracked via {%dim0, %dim1}.
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<?x?xf16>{%{{.+}}, %{{.+}}}
  %updated = loom.tensor.update %tile, %tensor[0, 0] : !loom.tile<64x64xf16> -> !loom.tensor<?x?xf16>{%dim0, %dim1}
  return %updated : !loom.tensor<?x?xf16>
}

// -----

// CHECK-LABEL: @update_static_tensor_dynamic_tile
func.func @update_static_tensor_dynamic_tile(%tile: !loom.tile<?x64xf16>, %tensor: !loom.tensor<256x256xf16>, %tsz0: index) -> !loom.tensor<256x256xf16> {
  // Static tensor, partially dynamic tile sizes.
  // Only the first tile dimension is dynamic.
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[0, 0] : !loom.tile<?x64xf16>{%{{.+}}} -> !loom.tensor<256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[0, 0] : !loom.tile<?x64xf16>{%tsz0} -> !loom.tensor<256x256xf16>
  return %updated : !loom.tensor<256x256xf16>
}

// -----

// CHECK-LABEL: @update_rank1
func.func @update_rank1(%tile: !loom.tile<128xf32>, %tensor: !loom.tensor<1024xf32>, %offset: index) -> !loom.tensor<1024xf32> {
  // Rank-1 update: simple 1D insertion.
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[%{{.+}}] : !loom.tile<128xf32> -> !loom.tensor<1024xf32>
  %updated = loom.tensor.update %tile, %tensor[%offset] : !loom.tile<128xf32> -> !loom.tensor<1024xf32>
  return %updated : !loom.tensor<1024xf32>
}

// -----

// CHECK-LABEL: @update_rank3
func.func @update_rank3(%tile: !loom.tile<1x64x64xf16>, %tensor: !loom.tensor<8x256x256xf16>, %offset0: index) -> !loom.tensor<8x256x256xf16> {
  // Rank-3 update: batch dimension with 2D spatial tile.
  //
  // CHECK: loom.tensor.update %{{.+}}, %{{.+}}[%{{.+}}, 0, 0] : !loom.tile<1x64x64xf16> -> !loom.tensor<8x256x256xf16>
  %updated = loom.tensor.update %tile, %tensor[%offset0, 0, 0] : !loom.tile<1x64x64xf16> -> !loom.tensor<8x256x256xf16>
  return %updated : !loom.tensor<8x256x256xf16>
}
