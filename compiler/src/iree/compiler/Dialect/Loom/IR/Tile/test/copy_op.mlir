// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Documented Examples (referenced by OpDocMetadata)
//===----------------------------------------------------------------------===//

// Tests that copy with static offsets and sizes round-trips correctly.
// CHECK-LABEL: @copy_static
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @copy_static(%source: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // Copies an 8x8 region from source at [0,0] to target at [16,32].
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][16, 32], [8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[16, 32], [8, 8]
      : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that copy with dynamic offsets round-trips correctly.
// CHECK-LABEL: @copy_dynamic_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[SOURCE_OFFSET0:[^:]+]]: index
// CHECK-SAME: %[[SOURCE_OFFSET1:[^:]+]]: index
// CHECK-SAME: %[[TARGET_OFFSET0:[^:]+]]: index
// CHECK-SAME: %[[TARGET_OFFSET1:[^:]+]]: index
func.func @copy_dynamic_offsets(%source: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>,
                                %source_offset0: index, %source_offset1: index, %target_offset0: index, %target_offset1: index) -> !loom.tile<64x64xf16> {
  // Source and target positions determined at runtime.
  // CHECK: loom.tile.copy %[[SOURCE]][%[[SOURCE_OFFSET0]], %[[SOURCE_OFFSET1]]], %[[TARGET]][%[[TARGET_OFFSET0]], %[[TARGET_OFFSET1]]], [8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[%source_offset0, %source_offset1], %target[%target_offset0, %target_offset1], [8, 8]
      : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that fully dynamic copy (tiles, offsets, sizes) round-trips correctly.
// CHECK-LABEL: @copy_fully_dynamic
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<?x?xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<?x?xf16>
// CHECK-SAME: %[[SOURCE_DIM0:[^:]+]]: index
// CHECK-SAME: %[[SOURCE_DIM1:[^:]+]]: index
// CHECK-SAME: %[[TARGET_DIM0:[^:]+]]: index
// CHECK-SAME: %[[TARGET_DIM1:[^:]+]]: index
// CHECK-SAME: %[[SOURCE_OFFSET0:[^:]+]]: index
// CHECK-SAME: %[[SOURCE_OFFSET1:[^:]+]]: index
// CHECK-SAME: %[[TARGET_OFFSET0:[^:]+]]: index
// CHECK-SAME: %[[TARGET_OFFSET1:[^:]+]]: index
// CHECK-SAME: %[[SIZE0:[^:]+]]: index
// CHECK-SAME: %[[SIZE1:[^:]+]]: index
func.func @copy_fully_dynamic(%source: !loom.tile<?x?xf16>, %target: !loom.tile<?x?xf16>,
                              %source_dim0: index, %source_dim1: index, %target_dim0: index, %target_dim1: index,
                              %source_offset0: index, %source_offset1: index, %target_offset0: index, %target_offset1: index,
                              %size0: index, %size1: index) -> !loom.tile<?x?xf16> {
  // All dimensions resolved at runtime.
  // CHECK: loom.tile.copy %[[SOURCE]][%[[SOURCE_OFFSET0]], %[[SOURCE_OFFSET1]]], %[[TARGET]][%[[TARGET_OFFSET0]], %[[TARGET_OFFSET1]]], [%[[SIZE0]], %[[SIZE1]]]
  // CHECK-SAME: : !loom.tile<?x?xf16>{%[[SOURCE_DIM0]], %[[SOURCE_DIM1]]} -> !loom.tile<?x?xf16>{%[[TARGET_DIM0]], %[[TARGET_DIM1]]}
  %result = loom.tile.copy %source[%source_offset0, %source_offset1], %target[%target_offset0, %target_offset1], [%size0, %size1]
      : !loom.tile<?x?xf16>{%source_dim0, %source_dim1} -> !loom.tile<?x?xf16>{%target_dim0, %target_dim1}
  return %result : !loom.tile<?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Additional Test Cases
//===----------------------------------------------------------------------===//

// Tests that mixed static/dynamic source offsets round-trip correctly.
// CHECK-LABEL: @copy_mixed_static_dynamic_source_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[SOURCE_OFFSET1:[^:]+]]: index
func.func @copy_mixed_static_dynamic_source_offsets(%source: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>,
                                                    %source_offset1: index) -> !loom.tile<64x64xf16> {
  // First source offset is static (0), second is dynamic.
  // CHECK: loom.tile.copy %[[SOURCE]][0, %[[SOURCE_OFFSET1]]], %[[TARGET]][16, 32], [8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, %source_offset1], %target[16, 32], [8, 8]
      : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that mixed static/dynamic target offsets round-trip correctly.
// CHECK-LABEL: @copy_mixed_static_dynamic_target_offsets
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET_OFFSET1:[^:]+]]: index
func.func @copy_mixed_static_dynamic_target_offsets(%source: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>,
                                                    %target_offset1: index) -> !loom.tile<64x64xf16> {
  // First target offset is static (16), second is dynamic.
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][16, %[[TARGET_OFFSET1]]], [8, 8] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[16, %target_offset1], [8, 8]
      : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that mixed static/dynamic sizes round-trip correctly.
// CHECK-LABEL: @copy_mixed_static_dynamic_sizes
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<16x16xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[SIZE1:[^:]+]]: index
func.func @copy_mixed_static_dynamic_sizes(%source: !loom.tile<16x16xf16>, %target: !loom.tile<64x64xf16>,
                                           %size1: index) -> !loom.tile<64x64xf16> {
  // First size is static (8), second is dynamic.
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][0, 0], [8, %[[SIZE1]]] : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[0, 0], [8, %size1]
      : !loom.tile<16x16xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that copy between tiles of the same type round-trips correctly.
// CHECK-LABEL: @copy_same_type_tiles
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<64x64xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<64x64xf16>
func.func @copy_same_type_tiles(%source: !loom.tile<64x64xf16>, %target: !loom.tile<64x64xf16>) -> !loom.tile<64x64xf16> {
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0], %[[TARGET]][32, 32], [16, 16] : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  %result = loom.tile.copy %source[0, 0], %target[32, 32], [16, 16]
      : !loom.tile<64x64xf16> -> !loom.tile<64x64xf16>
  return %result : !loom.tile<64x64xf16>
}

// -----

// Tests that 1D copy operation round-trips correctly.
// CHECK-LABEL: @copy_1d
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<128xf32>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<256xf32>
func.func @copy_1d(%source: !loom.tile<128xf32>, %target: !loom.tile<256xf32>) -> !loom.tile<256xf32> {
  // CHECK: loom.tile.copy %[[SOURCE]][0], %[[TARGET]][64], [32] : !loom.tile<128xf32> -> !loom.tile<256xf32>
  %result = loom.tile.copy %source[0], %target[64], [32]
      : !loom.tile<128xf32> -> !loom.tile<256xf32>
  return %result : !loom.tile<256xf32>
}

// -----

// Tests that 3D copy operation round-trips correctly.
// CHECK-LABEL: @copy_3d
// CHECK-SAME: %[[SOURCE:[^:]+]]: !loom.tile<8x8x8xf16>
// CHECK-SAME: %[[TARGET:[^:]+]]: !loom.tile<16x16x16xf16>
func.func @copy_3d(%source: !loom.tile<8x8x8xf16>, %target: !loom.tile<16x16x16xf16>) -> !loom.tile<16x16x16xf16> {
  // CHECK: loom.tile.copy %[[SOURCE]][0, 0, 0], %[[TARGET]][4, 4, 4], [4, 4, 4] : !loom.tile<8x8x8xf16> -> !loom.tile<16x16x16xf16>
  %result = loom.tile.copy %source[0, 0, 0], %target[4, 4, 4], [4, 4, 4]
      : !loom.tile<8x8x8xf16> -> !loom.tile<16x16x16xf16>
  return %result : !loom.tile<16x16x16xf16>
}
