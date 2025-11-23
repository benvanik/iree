// RUN: iree-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error Cases (Verification)
//===----------------------------------------------------------------------===//

func.func @error_too_few_dims(%dim0: index) -> !loom.tile<?x?xf32> {
  // Type has 2 dynamic dimensions but only 1 provided.
  // expected-error @+2 {{ERR_UTIL_SHAPE_0010: result '!loom.tile<?x?xf32>' has 2 dynamic dimensions but 1 were provided}}
  // expected-note @+1 {{Fix: Provide exactly 2 dynamic dimension values for the 'result'}}
  %0 = loom.tile.alloca : !loom.tile<?x?xf32>{%dim0}
  return %0 : !loom.tile<?x?xf32>
}

// -----

func.func @error_dims_on_static_type(%dim0: index) -> !loom.tile<4x4xf32> {
  // Static type should have no dynamic dimension values.
  // expected-error @+2 {{ERR_UTIL_SHAPE_0010: result '!loom.tile<4x4xf32>' has 0 dynamic dimensions but 1 were provided}}
  // expected-note @+1 {{Fix: Provide exactly 0 dynamic dimension values for the 'result'}}
  %0 = loom.tile.alloca : !loom.tile<4x4xf32>{%dim0}
  return %0 : !loom.tile<4x4xf32>
}

// -----

func.func @error_too_many_dims(%dim0: index, %dim1: index) -> !loom.tile<?xf32> {
  // Type has 1 dynamic dimension but 2 provided.
  // expected-error @+2 {{ERR_UTIL_SHAPE_0010: result '!loom.tile<?xf32>' has 1 dynamic dimensions but 2 were provided}}
  // expected-note @+1 {{Fix: Provide exactly 1 dynamic dimension values for the 'result'}}
  %0 = loom.tile.alloca : !loom.tile<?xf32>{%dim0, %dim1}
  return %0 : !loom.tile<?xf32>
}
