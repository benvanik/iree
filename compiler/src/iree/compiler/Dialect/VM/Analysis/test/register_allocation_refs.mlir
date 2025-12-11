// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-register-allocation)))" %s | FileCheck %s

// Tests for ref register MOVE bit computation.
// MOVE bit is indicated by uppercase 'R' (move=true) vs lowercase 'r' (move=false).
// Example: R0 = ref register 0 with move, r0 = ref register 0 without move.

//===----------------------------------------------------------------------===//
// Basic linear flow - simple cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_basic
vm.module @module_basic {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @ref_simple_use
  // Single use of ref - should be MOVE on last use.
  vm.func @ref_simple_use(%buf : !vm.buffer) {
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @ref_multiple_uses
  // Multiple uses - only last should be MOVE.
  vm.func @ref_multiple_uses(%buf : !vm.buffer) {
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @ref_used_in_return
  // Ref used in return - IS MOVE since return is last use in function.
  // (Ownership transfer to caller is handled separately by runtime.)
  vm.func @ref_used_in_return(%buf : !vm.buffer) -> !vm.buffer {
    // CHECK: vm.return
    // CHECK-SAME: operand_registers = ["R0"]
    vm.return %buf : !vm.buffer
  }
}

// -----

//===----------------------------------------------------------------------===//
// Same value used multiple times in same instruction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_same_value
vm.module @module_same_value {

  vm.import private @use_two_buffers(%a : !vm.buffer, %b : !vm.buffer)

  // CHECK-LABEL: @same_ref_twice
  // Same ref used twice in one call - only LAST operand gets MOVE.
  vm.func @same_ref_twice(%buf : !vm.buffer) {
    // CHECK: vm.call @use_two_buffers
    // CHECK-SAME: operand_registers = ["r0", "R0"]
    vm.call @use_two_buffers(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    vm.return
  }

  vm.import private @use_three_buffers(%a : !vm.buffer, %b : !vm.buffer, %c : !vm.buffer)

  // CHECK-LABEL: @same_ref_three_times
  // Same ref used three times - only LAST operand gets MOVE.
  vm.func @same_ref_three_times(%buf : !vm.buffer) {
    // CHECK: vm.call @use_three_buffers
    // CHECK-SAME: operand_registers = ["r0", "r0", "R0"]
    vm.call @use_three_buffers(%buf, %buf, %buf) : (!vm.buffer, !vm.buffer, !vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @same_ref_used_after
  // Same ref twice, but also used after - NO MOVE on any in first call.
  vm.func @same_ref_used_after(%buf : !vm.buffer) {
    // CHECK: vm.call @use_two_buffers
    // First call - not last use, no MOVE.
    // CHECK-SAME: operand_registers = ["r0", "r0"]
    vm.call @use_two_buffers(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    // CHECK: vm.call @use_two_buffers
    // Second call - this IS last use, MOVE on last operand.
    // CHECK-SAME: operand_registers = ["r0", "R0"]
    vm.call @use_two_buffers(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Control flow - branches
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_control_flow
vm.module @module_control_flow {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @diamond_both_paths_use
  // Diamond CFG - same ref passed to both branches.
  // Since same value appears twice in operand list, last occurrence gets MOVE.
  vm.func @diamond_both_paths_use(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^bb1(%buf : !vm.buffer), ^bb2(%buf : !vm.buffer)
  ^bb1(%b1 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%b1) : (!vm.buffer) -> ()
    vm.return
  ^bb2(%b2 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%b2) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @diamond_one_path_uses
  // Diamond - ref only passed to one branch.
  vm.func @diamond_one_path_uses(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // Only one branch gets the ref, so it IS MOVE.
    // CHECK-SAME: operand_registers = ["i0", "R0"]
    vm.cond_br %cond, ^bb1(%buf : !vm.buffer), ^bb2
  ^bb1(%b1 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%b1) : (!vm.buffer) -> ()
    vm.return
  ^bb2:
    vm.return
  }

  // CHECK-LABEL: @loop_with_ref
  // Loop - ref live across back-edge.
  vm.func @loop_with_ref(%count : i32, %buf : !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // Initial branch - buf is passed to loop header, MOVE since not used again.
    // CHECK-SAME: operand_registers = ["i1", "R0"]
    vm.br ^loop(%c0, %buf : i32, !vm.buffer)
  ^loop(%i : i32, %b : !vm.buffer):
    // Use the buffer.
    // CHECK: vm.call @consume_buffer
    // Ref escapes via back-edge, not last use.
    // CHECK-SAME: operand_registers = ["r1"]
    vm.call @consume_buffer(%b) : (!vm.buffer) -> ()
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %count : i32
    // CHECK: vm.cond_br
    // On continue path (back-edge), ref passed to loop header - MOVE transfers ownership.
    // On exit path, ref not passed - will be discarded by MaterializeRefDiscards.
    // CHECK-SAME: operand_registers = ["i1", "i3", "R1"]
    vm.cond_br %cmp, ^loop(%i_next, %b : i32, !vm.buffer), ^exit
  ^exit:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Select operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_select
vm.module @module_select {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @select_ref_both_used
  // Select - both operands are last use.
  vm.func @select_ref_both_used(%cond : i32, %a : !vm.buffer, %b : !vm.buffer) {
    // CHECK: vm.select.ref
    // CHECK-SAME: operand_registers = ["i0", "R0", "R1"]
    %result = vm.select.ref %cond, %a, %b : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R2"]
    vm.call @consume_buffer(%result) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @select_ref_one_reused
  // Select - one operand reused after.
  vm.func @select_ref_one_reused(%cond : i32, %a : !vm.buffer, %b : !vm.buffer) {
    // CHECK: vm.select.ref
    // %a is reused, %b is not.
    // CHECK-SAME: operand_registers = ["i0", "r0", "R1"]
    %result = vm.select.ref %cond, %a, %b : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R2"]
    vm.call @consume_buffer(%result) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%a) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Global refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_globals
vm.module @module_globals {

  vm.global.ref private mutable @global_buf : !vm.buffer

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @store_to_global
  // Storing to global - ref still in scope after store.
  vm.func @store_to_global(%buf : !vm.buffer) {
    // CHECK: vm.global.store.ref
    // CHECK-SAME: operand_registers = ["r0"]
    vm.global.store.ref %buf, @global_buf : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // After store, this IS the last use.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @load_and_use_global
  // Loading from global - creates new ref.
  vm.func @load_and_use_global() {
    %buf = vm.global.load.ref @global_buf : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // Loaded ref, this is last use.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// List operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_lists
vm.module @module_lists {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @list_set_ref
  // Setting ref into list - list takes a copy.
  // List operand does NOT get MOVE (isRefOperandMovable returns false for it).
  // Only value operand (operand 1) gets MOVE.
  vm.func @list_set_ref(%list : !vm.list<!vm.buffer>, %idx : i32, %buf : !vm.buffer) {
    // CHECK: vm.list.set.ref
    // CHECK-SAME: operand_registers = ["r0", "i0", "R1"]
    vm.list.set.ref %list, %idx, %buf : (!vm.list<!vm.buffer>, i32, !vm.buffer)
    vm.return
  }

  // CHECK-LABEL: @list_get_ref
  // Getting ref from list - creates new ref.
  // List operand does NOT get MOVE (isRefOperandMovable returns false for it).
  vm.func @list_get_ref(%list : !vm.list<!vm.buffer>, %idx : i32) {
    // CHECK: vm.list.get.ref
    // CHECK-SAME: operand_registers = ["r0", "i0"]
    %buf = vm.list.get.ref %list, %idx : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_comparisons
vm.module @module_comparisons {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @cmp_eq_ref
  vm.func @cmp_eq_ref(%a : !vm.buffer, %b : !vm.buffer) -> i32 {
    // CHECK: vm.cmp.eq.ref
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    %eq = vm.cmp.eq.ref %a, %b : !vm.buffer
    vm.return %eq : i32
  }

  // CHECK-LABEL: @cmp_eq_ref_reused
  // One ref reused after comparison.
  vm.func @cmp_eq_ref_reused(%a : !vm.buffer, %b : !vm.buffer) -> i32 {
    // CHECK: vm.cmp.eq.ref
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    %eq = vm.cmp.eq.ref %a, %b : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%a) : (!vm.buffer) -> ()
    vm.return %eq : i32
  }

  // CHECK-LABEL: @cmp_nz_ref
  vm.func @cmp_nz_ref(%buf : !vm.buffer) -> i32 {
    // CHECK: vm.cmp.nz.ref
    // CHECK-SAME: operand_registers = ["r0"]
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    vm.return %nz : i32
  }

  // CHECK-LABEL: @cmp_nz_ref_reused
  // Ref reused after null check.
  vm.func @cmp_nz_ref_reused(%buf : !vm.buffer) -> i32 {
    // CHECK: vm.cmp.nz.ref
    // CHECK-SAME: operand_registers = ["r0"]
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    vm.return %nz : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Const ref zero
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_const_ref
vm.module @module_const_ref {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @const_ref_zero
  // const.ref.zero creates a null ref.
  vm.func @const_ref_zero() {
    %null = vm.const.ref.zero : !vm.buffer
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%null) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Mixed refs and primitives
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_mixed
vm.module @module_mixed {

  vm.import private @mixed_args(%i : i32, %buf : !vm.buffer, %j : i32)

  // CHECK-LABEL: @mixed_operands
  // Refs mixed with primitives.
  vm.func @mixed_operands(%i : i32, %buf : !vm.buffer, %j : i32) {
    // CHECK: vm.call @mixed_args
    // Only ref should have MOVE consideration.
    // CHECK-SAME: operand_registers = ["i0", "R0", "i1"]
    vm.call @mixed_args(%i, %buf, %j) : (i32, !vm.buffer, i32) -> ()
    vm.return
  }

  // CHECK-LABEL: @mixed_with_ref_reuse
  // Ref reused after mixed call.
  vm.func @mixed_with_ref_reuse(%i : i32, %buf : !vm.buffer, %j : i32) {
    // CHECK: vm.call @mixed_args
    // Ref is NOT last use.
    // CHECK-SAME: operand_registers = ["i0", "r0", "i1"]
    vm.call @mixed_args(%i, %buf, %j) : (i32, !vm.buffer, i32) -> ()
    // CHECK: vm.call @mixed_args
    // Now ref IS last use.
    // CHECK-SAME: operand_registers = ["i0", "R0", "i1"]
    vm.call @mixed_args(%i, %buf, %j) : (i32, !vm.buffer, i32) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// MOVE bit with discard ops
//===----------------------------------------------------------------------===//

// Tests that when a call is followed by a discard, the call gets MOVE
// (not the discard), since the discard is not a "real" use.

// CHECK-LABEL: @module_move_with_discard
vm.module @module_move_with_discard {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @call_then_discard
  // Call followed by discard - the call should get MOVE since it's the last
  // "real" use. The discard is NOT a real use (just cleanup).
  vm.func @call_then_discard(%buf : !vm.buffer) {
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // Discard should NOT get MOVE - it's not a real use.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @multiple_calls_then_discard
  // Multiple calls then discard - only the last call gets MOVE.
  vm.func @multiple_calls_then_discard(%buf : !vm.buffer) {
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%buf) : (!vm.buffer) -> ()
    // Discard follows - should NOT get MOVE.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  // CHECK-LABEL: @discard_only
  // Discard is the only use - should NOT get MOVE (just release).
  // This tests the case where there's no preceding real use.
  vm.func @discard_only(%buf : !vm.buffer) {
    // No real use before discard - discard is never a "real" last use.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }

  vm.import private @use_two_buffers(%a : !vm.buffer, %b : !vm.buffer)

  // CHECK-LABEL: @partial_discard_coverage
  // Multiple refs in discard - one covered by preceding MOVE, one not.
  vm.func @partial_discard_coverage(%a : !vm.buffer, %b : !vm.buffer) {
    // Only %a is used in call, %b goes straight to discard.
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%a) : (!vm.buffer) -> ()
    // Discard both - %a was MOVE'd above, %b was not used.
    // Neither operand of discard should get MOVE.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    vm.discard.refs %a, %b : !vm.buffer, !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Edge case: Nested loops with refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_nested_loops
vm.module @module_nested_loops {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @nested_loop_outer_ref
  // Outer loop carries a ref, inner loop just iterates.
  // The ref should get MOVE on the outer back-edge but not be affected by inner.
  vm.func @nested_loop_outer_ref(%outer_n : i32, %inner_n : i32, %buf : !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // Initial branch to outer loop - MOVE since not used again in entry.
    // CHECK-SAME: operand_registers = ["i2", "R0"]
    vm.br ^outer(%c0, %buf : i32, !vm.buffer)
  ^outer(%outer_i : i32, %outer_buf : !vm.buffer):
    // Use the buffer in outer loop.
    // CHECK: vm.call @consume_buffer
    // Not last use - used again at outer back-edge.
    // CHECK-SAME: operand_registers = ["r1"]
    vm.call @consume_buffer(%outer_buf) : (!vm.buffer) -> ()
    vm.br ^inner(%c0 : i32)
  ^inner(%inner_i : i32):
    // Inner loop doesn't touch the ref.
    %inner_next = vm.add.i32 %inner_i, %c1 : i32
    %inner_cmp = vm.cmp.lt.i32.s %inner_next, %inner_n : i32
    vm.cond_br %inner_cmp, ^inner(%inner_next : i32), ^outer_check
  ^outer_check:
    %outer_next = vm.add.i32 %outer_i, %c1 : i32
    %outer_cmp = vm.cmp.lt.i32.s %outer_next, %outer_n : i32
    // CHECK: vm.cond_br {{.*}} ^bb1
    // Outer back-edge - ref gets MOVE to transfer ownership to next iteration.
    // CHECK-SAME: operand_registers = ["i4", "i5", "R1"]
    vm.cond_br %outer_cmp, ^outer(%outer_next, %outer_buf : i32, !vm.buffer), ^exit
  ^exit:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Edge case: Ref swap (ping-pong) in loop
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_ping_pong
vm.module @module_ping_pong {

  vm.import private @use_two_buffers(%a : !vm.buffer, %b : !vm.buffer)

  // CHECK-LABEL: @ping_pong_swap
  // Loop that swaps two refs on each iteration.
  // Both refs should get MOVE on the back-edge.
  vm.func @ping_pong_swap(%n : i32, %a : !vm.buffer, %b : !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // Initial branch - both refs MOVE to loop header.
    // CHECK-SAME: operand_registers = ["i1", "R0", "R1"]
    vm.br ^loop(%c0, %a, %b : i32, !vm.buffer, !vm.buffer)
  ^loop(%i : i32, %x : !vm.buffer, %y : !vm.buffer):
    // Use both buffers.
    // CHECK: vm.call @use_two_buffers
    // Neither is last use - both used at back-edge.
    // CHECK-SAME: operand_registers = ["r2", "r3"]
    vm.call @use_two_buffers(%x, %y) : (!vm.buffer, !vm.buffer) -> ()
    %i_next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %i_next, %n : i32
    // CHECK: vm.cond_br
    // Back-edge SWAPS positions: x->y_arg, y->x_arg.
    // Both should get MOVE since they're last uses.
    // CHECK-SAME: operand_registers = ["i1", "i3", "R3", "R2"]
    vm.cond_br %cmp, ^loop(%i_next, %y, %x : i32, !vm.buffer, !vm.buffer), ^exit
  ^exit:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Edge case: Same ref to same block via both edges
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_same_target
vm.module @module_same_target {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @same_ref_both_edges_same_target
  // cond_br where both edges go to same block with same ref.
  // This is like passing the same ref twice - only last operand gets MOVE.
  vm.func @same_ref_both_edges_same_target(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // Same ref to same target via both edges.
    // First occurrence (true branch operand) - NOT MOVE.
    // Second occurrence (false branch operand) - IS MOVE.
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^merge(%buf : !vm.buffer), ^merge(%buf : !vm.buffer)
  ^merge(%b : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%b) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Edge case: Diamond with asymmetric use
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_asymmetric_diamond
vm.module @module_asymmetric_diamond {

  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @diamond_asymmetric_call
  // Diamond where one path calls with ref, other doesn't use it.
  // At the branch, ref goes to both paths, so it's passed twice (same ref).
  vm.func @diamond_asymmetric_call(%cond : i32, %buf : !vm.buffer) {
    // CHECK: vm.cond_br
    // Same ref to both branches.
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^use_path(%buf : !vm.buffer), ^nouse_path(%buf : !vm.buffer)
  ^use_path(%b1 : !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // Last use of %b1 on this path.
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%b1) : (!vm.buffer) -> ()
    vm.br ^merge
  ^nouse_path(%b2 : !vm.buffer):
    // No use of %b2 here - it will be discarded by MaterializeRefDiscards.
    vm.br ^merge
  ^merge:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Scratch register in ref swap pattern (ping-pong)
//===----------------------------------------------------------------------===//

// When swapping refs across a loop back-edge (ping-pong pattern), register
// allocation uses a scratch register (r4) for the cyclic permutation.
// The scratch register must have MOVE semantics (R4) to release its ref
// after the copy, preventing leaks when the branch takes the exit path.

// CHECK-LABEL: @module_scratch_register_swap
vm.module @module_scratch_register_swap {

  // CHECK-LABEL: @ping_pong_swap
  vm.func @ping_pong_swap(%ref_a: !vm.buffer, %ref_b: !vm.buffer, %n: i32) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // CHECK: vm.br ^bb1
    // CHECK-SAME: operand_registers = ["i1", "R0", "R1"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"R1->r3", "R0->r2"{{\]}}{{\]}}
    vm.br ^loop(%c0, %ref_a, %ref_b : i32, !vm.buffer, !vm.buffer)
  ^loop(%i: i32, %x: !vm.buffer, %y: !vm.buffer):
    // CHECK: block_registers = ["i1", "r2", "r3"]
    %cmp_x = vm.cmp.nz.ref %x : !vm.buffer
    %cmp_y = vm.cmp.nz.ref %y : !vm.buffer
    %i_next = vm.add.i32 %i, %c1 : i32
    %continue = vm.cmp.lt.i32.s %i_next, %n : i32
    // CHECK: vm.cond_br
    // The swap creates a cyclic permutation requiring a scratch register (r4).
    // The scratch register source must have MOVE (R4) to release the ref.
    // CHECK-SAME: operand_registers = ["i1", "i3", "R3", "R2"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"r2->r4", "R3->r2", "i3->i1", "R4->r3"{{\]}}, {{\[}}{{\]}}{{\]}}
    vm.cond_br %continue, ^loop(%i_next, %y, %x : i32, !vm.buffer, !vm.buffer), ^exit
  ^exit:
    // CHECK: block_registers = []
    vm.return
  }
}

//===----------------------------------------------------------------------===//
// vm.br_table with refs
//===----------------------------------------------------------------------===//

// Branch table with refs tests the handling of multiple successors where the
// same ref may be forwarded to different targets with different block args.

// CHECK-LABEL: @module_br_table_refs
vm.module @module_br_table_refs {

  // Simple br_table: ref forwarded to all cases.
  // The same ref goes to all three targets.
  // CHECK-LABEL: @br_table_ref_all_cases
  vm.func @br_table_ref_all_cases(%idx: i32, %ref: !vm.buffer) {
    // CHECK: vm.br_table
    // All cases receive the same ref. The last case (case 1) gets MOVE.
    // CHECK: operand_registers = ["i0", "r0", "r0", "R0"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"r0->r1"{{\]}}, {{\[}}"r0->r1"{{\]}}, {{\[}}"R0->r1"{{\]}}{{\]}}
    vm.br_table %idx {
      default: ^bb_default(%ref : !vm.buffer),
      0: ^bb0(%ref : !vm.buffer),
      1: ^bb1(%ref : !vm.buffer)
    }
  ^bb_default(%arg_default: !vm.buffer):
    // CHECK: block_registers = ["r1"]
    vm.return
  ^bb0(%arg0: !vm.buffer):
    // CHECK: block_registers = ["r1"]
    vm.return
  ^bb1(%arg1: !vm.buffer):
    // CHECK: block_registers = ["r1"]
    vm.return
  }

  // br_table: ref forwarded to some cases, not others.
  // Cases without the ref need discards (handled by MaterializeRefDiscards).
  // CHECK-LABEL: @br_table_ref_some_cases
  vm.func @br_table_ref_some_cases(%idx: i32, %ref: !vm.buffer) {
    // CHECK: vm.br_table
    // Ref forwarded to default and case 0, but NOT case 1.
    // Case 0 gets MOVE since case 1 doesn't use the ref.
    // CHECK: operand_registers = ["i0", "r0", "R0"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"r0->r1"{{\]}}, {{\[}}"R0->r1"{{\]}}, {{\[}}{{\]}}{{\]}}
    vm.br_table %idx {
      default: ^bb_default(%ref : !vm.buffer),
      0: ^bb0(%ref : !vm.buffer),
      1: ^bb1
    }
  ^bb_default(%arg_default: !vm.buffer):
    // CHECK: block_registers = ["r1"]
    vm.return
  ^bb0(%arg0: !vm.buffer):
    // CHECK: block_registers = ["r1"]
    vm.return
  ^bb1:
    // No ref here - discard handled by MaterializeRefDiscards.
    // CHECK: block_registers = []
    vm.return
  }

  // br_table: different refs to different cases.
  // CHECK-LABEL: @br_table_different_refs
  vm.func @br_table_different_refs(%idx: i32, %ref_a: !vm.buffer, %ref_b: !vm.buffer) {
    // CHECK: vm.br_table
    // ref_a to default (MOVE since not used in case 0/1)
    // ref_b to case 0 (MOVE since not used in default/case 1)
    // neither to case 1 (both need discards there)
    // CHECK: operand_registers = ["i0", "R0", "R1"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"R0->r2"{{\]}}, {{\[}}"R1->r2"{{\]}}, {{\[}}{{\]}}{{\]}}
    vm.br_table %idx {
      default: ^bb_default(%ref_a : !vm.buffer),
      0: ^bb0(%ref_b : !vm.buffer),
      1: ^bb1
    }
  ^bb_default(%arg_default: !vm.buffer):
    // CHECK: block_registers = ["r2"]
    vm.return
  ^bb0(%arg0: !vm.buffer):
    // CHECK: block_registers = ["r2"]
    vm.return
  ^bb1:
    // Neither ref forwarded here.
    // CHECK: block_registers = []
    vm.return
  }
}

//===----------------------------------------------------------------------===//
// Irreducible CFG with refs
//===----------------------------------------------------------------------===//

// An irreducible CFG has multiple entry points to a loop-like structure.
// This tests that register allocation handles refs correctly when the CFG
// cannot be reduced to a simple loop structure.

// CHECK-LABEL: @module_irreducible_cfg
vm.module @module_irreducible_cfg {

  // Irreducible CFG: ^A and ^B can each reach the other, with external entry
  // to both. The ref must be tracked correctly through all paths.
  //
  //     entry
  //     /   \
  //    v     v
  //   ^A <-> ^B
  //    \     /
  //     v   v
  //     exit
  //
  // CHECK-LABEL: @irreducible_ref_flow
  vm.func @irreducible_ref_flow(%cond1: i32, %cond2: i32, %ref: !vm.buffer) {
    // Entry branches to either A or B, forwarding ref to both.
    // The last edge (to B) gets MOVE since ref is not used after.
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"r0->r1"{{\]}}, {{\[}}"R0->r1"{{\]}}{{\]}}
    vm.cond_br %cond1, ^A(%ref : !vm.buffer), ^B(%ref : !vm.buffer)

  ^A(%ref_a: !vm.buffer):
    // CHECK: block_registers = ["r1"]
    %nz_a = vm.cmp.nz.ref %ref_a : !vm.buffer
    // A can go to B (forwarding ref with MOVE) or exit.
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i1", "R1"]
    vm.cond_br %cond2, ^B(%ref_a : !vm.buffer), ^exit

  ^B(%ref_b: !vm.buffer):
    // CHECK: block_registers = ["r1"]
    %nz_b = vm.cmp.nz.ref %ref_b : !vm.buffer
    // B can go to A (forwarding ref with MOVE) or exit.
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i1", "R1"]
    vm.cond_br %cond2, ^A(%ref_b : !vm.buffer), ^exit

  ^exit:
    // Ref dies on edges to exit - handled by MaterializeRefDiscards.
    // CHECK: block_registers = []
    vm.return
  }

  // More complex irreducible: two refs with different flow patterns.
  // ref_x goes to A, ref_y goes to B, then they can swap between A/B.
  // CHECK-LABEL: @irreducible_two_refs
  vm.func @irreducible_two_refs(%cond1: i32, %cond2: i32,
                                 %ref_x: !vm.buffer, %ref_y: !vm.buffer) {
    // CHECK: vm.cond_br
    // Both refs get MOVE since each is only used on one branch.
    // CHECK-SAME: operand_registers = ["i0", "R0", "R1"]
    // CHECK-SAME: remap_registers = {{\[}}{{\[}}"R0->r2"{{\]}}, {{\[}}"R1->r2"{{\]}}{{\]}}
    vm.cond_br %cond1, ^A(%ref_x : !vm.buffer), ^B(%ref_y : !vm.buffer)

  ^A(%a_ref: !vm.buffer):
    // CHECK: block_registers = ["r2"]
    %nz_a = vm.cmp.nz.ref %a_ref : !vm.buffer
    // A forwards to B with MOVE.
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i1", "R2"]
    vm.cond_br %cond2, ^B(%a_ref : !vm.buffer), ^exit

  ^B(%b_ref: !vm.buffer):
    // CHECK: block_registers = ["r2"]
    %nz_b = vm.cmp.nz.ref %b_ref : !vm.buffer
    // B forwards to A with MOVE.
    // CHECK: vm.cond_br
    // CHECK-SAME: operand_registers = ["i1", "R2"]
    vm.cond_br %cond2, ^A(%b_ref : !vm.buffer), ^exit

  ^exit:
    // CHECK: block_registers = []
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Branch with discard in alternate path
//===----------------------------------------------------------------------===//

// When a ref is passed as a branch argument to one successor, and the other
// successor has a discard op for that ref, the branch argument should still
// get MOVE. The discard is not a "real" use - it's just cleanup.

// CHECK-LABEL: @module_branch_discard_simple
vm.module @module_branch_discard_simple {

  vm.import private @produce_buffer() -> !vm.buffer
  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @branch_ref_with_discard_in_other_path
  // cond_br where ref goes to success path as branch arg, and failure path
  // has a discard for the ref. The branch arg should get MOVE since the
  // discard is not a real use.
  vm.func @branch_ref_with_discard_in_other_path() {
    %buf = vm.call @produce_buffer() : () -> !vm.buffer
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    // CHECK: vm.cond_br
    // The branch argument should get MOVE (R0) since the discard in ^failure
    // is not a "real" use of the ref.
    // CHECK-SAME: operand_registers = ["i0", "R0"]
    vm.cond_br %nz, ^success(%buf : !vm.buffer), ^failure
  ^success(%b: !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume_buffer(%b) : (!vm.buffer) -> ()
    vm.return
  ^failure:
    // The discard here is just cleanup, not a real use.
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
}

// -----

// CHECK-LABEL: @module_branch_discard_multiple
vm.module @module_branch_discard_multiple {

  vm.import private @produce_buffer() -> !vm.buffer
  vm.import private @consume_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @branch_ref_with_multiple_discards
  // More complex case: ref used in success path, multiple failure paths all
  // have discards. The branch arg should still get MOVE.
  vm.func @branch_ref_with_multiple_discards(%cond: i32) {
    %buf = vm.call @produce_buffer() : () -> !vm.buffer
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    // CHECK: vm.cond_br
    // Branch arg should get MOVE.
    // CHECK-SAME: operand_registers = ["i1", "R0"]
    vm.cond_br %nz, ^success(%buf : !vm.buffer), ^check_cond2
  ^success(%b: !vm.buffer):
    // CHECK: vm.call @consume_buffer
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume_buffer(%b) : (!vm.buffer) -> ()
    vm.return
  ^check_cond2:
    // Another branch with discard on both paths.
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^fail1, ^fail2
  ^fail1:
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  ^fail2:
    // CHECK: vm.discard.refs
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Regression test: Discard-register collision with block arguments
//===----------------------------------------------------------------------===//

// This test verifies that when a discard and a block argument are at the same
// block (same instruction position), they get different registers.
//
// The bug occurs when:
// 1. A value (%outer) ends in a predecessor block
// 2. The successor block has a block arg (%inner) AND a discard for %outer
// 3. Both %outer's discard and %inner's definition are at the same position
// 4. Register allocation reuses %outer's register for %inner
// 5. The discard kills %inner instead of %outer!
//
// FIX: Extend %outer's interval to include the discard, preventing reuse.

// CHECK-LABEL: @discard_block_arg_collision
vm.module @my_module {
  vm.func @discard_block_arg_collision(%cond: i32) {
    // CHECK: %[[OUTER:.+]] = vm.const.ref.rodata @data
    // CHECK-SAME: result_registers = ["r0"]
    %outer = vm.const.ref.rodata @data : !vm.buffer
    // CHECK: vm.call @use_outer(%[[OUTER]])
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @use_outer(%outer) : (!vm.buffer) -> ()
    // CHECK: %[[INNER_SRC:.+]] = vm.call @create
    // CHECK-SAME: result_registers = ["r1"]
    %inner_src = vm.call @create() : () -> !vm.buffer
    // CHECK: vm.br ^{{.+}}(%[[INNER_SRC]] : !vm.buffer)
    // CHECK-SAME: operand_registers = ["R1"]
    vm.br ^bb1(%inner_src : !vm.buffer)

  // CHECK: ^{{.+}}(%[[INNER:.+]]: !vm.buffer):
  ^bb1(%inner: !vm.buffer):
    // Discard targets %outer (r0), NOT %inner (r1)
    // block_registers shows block arg is r1 (different from discard's r0)
    // CHECK: vm.discard.refs %[[OUTER]]
    // CHECK-SAME: block_registers = ["r1"]
    // CHECK-SAME: operand_registers = ["r0"]
    vm.discard.refs %outer : !vm.buffer
    // %inner is still alive in r1
    // CHECK: vm.call @use_inner(%[[INNER]])
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @use_inner(%inner) : (!vm.buffer) -> ()
    vm.return
  }

  vm.rodata private @data dense<[1, 2, 3]> : tensor<3xi8>
  vm.import private @use_outer(%buf: !vm.buffer)
  vm.import private @use_inner(%buf: !vm.buffer)
  vm.import private @create() -> !vm.buffer
}
