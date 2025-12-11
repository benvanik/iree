// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-register-allocation)))" %s | FileCheck %s

// Tests for Phase 3: Global ref allocation with correct MOVE bit handling.
// MOVE bit is indicated by uppercase 'R' (move=true) vs lowercase 'r' (move=false).
// Example: R0 = ref register 0 with move, r0 = ref register 0 without move.

//===----------------------------------------------------------------------===//
// Entry block ref args - monotonic allocation (ABI requirement)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_entry_args
vm.module @module_entry_args {

  vm.import private @consume3(%a : !vm.buffer, %b : !vm.buffer, %c : !vm.buffer)
  vm.import private @consume_mixed(%i0 : i32, %r0 : !vm.buffer, %i1 : i32, %r1 : !vm.buffer)

  // CHECK-LABEL: @entry_ref_args_monotonic
  // Entry block ref arguments should be allocated monotonically for ABI stability.
  vm.func @entry_ref_args_monotonic(%r0: !vm.buffer, %r1: !vm.buffer, %r2: !vm.buffer) {
    // CHECK: block_registers = ["r0", "r1", "r2"]
    vm.call @consume3(%r0, %r1, %r2) : (!vm.buffer, !vm.buffer, !vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @entry_mixed_types_monotonic
  // Mixed i32 and ref args should each be allocated monotonically in their banks.
  vm.func @entry_mixed_types_monotonic(%i0: i32, %r0: !vm.buffer, %i1: i32, %r1: !vm.buffer) {
    // CHECK: block_registers = ["i0", "r0", "i1", "r1"]
    vm.call @consume_mixed(%i0, %r0, %i1, %r1) : (i32, !vm.buffer, i32, !vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Cross-block ref register reuse
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_cross_block_reuse
vm.module @module_cross_block_reuse {

  vm.import private @consume(%buf : !vm.buffer)
  vm.import private @produce() -> !vm.buffer

  // CHECK-LABEL: @ref_dead_in_one_path
  // Ref dead in one path should allow register reuse in that path.
  vm.func @ref_dead_in_one_path(%cond: i32, %buf: !vm.buffer) {
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^use(%buf : !vm.buffer), ^skip
  ^use(%b: !vm.buffer):
    // CHECK: vm.call @consume
    // Block arg coalesces with branch operand (r0), MOVE on last use.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%b) : (!vm.buffer) -> ()
    vm.br ^exit
  ^skip:
    // %buf is dead here, so registers can be reused.
    %new_buf = vm.call @produce() : () -> !vm.buffer
    // CHECK: vm.call @consume
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%new_buf) : (!vm.buffer) -> ()
    vm.br ^exit
  ^exit:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Loop with ref - MOVE bit correctness
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_loop_refs
vm.module @module_loop_refs {

  vm.import private @use_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @loop_with_ref_no_move_in_body
  // Ref passed to loop - call inside loop should NOT have MOVE (ref still used in branch).
  // The branch operand DOES get MOVE because it's the last use of the SSA value.
  vm.func @loop_with_ref_no_move_in_body(%count: i32, %buf: !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    // Initial branch to loop - buf is MOVE since it's consumed by the loop.
    // CHECK: vm.br ^bb1
    // CHECK-SAME: operand_registers = ["i1", "R0"]
    vm.br ^loop(%c0, %buf : i32, !vm.buffer)
  ^loop(%i: i32, %b: !vm.buffer):
    // CHECK: vm.call @use_buffer
    // Critical: lowercase "r" - NOT MOVE because ref is used again in the branch.
    // Block args are [i1, r0] - the ref coalesces to r0.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @use_buffer(%b) : (!vm.buffer) -> ()
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %count : i32
    // CHECK: vm.cond_br
    // The ref operand IS MOVE on back-edge (ownership transfers to next iteration).
    // On exit: ref is discarded by MaterializeRefDiscards.
    // CHECK-SAME: operand_registers = ["i3", "i1", "R0"]
    vm.cond_br %cmp, ^loop(%next, %b : i32, !vm.buffer), ^exit
  ^exit:
    vm.return
  }

  // CHECK-LABEL: @loop_ref_used_after_exit
  // Ref used after loop exit - should not be moved until final use.
  vm.func @loop_ref_used_after_exit(%count: i32, %buf: !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    vm.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    // CHECK: vm.call @use_buffer
    // Not MOVE - ref is used after loop.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %count : i32
    vm.cond_br %cmp, ^loop(%next : i32), ^exit
  ^exit:
    // CHECK: vm.call @use_buffer
    // Final use - MOVE.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Same ref in multiple operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_multiple_operands
vm.module @module_multiple_operands {

  vm.import private @use_two(%a : !vm.buffer, %b : !vm.buffer)
  vm.import private @use_three(%a : !vm.buffer, %b : !vm.buffer, %c : !vm.buffer)

  // CHECK-LABEL: @same_ref_twice_last_gets_move
  // Same ref twice in one call - only LAST operand gets MOVE.
  vm.func @same_ref_twice_last_gets_move(%buf: !vm.buffer) {
    // CHECK: vm.call @use_two
    // CHECK-SAME: operand_registers = ["r0", "R0"]
    vm.call @use_two(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @same_ref_three_times
  // Same ref three times - only LAST operand gets MOVE.
  vm.func @same_ref_three_times(%buf: !vm.buffer) {
    // CHECK: vm.call @use_three
    // CHECK-SAME: operand_registers = ["r0", "r0", "R0"]
    vm.call @use_three(%buf, %buf, %buf) : (!vm.buffer, !vm.buffer, !vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @same_ref_multiple_calls
  // Same ref in multiple calls - only last call's last occurrence gets MOVE.
  vm.func @same_ref_multiple_calls(%buf: !vm.buffer) {
    // CHECK: vm.call @use_two
    // Not last use of %buf.
    // CHECK-SAME: operand_registers = ["r0", "r0"]
    vm.call @use_two(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    // CHECK: vm.call @use_two
    // Last use of %buf, last operand gets MOVE.
    // CHECK-SAME: operand_registers = ["r0", "R0"]
    vm.call @use_two(%buf, %buf) : (!vm.buffer, !vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Diamond CFG with refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_diamond
vm.module @module_diamond {

  vm.import private @consume(%buf : !vm.buffer)

  // CHECK-LABEL: @diamond_same_ref_both_paths
  // Same ref passed to both branches - last operand in branch gets MOVE.
  vm.func @diamond_same_ref_both_paths(%cond: i32, %buf: !vm.buffer) {
    // CHECK: vm.cond_br
    // Same ref twice in operand list - last gets MOVE.
    // CHECK-SAME: operand_registers = ["i0", "r0", "R0"]
    vm.cond_br %cond, ^left(%buf : !vm.buffer), ^right(%buf : !vm.buffer)
  ^left(%b1: !vm.buffer):
    // CHECK: vm.call @consume
    // Block arg coalesces to r0.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%b1) : (!vm.buffer) -> ()
    vm.return
  ^right(%b2: !vm.buffer):
    // CHECK: vm.call @consume
    // Block arg coalesces to r0.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%b2) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @diamond_ref_one_path_only
  // Ref passed to only one branch.
  vm.func @diamond_ref_one_path_only(%cond: i32, %buf: !vm.buffer) {
    // CHECK: vm.cond_br
    // Only one occurrence - gets MOVE.
    // CHECK-SAME: operand_registers = ["i0", "R0"]
    vm.cond_br %cond, ^use(%buf : !vm.buffer), ^skip
  ^use(%b: !vm.buffer):
    // CHECK: vm.call @consume
    // Block arg coalesces to r0.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%b) : (!vm.buffer) -> ()
    vm.return
  ^skip:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Global store then local use
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_global_refs
vm.module @module_global_refs {

  vm.global.ref private mutable @global_buf : !vm.buffer
  vm.import private @consume(%buf : !vm.buffer)

  // CHECK-LABEL: @global_store_then_local_use
  // Storing to global does NOT consume - local use after is still valid.
  vm.func @global_store_then_local_use(%buf: !vm.buffer) {
    // CHECK: vm.global.store.ref
    // Store retains (not MOVE).
    // CHECK-SAME: operand_registers = ["r0"]
    vm.global.store.ref %buf, @global_buf : !vm.buffer
    // CHECK: vm.call @consume
    // Local use after store - IS last use, gets MOVE.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @global_store_only
  // Only store to global - store is last use, but store retains.
  vm.func @global_store_only(%buf: !vm.buffer) {
    // CHECK: vm.global.store.ref
    // Store is last use but retains anyway (semantics of store).
    // Actually, store's operand IS the last use, so it should get MOVE.
    // The store operation handles the retain internally.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.global.store.ref %buf, @global_buf : !vm.buffer
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Ref coalescing at branches
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_coalescing
vm.module @module_coalescing {

  vm.import private @consume(%buf : !vm.buffer)

  // CHECK-LABEL: @branch_coalescing_simple
  // Branch operand should coalesce with block arg to avoid remap.
  vm.func @branch_coalescing_simple(%buf: !vm.buffer, %cond: i32) {
    // CHECK: vm.cond_br
    // CHECK-SAME: remap_registers = [
    vm.cond_br %cond, ^use(%buf : !vm.buffer), ^exit
  ^use(%arg: !vm.buffer):
    // CHECK: vm.call @consume
    vm.call @consume(%arg) : (!vm.buffer) -> ()
    vm.return
  ^exit:
    vm.return
  }

  // CHECK-LABEL: @branch_no_coalesce_swap
  // Branch swaps ref order - requires remap (cannot coalesce).
  vm.func @branch_no_coalesce_swap(%a: !vm.buffer, %b: !vm.buffer) {
    // CHECK: vm.br
    // Swap requires remap with scratch register.
    // CHECK-SAME: remap_registers = [
    vm.br ^use(%b, %a : !vm.buffer, !vm.buffer)
  ^use(%x: !vm.buffer, %y: !vm.buffer):
    vm.call @consume(%x) : (!vm.buffer) -> ()
    vm.call @consume(%y) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Select operations with refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_select
vm.module @module_select {

  vm.import private @consume(%buf : !vm.buffer)

  // CHECK-LABEL: @select_both_refs_consumed
  // Select with both refs as last use.
  vm.func @select_both_refs_consumed(%cond: i32, %a: !vm.buffer, %b: !vm.buffer) {
    // CHECK: vm.select.ref
    // Both operands are last use - both get MOVE.
    // Result coalesces with first operand (r0).
    // CHECK-SAME: operand_registers = ["i0", "R0", "R1"]
    // CHECK-SAME: result_registers = ["r0"]
    %result = vm.select.ref %cond, %a, %b : !vm.buffer
    // CHECK: vm.call @consume
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%result) : (!vm.buffer) -> ()
    vm.return
  }

  // CHECK-LABEL: @select_one_ref_reused
  // Select where one operand is reused after.
  vm.func @select_one_ref_reused(%cond: i32, %a: !vm.buffer, %b: !vm.buffer) {
    // CHECK: vm.select.ref
    // %a is reused (r0), %b is consumed (R1).
    // Result goes to r1 (coalesces with %b since %a is still live).
    // CHECK-SAME: operand_registers = ["i0", "r0", "R1"]
    // CHECK-SAME: result_registers = ["r1"]
    %result = vm.select.ref %cond, %a, %b : !vm.buffer
    // CHECK: vm.call @consume(%buffer)
    // CHECK-SAME: operand_registers = ["R1"]
    vm.call @consume(%result) : (!vm.buffer) -> ()
    // CHECK: vm.call @consume(%arg1)
    // %a is last use here.
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%a) : (!vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Comparison operations with refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_compare
vm.module @module_compare {

  vm.import private @consume(%buf : !vm.buffer)

  // CHECK-LABEL: @cmp_eq_both_consumed
  vm.func @cmp_eq_both_consumed(%a: !vm.buffer, %b: !vm.buffer) -> i32 {
    // CHECK: vm.cmp.eq.ref
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    %eq = vm.cmp.eq.ref %a, %b : !vm.buffer
    vm.return %eq : i32
  }

  // CHECK-LABEL: @cmp_eq_one_reused
  // One ref reused after comparison.
  vm.func @cmp_eq_one_reused(%a: !vm.buffer, %b: !vm.buffer) -> i32 {
    // CHECK: vm.cmp.eq.ref
    // CHECK-SAME: operand_registers = ["r0", "r1"]
    %eq = vm.cmp.eq.ref %a, %b : !vm.buffer
    // CHECK: vm.call @consume
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%a) : (!vm.buffer) -> ()
    vm.return %eq : i32
  }

  // CHECK-LABEL: @cmp_nz_ref
  vm.func @cmp_nz_ref(%buf: !vm.buffer) -> i32 {
    // CHECK: vm.cmp.nz.ref
    // CHECK-SAME: operand_registers = ["r0"]
    %nz = vm.cmp.nz.ref %buf : !vm.buffer
    vm.return %nz : i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Nested loops with refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_nested_loops
vm.module @module_nested_loops {

  vm.import private @use_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @nested_loop_ref_carried
  // Ref carried through nested loops - should never get MOVE inside loops.
  vm.func @nested_loop_ref_carried(%outer_count: i32, %inner_count: i32, %buf: !vm.buffer) {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    vm.br ^outer_header(%c0, %buf : i32, !vm.buffer)
  ^outer_header(%i: i32, %b_outer: !vm.buffer):
    %outer_cmp = vm.cmp.lt.i32.s %i, %outer_count : i32
    vm.cond_br %outer_cmp, ^inner_init(%i, %b_outer : i32, !vm.buffer), ^exit
  ^inner_init(%i2: i32, %b_inner_init: !vm.buffer):
    vm.br ^inner_header(%c0, %b_inner_init : i32, !vm.buffer)
  ^inner_header(%j: i32, %b_inner: !vm.buffer):
    %inner_cmp = vm.cmp.lt.i32.s %j, %inner_count : i32
    vm.cond_br %inner_cmp, ^inner_body(%j, %b_inner : i32, !vm.buffer), ^outer_latch(%i2, %b_inner : i32, !vm.buffer)
  ^inner_body(%j2: i32, %b_body: !vm.buffer):
    // CHECK: vm.call @use_buffer
    // Should be "r" - ref escapes via multiple back-edges.
    // Block arg coalesces to r0.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @use_buffer(%b_body) : (!vm.buffer) -> ()
    %j_next = vm.add.i32 %j2, %c1 : i32
    vm.br ^inner_header(%j_next, %b_body : i32, !vm.buffer)
  ^outer_latch(%i3: i32, %b_latch: !vm.buffer):
    %i_next = vm.add.i32 %i3, %c1 : i32
    vm.br ^outer_header(%i_next, %b_latch : i32, !vm.buffer)
  ^exit:
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// High register pressure
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_pressure
vm.module @module_pressure {

  vm.import private @produce() -> !vm.buffer
  vm.import private @use_five(%a : !vm.buffer, %b : !vm.buffer, %c : !vm.buffer, %d : !vm.buffer, %e : !vm.buffer)

  // CHECK-LABEL: @many_concurrent_refs
  // Many refs live concurrently - should allocate separate registers.
  vm.func @many_concurrent_refs() {
    %r0 = vm.call @produce() : () -> !vm.buffer
    %r1 = vm.call @produce() : () -> !vm.buffer
    %r2 = vm.call @produce() : () -> !vm.buffer
    %r3 = vm.call @produce() : () -> !vm.buffer
    %r4 = vm.call @produce() : () -> !vm.buffer
    // CHECK: vm.call @use_five
    // All are last use - all get MOVE.
    // CHECK-SAME: operand_registers = ["R0", "R1", "R2", "R3", "R4"]
    vm.call @use_five(%r0, %r1, %r2, %r3, %r4) : (!vm.buffer, !vm.buffer, !vm.buffer, !vm.buffer, !vm.buffer) -> ()
    vm.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Return values
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_return
vm.module @module_return {

  // CHECK-LABEL: @return_ref_is_move
  // Returning a ref - should be MOVE since it's the last use in function.
  vm.func @return_ref_is_move(%buf: !vm.buffer) -> !vm.buffer {
    // CHECK: vm.return
    // CHECK-SAME: operand_registers = ["R0"]
    vm.return %buf : !vm.buffer
  }

  vm.import private @use_buffer(%buf : !vm.buffer)

  // CHECK-LABEL: @use_then_return
  // Use then return - only return gets MOVE.
  vm.func @use_then_return(%buf: !vm.buffer) -> !vm.buffer {
    // CHECK: vm.call @use_buffer
    // Not MOVE - ref is returned after.
    // CHECK-SAME: operand_registers = ["r0"]
    vm.call @use_buffer(%buf) : (!vm.buffer) -> ()
    // CHECK: vm.return
    // CHECK-SAME: operand_registers = ["R0"]
    vm.return %buf : !vm.buffer
  }
}

// -----

//===----------------------------------------------------------------------===//
// List operations with refs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @module_lists
vm.module @module_lists {

  vm.import private @consume(%buf : !vm.buffer)

  // CHECK-LABEL: @list_set_ref_consumed
  // Setting ref into list - list operand does NOT get MOVE (isRefOperandMovable
  // returns false for it), only value operand (operand 1) gets MOVE.
  vm.func @list_set_ref_consumed(%list: !vm.list<!vm.buffer>, %idx: i32, %buf: !vm.buffer) {
    // CHECK: vm.list.set.ref
    // CHECK-SAME: operand_registers = ["r0", "i0", "R1"]
    vm.list.set.ref %list, %idx, %buf : (!vm.list<!vm.buffer>, i32, !vm.buffer)
    vm.return
  }

  // CHECK-LABEL: @list_get_ref
  // Getting ref from list - list operand does NOT get MOVE (isRefOperandMovable
  // returns false for it). Result coalesces with list operand (both die at same time).
  vm.func @list_get_ref(%list: !vm.list<!vm.buffer>, %idx: i32) {
    // CHECK: vm.list.get.ref
    // CHECK-SAME: operand_registers = ["r0", "i0"]
    // CHECK-SAME: result_registers = ["r0"]
    %buf = vm.list.get.ref %list, %idx : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
    // CHECK: vm.call @consume
    // CHECK-SAME: operand_registers = ["R0"]
    vm.call @consume(%buf) : (!vm.buffer) -> ()
    vm.return
  }
}
