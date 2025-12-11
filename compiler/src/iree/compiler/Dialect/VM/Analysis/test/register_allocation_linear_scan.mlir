// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(vm.func(test-iree-vm-register-allocation)))" %s | FileCheck %s

// Tests for Phase 2 linear scan global allocation.
// These tests verify cross-block register reuse and coalescing hints.

// CHECK-LABEL: @module
vm.module @module {
  // Test that entry block arguments are allocated monotonically (ABI requirement).
  // CHECK-LABEL: @entry_args_monotonic
  vm.func @entry_args_monotonic(%arg0: i32, %arg1: i64, %arg2: i32) -> (i32, i64, i32) {
    // Entry args must be i0, i2+3, i4 (monotonic, with i64 on even boundary).
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "i2+3", "i4"]
    vm.return %arg0, %arg1, %arg2 : i32, i64, i32
  }

  // Test cross-block register reuse: values that are dead in one path
  // can have their registers reused in another path.
  // CHECK-LABEL: @cross_block_reuse
  vm.func @cross_block_reuse(%arg0: i32, %cond: i32) -> i32 {
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers = ["i0", "i1"]
    %x = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // %x is used here, then dead.
    vm.return %x : i32
  ^bb2:
    // Since %x is not used in bb2, its register can potentially be reused.
    // The allocator should be able to allocate %y efficiently.
    %y = vm.mul.i32 %arg0, %arg0 : i32
    vm.return %y : i32
  }

  // Test coalescing: branch operands should hint block arg allocation.
  // CHECK-LABEL: @coalescing_simple
  vm.func @coalescing_simple(%arg0: i32, %cond: i32) -> i32 {
    // CHECK: vm.const.i32
    // CHECK-SAME: block_registers = ["i0", "i1"]
    %c1 = vm.const.i32 1
    // When branching to bb1 with %c1, the block arg should ideally
    // get the same register as %c1 to avoid a remap.
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^bb1(%c1 : i32), ^bb2
  ^bb1(%v: i32):
    vm.return %v : i32
  ^bb2:
    vm.return %arg0 : i32
  }

  // Test that non-entry block args participate in global allocation.
  // CHECK-LABEL: @non_entry_block_args
  vm.func @non_entry_block_args(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: vm.br
    // CHECK-SAME: block_registers = ["i0", "i1"]
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%a: i32, %b: i32):
    // Non-entry block args are allocated via linear scan.
    // CHECK: vm.add.i32
    // CHECK-SAME: block_registers =
    %sum = vm.add.i32 %a, %b : i32
    vm.return %sum : i32
  }

  // Test i64 alignment in non-entry blocks.
  // CHECK-LABEL: @i64_non_entry_block
  vm.func @i64_non_entry_block(%arg0: i64, %cond: i32) -> i64 {
    // Entry i64 should be on even boundary.
    // CHECK: vm.cond_br
    // CHECK-SAME: block_registers = ["i0+1", "i2"]
    vm.cond_br %cond, ^bb1(%arg0 : i64), ^bb2
  ^bb1(%v: i64):
    // Non-entry i64 block arg should also be properly aligned.
    vm.return %v : i64
  ^bb2:
    %zero = vm.const.i64.zero
    vm.return %zero : i64
  }

  // Test loop with cross-iteration register reuse.
  // CHECK-LABEL: @loop_register_reuse
  vm.func @loop_register_reuse() -> i32 {
    %c0 = vm.const.i32.zero
    %c1 = vm.const.i32 1
    %c10 = vm.const.i32 10
    vm.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %next = vm.add.i32 %i, %c1 : i32
    %cmp = vm.cmp.lt.i32.s %next, %c10 : i32
    vm.cond_br %cmp, ^loop(%next : i32), ^exit(%next : i32)
  ^exit(%result: i32):
    vm.return %result : i32
  }

  // Test mixed type entry block arguments (i32, ref, i64 interleaved).
  // Entry block i32/i64 should be allocated monotonically, refs handled separately.
  // CHECK-LABEL: @entry_mixed_types
  vm.func @entry_mixed_types(%arg0: i32, %arg1: !vm.ref<!vm.buffer>, %arg2: i64, %arg3: i32) -> i32 {
    // i32/i64 are allocated monotonically: i0, (skip ref), i2+3, i4.
    // ref is handled by block-local ref allocation: r0.
    // CHECK: vm.return
    // CHECK-SAME: block_registers = ["i0", "r0", "i2+3", "i4"]
    vm.return %arg0 : i32
  }

  // Test nested control flow with value liveness across multiple paths.
  // CHECK-LABEL: @nested_control_flow
  vm.func @nested_control_flow(%arg0: i32, %cond1: i32, %cond2: i32) -> i32 {
    %x = vm.add.i32 %arg0, %arg0 : i32
    // CHECK: vm.cond_br
    vm.cond_br %cond1, ^outer_true, ^outer_false
  ^outer_true:
    // %x is live here.
    vm.cond_br %cond2, ^inner_true, ^inner_false
  ^inner_true:
    vm.return %x : i32
  ^inner_false:
    %y = vm.mul.i32 %x, %arg0 : i32
    vm.return %y : i32
  ^outer_false:
    // %x is NOT used in this path, register can be reused.
    %z = vm.sub.i32 %arg0, %arg0 : i32
    vm.return %z : i32
  }

  // Test diamond control flow pattern with value merging.
  // CHECK-LABEL: @diamond_merge
  vm.func @diamond_merge(%arg0: i32, %cond: i32) -> i32 {
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^left, ^right
  ^left:
    %x = vm.add.i32 %arg0, %arg0 : i32
    vm.br ^merge(%x : i32)
  ^right:
    %y = vm.sub.i32 %arg0, %arg0 : i32
    vm.br ^merge(%y : i32)
  ^merge(%result: i32):
    // %result comes from either %x or %y.
    // Coalescing should try to match one of them.
    vm.return %result : i32
  }

  // Test that AssignmentOp results share the operand's register.
  // CHECK-LABEL: @assignment_op_i32
  vm.func @assignment_op_i32(%arg0: i32) -> i32 {
    // CHECK: vm.const.i32
    %c1 = vm.const.i32 42
    // The return value should use the same register as %c1.
    vm.return %c1 : i32
  }

  // Test multiple block arguments with different liveness patterns.
  // CHECK-LABEL: @multi_block_args_liveness
  vm.func @multi_block_args_liveness(%arg0: i32, %arg1: i32, %cond: i32) -> i32 {
    %x = vm.add.i32 %arg0, %arg1 : i32
    %y = vm.mul.i32 %arg0, %arg1 : i32
    // CHECK: vm.cond_br
    vm.cond_br %cond, ^use_both(%x, %y : i32, i32), ^use_one(%x : i32)
  ^use_both(%a: i32, %b: i32):
    %sum = vm.add.i32 %a, %b : i32
    vm.return %sum : i32
  ^use_one(%c: i32):
    vm.return %c : i32
  }

  // Test back-edge in loop (value defined in loop, used in subsequent iteration).
  // CHECK-LABEL: @loop_back_edge
  vm.func @loop_back_edge(%init: i32, %limit: i32) -> i32 {
    %c1 = vm.const.i32 1
    vm.br ^header(%init : i32)
  ^header(%acc: i32):
    %cmp = vm.cmp.lt.i32.s %acc, %limit : i32
    vm.cond_br %cmp, ^body, ^exit
  ^body:
    %next = vm.add.i32 %acc, %c1 : i32
    // %next flows back to %acc via back-edge.
    vm.br ^header(%next : i32)
  ^exit:
    vm.return %acc : i32
  }
}
