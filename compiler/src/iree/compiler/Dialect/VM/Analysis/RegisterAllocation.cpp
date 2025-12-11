// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"

#include <algorithm>
#include <map>
#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Analysis/LinearScan/LiveIntervals.h"
#include "iree/compiler/Dialect/VM/Analysis/LinearScan/RegisterBank.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"

namespace mlir::iree_compiler {

static Attribute getStrArrayAttr(Builder &builder,
                                 ArrayRef<std::string> values) {
  return builder.getStrArrayAttr(llvm::map_to_vector<8>(
      values, [](const std::string &value) { return StringRef(value); }));
}

// static
LogicalResult RegisterAllocation::annotateIR(IREE::VM::FuncOp funcOp) {
  RegisterAllocation registerAllocation;
  if (failed(registerAllocation.recalculate(funcOp))) {
    funcOp.emitOpError() << "failed to allocate registers for function";
    return failure();
  }

  Builder builder(funcOp.getContext());
  for (auto &block : funcOp.getBlocks()) {
    SmallVector<std::string, 8> blockRegStrs;
    blockRegStrs.reserve(block.getNumArguments());
    for (auto blockArg : block.getArguments()) {
      auto reg = registerAllocation.map_[blockArg];
      blockRegStrs.push_back(reg.toString());
    }
    block.front().setAttr("block_registers",
                          getStrArrayAttr(builder, blockRegStrs));

    for (auto &op : block.getOperations()) {
      // Emit operand registers with MOVE bits for ref operands.
      if (op.getNumOperands() > 0) {
        SmallVector<std::string, 8> operandRegStrs;
        operandRegStrs.reserve(op.getNumOperands());

        // For branch operations, use remapSuccessorRegisters to get the
        // correct MOVE bits for branch arguments.
        // Build a map from operand index to the correct register with MOVE bit.
        llvm::DenseMap<unsigned, Register> branchOperandRegs;
        if (auto branchOp = dyn_cast<BranchOpInterface>(&op)) {
          for (unsigned i = 0; i < op.getNumSuccessors(); ++i) {
            auto srcDstRegs =
                registerAllocation.remapSuccessorRegisters(&op, i);
            auto succOperands =
                branchOp.getSuccessorOperands(i).getForwardedOperands();
            if (succOperands.empty())
              continue;
            unsigned baseIdx = succOperands.getBeginOperandIndex();
            // remapSuccessorRegisters only returns pairs where src != dst.
            // For display, we need ALL operands with correct MOVE bits.
            // Re-compute with the same logic as remapSuccessorRegisters.
            (void)srcDstRegs; // Unused - we recompute for display.

            // Build lastOccurrence map: only the last occurrence of a ref value
            // in this successor's operand list can get MOVE (matches
            // remapSuccessorRegisters logic).
            llvm::DenseMap<Value, unsigned> lastOccurrence;
            for (auto [idx, operand] : llvm::enumerate(succOperands)) {
              if (isa<IREE::VM::RefType>(operand.getType())) {
                lastOccurrence[operand] = idx;
              }
            }

            for (auto [localIdx, operand] : llvm::enumerate(succOperands)) {
              unsigned globalIdx = baseIdx + localIdx;
              auto srcReg = registerAllocation.mapToRegister(operand);
              // Apply same MOVE logic as remapSuccessorRegisters.
              if (srcReg.isRef()) {
                bool isLastOccurrence =
                    (lastOccurrence.lookup(operand) == localIdx);
                bool isLastUse =
                    registerAllocation.liveness_.isLastRealValueUse(
                        operand, &op, globalIdx);
                if (isLastOccurrence && isLastUse) {
                  srcReg.setMove(true);
                }
              }
              branchOperandRegs[globalIdx] = srcReg;
            }
          }
        }

        for (auto &operand : op.getOpOperands()) {
          unsigned idx = operand.getOperandNumber();
          Register reg;
          if (branchOperandRegs.count(idx)) {
            // Use the pre-computed register with correct MOVE bit.
            reg = branchOperandRegs[idx];
          } else {
            // Use standard mapUseToRegister for non-branch operands.
            reg = registerAllocation.mapUseToRegister(operand.get(), &op, idx);
          }
          operandRegStrs.push_back(reg.toString());
        }
        op.setAttr("operand_registers",
                   getStrArrayAttr(builder, operandRegStrs));
      }
      if (op.getNumResults() == 0)
        continue;
      SmallVector<std::string, 8> regStrs;
      regStrs.reserve(op.getNumResults());
      for (auto result : op.getResults()) {
        auto reg = registerAllocation.map_[result];
        regStrs.push_back(reg.toString());
      }
      op.setAttr("result_registers", getStrArrayAttr(builder, regStrs));
    }

    Operation *terminatorOp = block.getTerminator();
    if (terminatorOp->getNumSuccessors() > 0) {
      SmallVector<Attribute, 2> successorAttrs;
      successorAttrs.reserve(terminatorOp->getNumSuccessors());
      for (int i = 0; i < terminatorOp->getNumSuccessors(); ++i) {
        auto srcDstRegs =
            registerAllocation.remapSuccessorRegisters(terminatorOp, i);
        SmallVector<std::string, 8> remappingStrs;
        for (auto &srcDstReg : srcDstRegs) {
          remappingStrs.push_back(llvm::formatv("{}->{}",
                                                srcDstReg.first.toString(),
                                                srcDstReg.second.toString())
                                      .str());
        }
        successorAttrs.push_back(getStrArrayAttr(builder, remappingStrs));
      }
      terminatorOp->setAttr("remap_registers",
                            builder.getArrayAttr(successorAttrs));
    }
  }

  return success();
}

// Sorts blocks in dominance order such that the entry block is first and
// all of the following blocks are dominated only by blocks that have come
// before them in the list. This ensures that we always know all registers for
// block live-in values as we walk the blocks.
static SmallVector<Block *, 8>
sortBlocksInDominanceOrder(IREE::VM::FuncOp funcOp) {
  if (funcOp.getBlocks().size() == 1) {
    // Dominance info cannot be computed for regions with one block.
    return {&funcOp.getBlocks().front()};
  }

  DominanceInfo dominanceInfo(funcOp);
  llvm::SmallSetVector<Block *, 8> unmarkedBlocks;
  for (auto &block : funcOp.getBlocks()) {
    unmarkedBlocks.insert(&block);
  }
  llvm::SmallSetVector<Block *, 8> markedBlocks;
  std::function<void(Block *)> visit = [&](Block *block) {
    if (markedBlocks.count(block) > 0)
      return;
    for (auto *childBlock : dominanceInfo.getNode(block)->children()) {
      visit(childBlock->getBlock());
    }
    markedBlocks.insert(block);
  };
  while (!unmarkedBlocks.empty()) {
    visit(unmarkedBlocks.pop_back_val());
  }
  auto orderedBlocks = markedBlocks.takeVector();
  std::reverse(orderedBlocks.begin(), orderedBlocks.end());
  return orderedBlocks;
}

// Linear scan allocator implementation.
// Uses LiveIntervals for global register lifetime tracking and enables
// cross-block register reuse for both integer and reference registers.
LogicalResult RegisterAllocation::recalculate(IREE::VM::FuncOp funcOp) {
  map_.clear();
  dominanceInfo_.reset();
  maxI32RegisterOrdinal_ = -1;
  maxRefRegisterOrdinal_ = -1;
  scratchI32RegisterCount_ = 0;
  scratchRefRegisterCount_ = 0;

  // Build live intervals for global allocation.
  LiveIntervals liveIntervals;
  if (failed(liveIntervals.build(funcOp))) {
    return funcOp.emitError() << "failed to build live intervals";
  }

  // Also compute liveness for MOVE bit computation (still needed).
  if (failed(liveness_.recalculate(funcOp))) {
    return funcOp.emitError()
           << "failed to calculate required liveness information";
  }

  // Compute dominance info for back-edge detection in MOVE bit logic.
  // Only needed for functions with more than one block.
  if (funcOp.getBlocks().size() > 1) {
    dominanceInfo_.emplace(funcOp);
  }

  // Global register banks (single instance for entire function).
  IREE::VM::RegisterBank intBank(Register::kInt32RegisterCount);
  IREE::VM::RegisterBank refBank(Register::kRefRegisterCount);

  // ===== PHASE A: Pre-allocate entry block args (monotonic, ABI) =====
  // Also set the assigned field in the interval so it can be properly expired.
  Block *entryBlock = &funcOp.getBlocks().front();
  for (auto blockArg : entryBlock->getArguments()) {
    Type type = blockArg.getType();
    if (isa<IREE::VM::RefType>(type)) {
      // Entry block ref args: monotonic allocation for ABI stability.
      auto ord = refBank.allocate(0, /*fromEnd=*/true);
      if (!ord) {
        return funcOp.emitError()
               << "register allocation failed for entry block arg "
               << blockArg.getArgNumber() << " (ref bank overflow)";
      }
      map_[blockArg] = Register::getRef(type, *ord, /*isMove=*/false);
      // Set assigned in interval for expiration tracking.
      if (LiveInterval *interval =
              const_cast<LiveInterval *>(liveIntervals.getInterval(blockArg))) {
        interval->assigned = *ord;
      }
    } else if (type.isIntOrFloat()) {
      // Entry block int args: monotonic allocation for ABI stability.
      size_t byteWidth = IREE::Util::getRoundedElementByteWidth(type);
      auto ord = intBank.allocate(byteWidth, /*fromEnd=*/true);
      if (!ord) {
        return funcOp.emitError()
               << "register allocation failed for entry block arg "
               << blockArg.getArgNumber() << " (int bank overflow)";
      }
      map_[blockArg] = Register::getValue(type, *ord);
      // Set assigned in interval for expiration tracking.
      if (LiveInterval *interval =
              const_cast<LiveInterval *>(liveIntervals.getInterval(blockArg))) {
        interval->assigned = *ord;
      }
    }
  }

  // ===== PHASE B: Set coalescing hints from branch operands =====
  // Maps block arg -> preferred register ordinal.
  // Separate maps for i32 and ref banks.
  llvm::DenseMap<Value, int> intPreferences;
  llvm::DenseMap<Value, int> refPreferences;
  for (auto *block : liveIntervals.getBlockOrder()) {
    Operation *term = block->getTerminator();
    if (!term->getNumSuccessors()) {
      continue;
    }
    auto branchOp = dyn_cast<BranchOpInterface>(term);
    if (!branchOp) {
      continue;
    }

    for (unsigned succIdx = 0; succIdx < term->getNumSuccessors(); ++succIdx) {
      Block *succ = term->getSuccessor(succIdx);
      OperandRange operands =
          branchOp.getSuccessorOperands(succIdx).getForwardedOperands();
      for (auto [idx, operand] : llvm::enumerate(operands)) {
        if (idx >= succ->getNumArguments()) {
          break;
        }
        BlockArgument arg = succ->getArgument(idx);
        // If source is already assigned, record preference.
        auto srcIt = map_.find(operand);
        if (srcIt != map_.end()) {
          if (isa<IREE::VM::RefType>(arg.getType())) {
            refPreferences[arg] = srcIt->second.ordinal();
          } else {
            intPreferences[arg] = srcIt->second.ordinal();
          }
        }
      }
    }
  }

  // ===== PHASE C: Linear scan for i32 intervals =====
  // Active intervals sorted by end position.
  SmallVector<const LiveInterval *> activeInt;

  // Seed active list with pre-allocated entry block arg intervals.
  // This allows their registers to be expired and reused.
  // TODO: Refactor Phase C and D to share code - the int and ref linear scan
  // logic is nearly identical.
  for (auto blockArg : entryBlock->getArguments()) {
    if (blockArg.getType().isIntOrFloat()) {
      if (const LiveInterval *interval = liveIntervals.getInterval(blockArg)) {
        // Insert in sorted order by end position.
        auto it =
            std::lower_bound(activeInt.begin(), activeInt.end(), interval,
                             [](const LiveInterval *a, const LiveInterval *b) {
                               return a->end < b->end;
                             });
        activeInt.insert(it, interval);
      }
    }
  }

  // Helper to expire old int intervals.
  // Use < (not <=) so intervals ending at currentStart are NOT expired yet.
  // This prevents the bug where multiple block args with the same start
  // position get assigned the same register.
  auto expireOldIntIntervals = [&](uint32_t currentStart) {
    while (!activeInt.empty() && activeInt.front()->end < currentStart) {
      const LiveInterval *expired = activeInt.front();
      intBank.release(expired->assigned, expired->byteWidth);
      activeInt.erase(activeInt.begin());
    }
  };

  // Helper to insert into active int list (maintain sorted by end).
  auto insertActiveInt = [&](const LiveInterval *interval) {
    auto it =
        std::lower_bound(activeInt.begin(), activeInt.end(), interval,
                         [](const LiveInterval *a, const LiveInterval *b) {
                           return a->end < b->end;
                         });
    activeInt.insert(it, interval);
  };

  for (const LiveInterval *interval : liveIntervals.getSortedByStart()) {
    // Skip refs (handled in Phase D).
    if (interval->isRef) {
      continue;
    }

    // Skip already-assigned values (entry block args).
    if (map_.count(interval->value)) {
      continue;
    }

    // Expire intervals that ended before this one starts.
    expireOldIntIntervals(interval->start);

    // Try preference (coalescing hint) first.
    int ord = -1;
    auto prefIt = intPreferences.find(interval->value);
    if (prefIt != intPreferences.end()) {
      int prefOrd = prefIt->second;
      if (intBank.allocateAt(prefOrd, interval->byteWidth)) {
        ord = prefOrd;
      }
    }

    // Fall back to first-fit allocation.
    if (ord < 0) {
      auto allocated = intBank.allocate(interval->byteWidth);
      if (!allocated) {
        return funcOp.emitError()
               << "register allocation failed for value (int bank overflow)";
      }
      ord = *allocated;
    }

    // Store assignment in interval (for debugging) and map.
    const_cast<LiveInterval *>(interval)->assigned = ord;
    map_[interval->value] = Register::getValue(interval->value.getType(), ord);

    // Add to active set.
    insertActiveInt(interval);
  }

  maxI32RegisterOrdinal_ = intBank.getMaxUsed();

  // ===== PHASE D: Global linear scan for ref intervals =====
  // Active ref intervals sorted by end position.
  SmallVector<const LiveInterval *> activeRef;

  // Seed active list with pre-allocated entry block arg intervals.
  // This allows their registers to be expired and reused.
  for (auto blockArg : entryBlock->getArguments()) {
    if (isa<IREE::VM::RefType>(blockArg.getType())) {
      if (const LiveInterval *interval = liveIntervals.getInterval(blockArg)) {
        // Insert in sorted order by end position.
        auto it =
            std::lower_bound(activeRef.begin(), activeRef.end(), interval,
                             [](const LiveInterval *a, const LiveInterval *b) {
                               return a->end < b->end;
                             });
        activeRef.insert(it, interval);
      }
    }
  }

  // Helper to expire old ref intervals.
  // Use < (not <=) so intervals ending at currentStart are NOT expired yet.
  // This prevents the bug where multiple block args with the same start
  // position get assigned the same register.
  auto expireOldRefIntervals = [&](uint32_t currentStart) {
    while (!activeRef.empty() && activeRef.front()->end < currentStart) {
      const LiveInterval *expired = activeRef.front();
      refBank.release(expired->assigned, 0);
      activeRef.erase(activeRef.begin());
    }
  };

  // Helper to insert into active ref list (maintain sorted by end).
  auto insertActiveRef = [&](const LiveInterval *interval) {
    auto it =
        std::lower_bound(activeRef.begin(), activeRef.end(), interval,
                         [](const LiveInterval *a, const LiveInterval *b) {
                           return a->end < b->end;
                         });
    activeRef.insert(it, interval);
  };

  for (const LiveInterval *interval : liveIntervals.getSortedByStart()) {
    // Only process refs.
    if (!interval->isRef) {
      continue;
    }

    // Skip already-assigned values (entry block args).
    if (map_.count(interval->value)) {
      continue;
    }

    // Expire intervals that ended before this one starts.
    expireOldRefIntervals(interval->start);

    // Try preference (coalescing hint) first.
    int ord = -1;
    auto prefIt = refPreferences.find(interval->value);
    if (prefIt != refPreferences.end()) {
      int prefOrd = prefIt->second;
      if (refBank.allocateAt(prefOrd, 0)) {
        ord = prefOrd;
      }
    }

    // Fall back to first-fit allocation.
    if (ord < 0) {
      auto allocated = refBank.allocate(0);
      if (!allocated) {
        return funcOp.emitError() << "register allocation failed for ref value "
                                     "(ref bank overflow)";
      }
      ord = *allocated;
    }

    // Store assignment in interval (for debugging) and map.
    const_cast<LiveInterval *>(interval)->assigned = ord;
    map_[interval->value] =
        Register::getRef(interval->value.getType(), ord, /*isMove=*/false);

    // Add to active set.
    insertActiveRef(interval);
  }

  maxRefRegisterOrdinal_ = refBank.getMaxUsed();

  // Verify bounds.
  if (maxI32RegisterOrdinal_ > Register::kInt32RegisterCount ||
      maxRefRegisterOrdinal_ > Register::kRefRegisterCount) {
    return funcOp.emitError() << "function overflows stack register banks; "
                                 "spilling to memory not yet implemented";
  }

  // Identify discards that can be elided because their operands were released
  // via MOVE on preceding operations.
  computeElidableDiscards(funcOp);

  return success();
}

void RegisterAllocation::computeElidableDiscards(IREE::VM::FuncOp funcOp) {
  discardOperandElidability_.clear();

  // For each discard operand, check if it was released via MOVE on a preceding
  // operation. An operand is elidable when:
  // 1. The value has a "real" use (non-discard) before this discard in the
  //    same block, AND
  // 2. That real use actually got MOVE set (isLastRealValueUse is true and
  //    the op implements RefMoveInterface)
  //
  // Values that only have the discard as their use in a block (e.g., coming
  // from block arguments or definitions in predecessors with no real use
  // before the discard) cannot be elided - the discard is the only thing
  // releasing them.
  for (auto &block : funcOp.getBlocks()) {
    for (auto &op : block.getOperations()) {
      auto discardOp = dyn_cast<IREE::VM::DiscardRefsOp>(&op);
      if (!discardOp)
        continue;

      SmallVector<bool> operandElidability;
      for (Value ref : discardOp.getRefs()) {
        // Check if there's a preceding real use in this block that got MOVE.
        bool hasPrecedingMoveUse = false;
        for (auto it = Block::iterator(discardOp); it != block.begin();) {
          --it;
          Operation *precedingOp = &*it;
          for (OpOperand &operand : precedingOp->getOpOperands()) {
            if (operand.get() == ref &&
                !isa<IREE::VM::DiscardRefsOp>(precedingOp)) {
              // Found a real use before the discard.
              // Check if MOVE was actually set on this use.
              if (liveness_.isLastRealValueUse(ref, precedingOp,
                                               operand.getOperandNumber())) {
                // Also check if the op implements RefMoveInterface and says
                // this operand is movable.
                if (auto moveOp =
                        dyn_cast<IREE::VM::RefMoveInterface>(precedingOp)) {
                  if (moveOp.isRefOperandMovable(operand.getOperandNumber())) {
                    hasPrecedingMoveUse = true;
                  }
                }
              }
              break;
            }
          }
          if (hasPrecedingMoveUse)
            break;
        }
        operandElidability.push_back(hasPrecedingMoveUse);
      }
      discardOperandElidability_[discardOp] = std::move(operandElidability);
    }
  }
}

Register RegisterAllocation::mapToRegister(Value value) const {
  auto it = map_.find(value);
  assert(it != map_.end());
  return it->getSecond();
}

Register RegisterAllocation::mapUseToRegister(Value value, Operation *useOp,
                                              int operandIndex) {
  auto reg = mapToRegister(value);
  if (reg.isRef() && liveness_.isLastRealValueUse(value, useOp, operandIndex)) {
    // Set MOVE bit when this is the last real use of the value. Only set MOVE
    // if the op implements RefMoveInterface and says this operand is movable.
    // This prevents setting MOVE on ops that don't honor it in the runtime.
    if (auto moveOp = dyn_cast<IREE::VM::RefMoveInterface>(useOp)) {
      if (moveOp.isRefOperandMovable(operandIndex)) {
        reg.setMove(true);
      }
    }
  }
  return reg;
}

// A feedback arc set containing the minimal list of cycle-causing edges.
// https://en.wikipedia.org/wiki/Feedback_arc_set
struct FeedbackArcSet {
  using NodeID = Register;
  using Edge = std::pair<NodeID, NodeID>;

  // Edges making up a DAG (inputEdges - feedbackEdges).
  SmallVector<Edge, 8> acyclicEdges;

  // Edges of the feedback arc set that, if added to acyclicEdges, would cause
  // cycles.
  SmallVector<Edge, 8> feedbackEdges;

  // Computes the FAS of a given directed graph that may contain cycles.
  static FeedbackArcSet compute(ArrayRef<Edge> inputEdges) {
    FeedbackArcSet result;
    if (inputEdges.empty()) {
      return result;
    } else if (inputEdges.size() == 1) {
      result.acyclicEdges.push_back(inputEdges.front());
      return result;
    }

    struct FASNode {
      NodeID id;
      int indegree = 0;
      int outdegree = 0;
    };
    // This should not be modified after creation in this loop. We take pointers
    // to its entries so do not want to invalidate them with reallocation.
    llvm::SmallDenseMap<NodeID, FASNode> nodes;
    for (auto &edge : inputEdges) {
      NodeID sourceID = edge.first.asBaseRegister();
      NodeID sinkID = edge.second.asBaseRegister();
      assert(sourceID != sinkID && "self-cycles not supported");
      if (nodes.count(sourceID) == 0) {
        nodes.insert({sourceID, {sourceID, 0, 0}});
      }
      if (nodes.count(sinkID) == 0) {
        nodes.insert({sinkID, {sinkID, 0, 0}});
      }
    }

    struct FASEdge {
      FASNode *source;
      FASNode *sink;
    };
    int maxOutdegree = 0;
    int maxIndegree = 0;
    SmallVector<FASEdge, 8> edges;
    for (auto &edge : inputEdges) {
      NodeID sourceID = edge.first.asBaseRegister();
      NodeID sinkID = edge.second.asBaseRegister();
      auto &sourceNode = nodes[sourceID];
      ++sourceNode.outdegree;
      maxOutdegree = std::max(maxOutdegree, sourceNode.outdegree);
      auto &sinkNode = nodes[sinkID];
      ++sinkNode.indegree;
      maxIndegree = std::max(maxIndegree, sinkNode.indegree);
      edges.push_back({&sourceNode, &sinkNode});
    }

    std::vector<SmallVector<FASNode *, 2>> buckets;
    buckets.resize(std::max(maxOutdegree, maxIndegree) + 2);
    auto nodeToBucketIndex = [&](FASNode *node) {
      return node->indegree == 0 || node->outdegree == 0
                 ? buckets.size() - 1
                 : std::abs(node->outdegree - node->indegree);
    };
    auto assignBucket = [&](FASNode *node) {
      buckets[nodeToBucketIndex(node)].push_back(node);
    };
    auto removeBucket = [&](FASNode *node) {
      int index = nodeToBucketIndex(node);
      auto it = std::find(buckets[index].begin(), buckets[index].end(), node);
      if (it != buckets[index].end()) {
        buckets[index].erase(it);
      }
    };
    llvm::SmallPtrSet<FASNode *, 8> remainingNodes;
    for (auto &nodeEntry : nodes) {
      assignBucket(&nodeEntry.getSecond());
      remainingNodes.insert(&nodeEntry.getSecond());
    }

    auto removeNode = [&](FASNode *node) {
      SmallVector<FASEdge> inEdges;
      inEdges.reserve(node->indegree);
      SmallVector<FASEdge> outEdges;
      outEdges.reserve(node->outdegree);
      for (auto &edge : edges) {
        if (edge.sink == node)
          inEdges.push_back(edge);
        if (edge.source == node)
          outEdges.push_back(edge);
      }
      bool collectInEdges = node->indegree <= node->outdegree;
      bool collectOutEdges = !collectInEdges;

      SmallVector<Edge> results;
      for (auto &edge : inEdges) {
        if (edge.source == node)
          continue;
        if (collectInEdges) {
          results.push_back({edge.source->id, edge.sink->id});
        }
        removeBucket(edge.source);
        --edge.source->outdegree;
        assert(edge.source->outdegree >= 0 && "outdegree has become negative");
        assignBucket(edge.source);
      }
      for (auto &edge : outEdges) {
        if (edge.sink == node)
          continue;
        if (collectOutEdges) {
          results.push_back({edge.source->id, edge.sink->id});
        }
        removeBucket(edge.sink);
        --edge.sink->indegree;
        assert(edge.sink->indegree >= 0 && "indegree has become negative");
        assignBucket(edge.sink);
      }

      remainingNodes.erase(node);
      edges.erase(std::remove_if(edges.begin(), edges.end(),
                                 [&](const FASEdge &edge) {
                                   return edge.source == node ||
                                          edge.sink == node;
                                 }),
                  edges.end());
      return results;
    };
    auto ends = buckets.back();
    while (!remainingNodes.empty()) {
      while (!ends.empty()) {
        auto *node = ends.front();
        ends.erase(ends.begin());
        removeNode(node);
      }
      if (remainingNodes.empty())
        break;
      for (ssize_t i = buckets.size() - 1; i >= 0; --i) {
        if (buckets[i].empty())
          continue;
        auto *bucket = buckets[i].front();
        buckets[i].erase(buckets[i].begin());
        auto feedbackEdges = removeNode(bucket);
        result.feedbackEdges.append(feedbackEdges.begin(), feedbackEdges.end());
        break;
      }
    }

    // Build the DAG of the remaining edges now that we've isolated the ones
    // that cause cycles.
    llvm::SmallSetVector<NodeID, 8> acyclicNodes;
    SmallVector<Edge, 8> acyclicEdges;
    for (auto &inputEdge : inputEdges) {
      auto it = std::find_if(result.feedbackEdges.begin(),
                             result.feedbackEdges.end(), [&](const Edge &edge) {
                               return edge.first == inputEdge.first &&
                                      edge.second == inputEdge.second;
                             });
      if (it == result.feedbackEdges.end()) {
        acyclicEdges.push_back(inputEdge);
        acyclicNodes.insert(inputEdge.first.asBaseRegister());
        acyclicNodes.insert(inputEdge.second.asBaseRegister());
      }
    }

    // Topologically sort the DAG so that we don't overwrite anything.
    llvm::SmallSetVector<NodeID, 8> unmarkedNodes = acyclicNodes;
    llvm::SmallSetVector<NodeID, 8> markedNodes;
    std::function<void(NodeID)> visit = [&](NodeID node) {
      if (markedNodes.count(node) > 0)
        return;
      for (auto &edge : acyclicEdges) {
        if (edge.first != node)
          continue;
        visit(edge.second);
      }
      markedNodes.insert(node);
    };
    while (!unmarkedNodes.empty()) {
      visit(unmarkedNodes.pop_back_val());
    }
    for (auto node : markedNodes.takeVector()) {
      for (auto &edge : acyclicEdges) {
        if (edge.first != node)
          continue;
        result.acyclicEdges.push_back({edge.first, edge.second});
      }
    }

    return result;
  }
};

SmallVector<std::pair<Register, Register>, 8>
RegisterAllocation::remapSuccessorRegisters(Operation *op, int successorIndex) {
  auto branchOp = cast<BranchOpInterface>(op);
  auto *targetBlock = op->getSuccessor(successorIndex);
  auto targetOperands =
      branchOp.getSuccessorOperands(successorIndex).getForwardedOperands();

  // Get the base operand index for this successor's operands in the branch op.
  // This is needed to query isLastRealValueUse with the correct global index.
  // Note: getBeginOperandIndex() asserts on empty ranges, so we handle that.
  unsigned baseOperandIndex =
      targetOperands.empty() ? 0 : targetOperands.getBeginOperandIndex();

  return remapSuccessorRegisters(op, targetBlock, targetOperands,
                                 baseOperandIndex);
}

SmallVector<std::pair<Register, Register>, 8>
RegisterAllocation::remapSuccessorRegisters(Location loc, Block *targetBlock,
                                            OperandRange targetOperands) {
  // Legacy overload - called without branch op context, so no MOVE bits.
  // This path is only used for testing/annotation where we don't have the op.
  return remapSuccessorRegisters(/*branchOp=*/nullptr, targetBlock,
                                 targetOperands, /*baseOperandIndex=*/0);
}

// Computes register remapping for branch operands from source block to target.
//
// This function handles the MOVE bit logic for ref operands:
// - MOVE transfers ownership (runtime nulls source after copy)
// - No MOVE retains (runtime increments refcount of destination)
//
// MOVE is set when all three conditions are true:
// 1. This is the last occurrence of the ref value in this successor's list
//    (handles same ref appearing multiple times like `^bb(%ref, %ref)`)
// 2. The ref is not live after this branch (checked via isLastRealValueUse)
// 3. The branch op supports MOVE on this operand (RefMoveInterface)
//
// MOVE on back-edges (loops) is CORRECT:
// - Ownership transfers to the next iteration's block argument
// - NOT MOVE would leak: source retains AND destination retains = refcount++
// - At loop exit, source register would be non-null causing spurious release
//
// Returns a list of (src, dst) register pairs that need to be remapped.
// Pairs where src == dst (coalesced) are omitted.
SmallVector<std::pair<Register, Register>, 8>
RegisterAllocation::remapSuccessorRegisters(Operation *branchOp,
                                            Block *targetBlock,
                                            OperandRange targetOperands,
                                            unsigned baseOperandIndex) {
  // Compute the initial directed graph of register movements.
  // This may contain cycles ([reg 0->1], [reg 1->0], ...) that would not be
  // possible to evaluate as a direct remapping.
  SmallVector<std::pair<Register, Register>, 8> srcDstRegs;

  // Build last-occurrence map for ref operands in this successor's list.
  // Only the last occurrence of a value can get MOVE (for multi-use cases like
  // passing same ref to both branches of cond_br).
  llvm::DenseMap<Value, unsigned> lastOccurrence;
  for (auto [idx, operand] : llvm::enumerate(targetOperands)) {
    if (isa<IREE::VM::RefType>(operand.getType())) {
      lastOccurrence[operand] = idx;
    }
  }

  for (auto [idx, operand] : llvm::enumerate(targetOperands)) {
    auto srcReg = mapToRegister(operand);
    BlockArgument targetArg = targetBlock->getArgument(idx);
    auto dstReg = mapToRegister(targetArg);

    // Apply MOVE bit logic for ref registers.
    if (branchOp && srcReg.isRef()) {
      // Check if this is the last occurrence of this value in the operand list.
      bool isLastOccurrence = (lastOccurrence.lookup(operand) == idx);

      // Compute global operand index for liveness query.
      unsigned globalOperandIndex = baseOperandIndex + idx;

      // Check if this is the last real use of the value (not live after branch,
      // not used elsewhere in the function after this point).
      bool isLastUse =
          liveness_.isLastRealValueUse(operand, branchOp, globalOperandIndex);

      // Check if the branch op supports MOVE on this operand.
      bool isMovable = false;
      if (auto moveOp = dyn_cast<IREE::VM::RefMoveInterface>(branchOp)) {
        isMovable = moveOp.isRefOperandMovable(globalOperandIndex);
      }

      // Set MOVE if:
      // 1. This is the last occurrence in this successor's operand list
      // 2. This is the last real use of the value (not live after branch)
      // 3. The branch op supports MOVE on this operand
      // Note: MOVE is correct even on back-edges. The ref ownership transfers
      // to the destination block arg, which is correct for loop-carried values.
      if (isLastOccurrence && isLastUse && isMovable) {
        srcReg.setMove(true);
      }
    }

    if (srcReg != dstReg) {
      srcDstRegs.push_back({srcReg, dstReg});
    }
  }

  // Compute the feedback arc set to determine which edges are the ones inducing
  // cycles, if any. This also provides us a DAG that we can trivially remap
  // without worrying about cycles.
  auto feedbackArcSet = FeedbackArcSet::compute(srcDstRegs);
  assert(feedbackArcSet.acyclicEdges.size() +
                 feedbackArcSet.feedbackEdges.size() ==
             srcDstRegs.size() &&
         "lost an edge during feedback arc set computation");

  // If there's no cycles we can simply use the sorted DAG produced.
  if (feedbackArcSet.feedbackEdges.empty()) {
    return feedbackArcSet.acyclicEdges;
  }

  // The tail registers in each bank is reserved for swapping, when required.
  int localScratchI32RegCount = 0;
  int localScratchRefRegCount = 0;
  for (auto feedbackEdge : feedbackArcSet.feedbackEdges) {
    Register scratchReg;
    if (feedbackEdge.first.isRef()) {
      localScratchRefRegCount += 1;
      scratchReg = Register::getWithSameType(
          feedbackEdge.first, maxRefRegisterOrdinal_ + localScratchRefRegCount);
    } else {
      localScratchI32RegCount += 1;
      scratchReg = Register::getWithSameType(
          feedbackEdge.first, maxI32RegisterOrdinal_ + localScratchI32RegCount);
      // Integer types that use more than one register slot will be emitted
      // as remapping per 4-byte word, so we have to account for the extra
      // temporaries. See BytecodeEncoder:encodeBranch().
      assert(scratchReg.byteWidth() >= 4 && "expected >= i32");
      localScratchI32RegCount += scratchReg.byteWidth() / 4 - 1;
    }
    feedbackArcSet.acyclicEdges.insert(feedbackArcSet.acyclicEdges.begin(),
                                       {feedbackEdge.first, scratchReg});
    // The scratch register's use as source is its last use, so set MOVE.
    // This ensures the ref in the scratch register is released after the copy,
    // preventing leaks when the branch takes an alternate path (e.g., loop
    // exit).
    Register scratchSrc = scratchReg;
    if (scratchSrc.isRef()) {
      scratchSrc.setMove(true);
    }
    feedbackArcSet.acyclicEdges.push_back({scratchSrc, feedbackEdge.second});
  }
  if (localScratchI32RegCount > 0) {
    scratchI32RegisterCount_ =
        std::max(scratchI32RegisterCount_, localScratchI32RegCount);
    assert(getMaxI32RegisterOrdinal() <= Register::kInt32RegisterCount &&
           "spilling i32 regs");
    if (getMaxI32RegisterOrdinal() > Register::kInt32RegisterCount &&
        branchOp) {
      mlir::emitError(branchOp->getLoc())
          << "spilling entire i32 register address space";
    }
  }
  if (localScratchRefRegCount > 0) {
    scratchRefRegisterCount_ =
        std::max(scratchRefRegisterCount_, localScratchRefRegCount);
    assert(getMaxRefRegisterOrdinal() <= Register::kRefRegisterCount &&
           "spilling ref regs");
    if (getMaxRefRegisterOrdinal() > Register::kRefRegisterCount && branchOp) {
      mlir::emitError(branchOp->getLoc())
          << "spilling entire ref register address space";
    }
  }

  return feedbackArcSet.acyclicEdges;
}

} // namespace mlir::iree_compiler
