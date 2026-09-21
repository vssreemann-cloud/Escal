"""
Sliding-Window Reservation Ledger: Manages future resource availability matrices.
"""

import asyncio
import time
from typing import List, Dict, Optional
from core.models import ResourceVector, ServiceGraph


class SlidingWindowReservationLedger:
    def __init__(self, capacity: ResourceVector, lookahead_sec: float = 10.0, slot_size_sec: float = 0.5):
        self.capacity = capacity
        self.lookahead_sec = lookahead_sec
        self.slot_size_sec = slot_size_sec
        self.num_slots = int(lookahead_sec / slot_size_sec)
        self.time_slots: List[ResourceVector] = [ResourceVector() for _ in range(self.num_slots)]
        self.lock = asyncio.Lock()

    def _get_slot_index(self, start_time: float, target_time: float) -> Optional[int]:
        delta = target_time - start_time
        if delta < 0:
            return 0
        idx = int(delta / self.slot_size_sec)
        return idx if idx < self.num_slots else None

    async def try_reserve(self, graph: ServiceGraph) -> bool:
        async with self.lock:
            now = time.time()
            temp_slots = [
                ResourceVector(s.llm_tokens_per_sec, s.gpu_memory_mb, s.tool_concurrency_slots)
                for s in self.time_slots
            ]

            reservations: Dict[int, ResourceVector] = {}

            def traverse(node_id: str, current_time: float, cum_prob: float):
                if node_id not in graph.nodes:
                    return
                node = graph.nodes[node_id]
                eff_prob = cum_prob * node.probability

                start_idx = self._get_slot_index(now, current_time)
                end_idx = self._get_slot_index(now, current_time + node.est_duration_sec)

                if start_idx is not None:
                    end_idx = min(self.num_slots, end_idx or self.num_slots)
                    for s in range(start_idx, max(start_idx + 1, end_idx)):
                        weighted_req = ResourceVector(
                            llm_tokens_per_sec=node.resource_req.llm_tokens_per_sec * eff_prob,
                            gpu_memory_mb=node.resource_req.gpu_memory_mb * eff_prob,
                            tool_concurrency_slots=node.resource_req.tool_concurrency_slots * eff_prob,
                        )
                        reservations[s] = reservations.get(s, ResourceVector()) + weighted_req

                for child_id, branch_p in node.children:
                    traverse(child_id, current_time + node.est_duration_sec, eff_prob * branch_p)

            traverse(graph.entry_node_id, now, 1.0)

            # Capacity Violation Check
            for s_idx, req in reservations.items():
                if s_idx < self.num_slots:
                    projected = temp_slots[s_idx] + req
                    if not projected.fits_in(self.capacity):
                        return False

            # Commit Reservation
            for s_idx, req in reservations.items():
                if s_idx < self.num_slots:
                    self.time_slots[s_idx] = self.time_slots[s_idx] + req

            return True

    async def release_pruned_branch(self, graph: ServiceGraph, unchosen_node_id: str):
        """Dynamic Speculative Pruning: Instantly reclaim reserved branch budget."""
        async with self.lock:
            if unchosen_node_id in graph.nodes:
                node = graph.nodes[unchosen_node_id]
                freed = ResourceVector(
                    llm_tokens_per_sec=node.resource_req.llm_tokens_per_sec * node.probability,
                    gpu_memory_mb=node.resource_req.gpu_memory_mb * node.probability,
                    tool_concurrency_slots=node.resource_req.tool_concurrency_slots * node.probability,
                )
                # Deduct freed resources from time slots
                self.time_slots = [s - freed for s in self.time_slots]
