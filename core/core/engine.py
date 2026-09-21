"""
Execution Controller: Simulates Naive Execution vs RSGC Dynamic Control.
"""

import asyncio
import time
from typing import List, Tuple
from config import Config
from core.models import ResourceVector, SimulationResult, WorkflowRequest
from core.predictor import ExecutionGraphPredictor
from core.ledger import SlidingWindowReservationLedger


class WorkloadSimulator:
    def __init__(self):
        self.capacity = ResourceVector(
            llm_tokens_per_sec=Config.TOTAL_LLM_TOKENS_PER_SEC,
            gpu_memory_mb=Config.TOTAL_GPU_MEMORY_MB,
            tool_concurrency_slots=Config.TOTAL_TOOL_SLOTS,
        )

    async def run_naive_simulation(self, requests: List[WorkflowRequest]) -> SimulationResult:
        """Naive FIFO Mode: Fires tasks immediately without reservation, causing 429s/crashes."""
        current_load = ResourceVector()
        logs = []
        success = 0
        failed = 0
        rate_limit_errors = 0
        start_time = time.time()

        async def execute_naive(req: WorkflowRequest):
            nonlocal current_load, success, failed, rate_limit_errors
            graph = ExecutionGraphPredictor.predict_graph(req.workflow_id, req.prompt_intent)
            
            # Sum up immediate node requirements
            root = graph.nodes[graph.entry_node_id]
            req_vec = root.resource_req

            if not (current_load + req_vec).fits_in(self.capacity):
                rate_limit_errors += 1
                failed += 1
                logs.append(f"❌ [NAIVE] {req.workflow_id}: 429 RateLimit/CapacityExceeded!")
                return

            current_load = current_load + req_vec
            logs.append(f"▶️ [NAIVE] {req.workflow_id}: Started execution.")
            await asyncio.sleep(root.est_duration_sec)
            current_load = current_load - req_vec
            success += 1
            logs.append(f"✅ [NAIVE] {req.workflow_id}: Finished successfully.")

        await asyncio.gather(*[execute_naive(r) for r in requests])
        elapsed = time.time() - start_time

        return SimulationResult(
            mode="Naive FIFO Execution",
            total_workflows=len(requests),
            successful_workflows=success,
            failed_workflows=failed,
            rate_limit_errors=rate_limit_errors,
            avg_latency_sec=round(elapsed, 2),
            logs=logs,
        )

    async def run_rsgc_simulation(self, requests: List[WorkflowRequest]) -> SimulationResult:
        """RSGC Mode: Predicts DAGs, reserves sliding slots, and queues smoothly."""
        ledger = SlidingWindowReservationLedger(capacity=self.capacity)
        logs = []
        success = 0
        failed = 0
        rate_limit_errors = 0
        start_time = time.time()

        async def execute_rsgc(req: WorkflowRequest):
            nonlocal success, failed, rate_limit_errors
            graph = ExecutionGraphPredictor.predict_graph(req.workflow_id, req.prompt_intent)

            reserved = False
            retries = 0
            while not reserved and retries < 5:
                reserved = await ledger.try_reserve(graph)
                if not reserved:
                    retries += 1
                    logs.append(f"⏳ [RSGC] {req.workflow_id}: Capacity full. Queued (retry {retries})...")
                    await asyncio.sleep(0.6)

            if not reserved:
                failed += 1
                logs.append(f"⚠️ [RSGC] {req.workflow_id}: Dropped after max queue retries.")
                return

            logs.append(f"🔒 [RSGC] {req.workflow_id}: Granted downstream reservation. Executing...")
            root = graph.nodes[graph.entry_node_id]
            await asyncio.sleep(root.est_duration_sec)

            # Speculative branch selection simulation
            chosen = "web_search_branch"
            pruned = "code_exec_branch"
            logs.append(f"✂️ [RSGC] {req.workflow_id}: Took '{chosen}', pruned '{pruned}'. Reclaimed slots!")
            await ledger.release_pruned_branch(graph, pruned)

            await asyncio.sleep(graph.nodes[chosen].est_duration_sec)
            success += 1
            logs.append(f"✅ [RSGC] {req.workflow_id}: Completed cleanly with zero rate limits.")

        await asyncio.gather(*[execute_rsgc(r) for r in requests])
        elapsed = time.time() - start_time

        return SimulationResult(
            mode="RSGC Predictive Reservation",
            total_workflows=len(requests),
            successful_workflows=success,
            failed_workflows=failed,
            rate_limit_errors=rate_limit_errors,
            avg_latency_sec=round(elapsed, 2),
            logs=logs,
        )
