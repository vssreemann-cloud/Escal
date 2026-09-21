"""
Core data structures representing resources, nodes, graphs, and simulation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel


@dataclass
class ResourceVector:
    llm_tokens_per_sec: float = 0.0
    gpu_memory_mb: float = 0.0
    tool_concurrency_slots: float = 0.0

    def __add__(self, other: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            llm_tokens_per_sec=self.llm_tokens_per_sec + other.llm_tokens_per_sec,
            gpu_memory_mb=self.gpu_memory_mb + other.gpu_memory_mb,
            tool_concurrency_slots=self.tool_concurrency_slots + other.tool_concurrency_slots,
        )

    def __sub__(self, other: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            llm_tokens_per_sec=max(0.0, self.llm_tokens_per_sec - other.llm_tokens_per_sec),
            gpu_memory_mb=max(0.0, self.gpu_memory_mb - other.gpu_memory_mb),
            tool_concurrency_slots=max(0.0, self.tool_concurrency_slots - other.tool_concurrency_slots),
        )

    def fits_in(self, capacity: "ResourceVector") -> bool:
        return (
            self.llm_tokens_per_sec <= capacity.llm_tokens_per_sec
            and self.gpu_memory_mb <= capacity.gpu_memory_mb
            and self.tool_concurrency_slots <= capacity.tool_concurrency_slots
        )


@dataclass
class GraphNode:
    node_id: str
    service_type: str  # 'llm_call', 'code_sandbox', 'web_search', etc.
    resource_req: ResourceVector
    est_duration_sec: float
    probability: float = 1.0
    children: List[Tuple[str, float]] = field(default_factory=list)  # [(child_node_id, branch_prob)]


@dataclass
class ServiceGraph:
    workflow_id: str
    nodes: Dict[str, GraphNode]
    entry_node_id: str
    priority: int = 1


class WorkflowRequest(BaseModel):
    workflow_id: str
    prompt_intent: str
    priority: int = 1


class SimulationResult(BaseModel):
    mode: str
    total_workflows: int
    successful_workflows: int
    failed_workflows: int
    rate_limit_errors: int
    avg_latency_sec: float
    logs: List[str]
