"""
Service Graph Predictor: Analyzes agent task intentions and generates probabilistic DAGs.
"""

from core.models import ServiceGraph, GraphNode, ResourceVector


class ExecutionGraphPredictor:
    """Predicts execution service-graphs based on task metadata and intention keywords."""

    @staticmethod
    def predict_graph(workflow_id: str, prompt_intent: str, priority: int = 1) -> ServiceGraph:
        intent_lower = prompt_intent.lower()
        nodes = {}

        # Root Planning LLM Node
        nodes["plan"] = GraphNode(
            node_id="plan",
            service_type="llm_planner",
            resource_req=ResourceVector(llm_tokens_per_sec=100.0, gpu_memory_mb=1024.0),
            est_duration_sec=1.0,
            probability=1.0,
            children=[("web_search_branch", 0.7), ("code_exec_branch", 0.3)]
        )

        if "code" in intent_lower or "python" in intent_lower or "debug" in intent_lower:
            # Shift probabilities toward code sandbox
            nodes["plan"].children = [("web_search_branch", 0.2), ("code_exec_branch", 0.8)]

        # Branch 1: Web Search & Summarization
        nodes["web_search_branch"] = GraphNode(
            node_id="web_search_branch",
            service_type="web_search",
            resource_req=ResourceVector(tool_concurrency_slots=1.0),
            est_duration_sec=2.0,
            probability=0.7,
            children=[("summarize_results", 1.0)]
        )

        nodes["summarize_results"] = GraphNode(
            node_id="summarize_results",
            service_type="llm_call",
            resource_req=ResourceVector(llm_tokens_per_sec=120.0, gpu_memory_mb=1024.0),
            est_duration_sec=1.5,
            probability=0.7,
            children=[]
        )

        # Branch 2: Python Code Execution Sandbox
        nodes["code_exec_branch"] = GraphNode(
            node_id="code_exec_branch",
            service_type="code_sandbox",
            resource_req=ResourceVector(gpu_memory_mb=2048.0, tool_concurrency_slots=1.0),
            est_duration_sec=2.5,
            probability=0.3,
            children=[]
        )

        return ServiceGraph(
            workflow_id=workflow_id,
            nodes=nodes,
            entry_node_id="plan",
            priority=priority
        )
