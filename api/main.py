"""
FastAPI Application Entry Point: REST API Endpoints.
"""

from typing import List
from fastapi import FastAPI
from config import Config
from core.models import WorkflowRequest, SimulationResult
from core.engine import WorkloadSimulator

app = FastAPI(
    title="Runtime Service-Graph Controller API",
    description="Predictive Multi-Agent Resource Reservation System",
    version="1.0.0"
)

simulator = WorkloadSimulator()


@app.get("/")
def read_root():
    return {
        "status": "active",
        "system": "Runtime Service-Graph Controller (RSGC)",
        "capacity_limits": {
            "tokens_per_sec": Config.TOTAL_LLM_TOKENS_PER_SEC,
            "gpu_memory_mb": Config.TOTAL_GPU_MEMORY_MB,
            "tool_slots": Config.TOTAL_TOOL_SLOTS
        }
    }


@app.post("/simulate/naive", response_model=SimulationResult)
async def simulate_naive(requests: List[WorkflowRequest]):
    return await simulator.run_naive_simulation(requests)


@app.post("/simulate/rsgc", response_model=SimulationResult)
async def simulate_rsgc(requests: List[WorkflowRequest]):
    return await simulator.run_rsgc_simulation(requests)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
