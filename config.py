"""
Configuration settings for Cluster Capacity and System Parameters.
"""

class Config:
    # Cluster Resource Limits
    TOTAL_LLM_TOKENS_PER_SEC: float = 250.0  # Peak tokens/sec across LLM providers
    TOTAL_GPU_MEMORY_MB: float = 8192.0      # Total GPU VRAM pool
    TOTAL_TOOL_SLOTS: float = 3.0            # Max concurrent external tool sandboxes

    # Ledger Time Parameters
    LOOKAHEAD_WINDOW_SEC: float = 10.0       # Horizon length to predict
    SLOT_SIZE_SEC: float = 0.5               # Granularity of time slots (500ms)

    # Server Settings
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
