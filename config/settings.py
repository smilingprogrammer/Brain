from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash-exp"

    # Memory
    working_memory_capacity: int = 7
    hippocampus_size: int = 100000

    # Neural parameters
    default_neuron_type: str = "lif"
    spike_threshold: float = 1.0

    # System
    log_level: str = "INFO"
    enable_metrics: bool = True
    event_bus_type: str = "asyncio"

    # Reasoning
    reasoning_timeout: float = 180.0
    parallel_reasoning_paths: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()