from dataclasses import dataclass


@dataclass
class EngineConfig:
    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = -1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    verbose: bool = False


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8
    max_waiting_queue_size: int = 100
    preemption_enabled: bool = True
    priority_decay_constant: float = 60.0


@dataclass
class MemoryConfig:
    total_pages: int = 256
    page_size_tokens: int = 16


@dataclass
class HeliosConfig:
    engine: EngineConfig
    scheduler: SchedulerConfig
    memory: MemoryConfig