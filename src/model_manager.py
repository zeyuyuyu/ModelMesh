import os
import logging
from typing import Dict, List, Optional, Union
import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    max_batch_size: int
    min_memory_mb: int
    max_memory_mb: int
    scaling_factor: float = 1.0

class ModelManager:
    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

    def register_model(self, model: torch.nn.Module, config: ModelConfig) -> None:
        """Register a model with its configuration for management."""
        self.models[config.name] = model.to(self.device)
        self.configs[config.name] = config
        self.logger.info(f'Registered model {config.name}')

    def auto_batch(self, model_name: str, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Automatically batch inputs based on model configuration and available resources."""
        if model_name not in self.models:
            raise KeyError(f'Model {model_name} not found')

        config = self.configs[model_name]
        total_samples = len(inputs)
        batch_size = min(total_samples, config.max_batch_size)

        # Adjust batch size based on available memory
        available_memory = torch.cuda.get_device_properties(self.device).total_memory
        memory_per_sample = config.min_memory_mb * 1024 * 1024  # Convert to bytes
        max_possible_batch = available_memory // memory_per_sample
        batch_size = min(batch_size, max_possible_batch)

        results = []
        for i in range(0, total_samples, batch_size):
            batch = torch.stack(inputs[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                output = self.models[model_name](batch)
            results.extend(output.cpu().split(1))

        return results

    def scale_model(self, model_name: str, scaling_factor: Optional[float] = None) -> None:
        """Dynamically scale model resources based on load or explicit scaling factor."""
        if model_name not in self.models:
            raise KeyError(f'Model {model_name} not found')

        config = self.configs[model_name]
        if scaling_factor is None:
            # Auto-determine scaling factor based on device utilization
            if torch.cuda.is_available():
                utilization = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                scaling_factor = 1.0 + (utilization - 0.5)  # Scale up/down based on 50% threshold

        config.scaling_factor = max(0.1, min(2.0, scaling_factor))  # Limit scaling range
        config.max_batch_size = int(config.max_batch_size * config.scaling_factor)
        
        self.logger.info(f'Scaled model {model_name} by factor {config.scaling_factor}')

    def get_model_stats(self, model_name: str) -> Dict[str, Union[int, float]]:
        """Get current statistics for a model."""
        if model_name not in self.models:
            raise KeyError(f'Model {model_name} not found')

        config = self.configs[model_name]
        return {
            'current_batch_size': config.max_batch_size,
            'scaling_factor': config.scaling_factor,
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
            'memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0
        }
