import os
import gc
import torch
import logging
from typing import Dict, Optional

class ModelManager:
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_memory_usage: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    def get_model_memory_gb(self, model: torch.nn.Module) -> float:
        """Estimate memory usage of a PyTorch model in GB"""
        mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        return (mem_params + mem_bufs) / 1024**3

    def get_available_memory_gb(self) -> float:
        """Get available GPU memory if CUDA is available, otherwise system RAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            import psutil
            return psutil.virtual_memory().available / 1024**3

    def load_model(self, model_id: str, model_class: torch.nn.Module, **kwargs) -> Optional[torch.nn.Module]:
        """Load a model while respecting memory constraints"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        try:
            model = model_class(**kwargs)
            model_size = self.get_model_memory_gb(model)

            # Check if loading this model would exceed memory limit
            current_usage = sum(self.model_memory_usage.values())
            if current_usage + model_size > self.max_memory_gb:
                self.free_memory(model_size)

            if torch.cuda.is_available():
                model = model.cuda()

            self.loaded_models[model_id] = model
            self.model_memory_usage[model_id] = model_size
            self.logger.info(f'Successfully loaded model {model_id}, size: {model_size:.2f}GB')
            return model

        except Exception as e:
            self.logger.error(f'Failed to load model {model_id}: {str(e)}')
            return None

    def free_memory(self, required_memory: float):
        """Free memory by unloading least recently used models"""
        models_to_unload = []
        freed_memory = 0

        for model_id, memory_usage in sorted(
            self.model_memory_usage.items(),
            key=lambda x: x[1]  # Sort by memory usage
        ):
            models_to_unload.append(model_id)
            freed_memory += memory_usage
            if freed_memory >= required_memory:
                break

        for model_id in models_to_unload:
            self.unload_model(model_id)

    def unload_model(self, model_id: str):
        """Unload a model and free its memory"""
        if model_id in self.loaded_models:
            model = self.loaded_models[model_id]
            if torch.cuda.is_available():
                model.cpu()
            del self.loaded_models[model_id]
            del self.model_memory_usage[model_id]
            del model
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info(f'Unloaded model {model_id}')

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Get a loaded model by ID"""
        return self.loaded_models.get(model_id)

    def get_loaded_models(self) -> Dict[str, float]:
        """Get dictionary of loaded model IDs and their memory usage"""
        return self.model_memory_usage.copy()
