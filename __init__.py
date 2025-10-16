from .civitai_lora_loader import CivitAI_LORA_Loader
from .civitai_checkpoint_loader import CivitAI_Checkpoint_Loader
from .fast_civitai_lora_loader import CivitAI_Fast_LORA_Loader

NODE_CLASS_MAPPINGS = {
    "CivitAI_Lora_Loader": CivitAI_LORA_Loader,
    "CivitAI_Checkpoint_Loader": CivitAI_Checkpoint_Loader,
    "CivitAI_Fast_Lora_Loader": CivitAI_Fast_LORA_Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitAI_Lora_Loader": "CivitAI xereca Loader",
    "CivitAI_Checkpoint_Loader": "CivitAI Checkpoint Loader",
    "CivitAI_Fast_Lora_Loader": "CivitAI Fast LORA",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
