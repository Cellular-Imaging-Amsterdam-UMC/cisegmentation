from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from cisegmentation import MODEL_REGISTRY

print(
    f"torch={torch.__version__} cuda_runtime={torch.version.cuda} available={torch.cuda.is_available()}"
)
if torch.cuda.is_available():
    tensor = torch.ones((64, 64), device="cuda")
    print(f"gpu={torch.cuda.get_device_name(0)} sum={tensor.sum().item()}")
print(f"registered_models={len(MODEL_REGISTRY)}")
