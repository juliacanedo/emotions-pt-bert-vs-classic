# Verifica a instalação do PyTorch e a disponibilidade da GPU CUDA

import torch, platform
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))