import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
else:
    print("CUDA is NOT available. This could be because:")
    print("1. You don't have an NVIDIA GPU.")
    print("2. NVIDIA drivers are not installed.")
    print("3. PyTorch was installed without CUDA support (though your version says +cu121).")

try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("Ultralytics is not installed.")
