import os
import torch
import torchvision
import torchaudio
import subprocess

# Check if CUDA (GPU) support is available
cuda_available = torch.cuda.is_available()

# Get the CUDA version used by PyTorch
cuda_version = torch.version.cuda

# Check if PyTorch is installed
pytorch_installed = torch.__version__ is not None

if cuda_available:
    # Get the number of available GPU devices
    num_gpu_devices = torch.cuda.device_count()

    # Get the name of each available GPU device
    gpu_device_names = [torch.cuda.get_device_name(i) for i in range(num_gpu_devices)]

    print(f"CUDA (GPU) support is available.")
    print(f"Number of GPU devices: {num_gpu_devices}")
    print(f"GPU device names: {gpu_device_names}")
else:
    print("CUDA (GPU) support is not available. Please install CUDA and CUDNN.")

print(f"PyTorch installed: {pytorch_installed}")
print(f"CUDA support available: {cuda_available}")
print(f"CUDA version used by PyTorch: {cuda_version}")

# Check if packages are installed and print their versions
print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"torchaudio version: {torchaudio.__version__}")

# Check if GPU is available
cuda_available = torch.cuda.is_available()

# Check if transformers is installed
try:
    import transformers
    print("transformers is installed.")
    print(f"Version: {transformers.__version__}")
except ImportError:
    print("transformers is not installed. Please install it.")

# Check CUDA_PATH and CUDNN
cuda_path = os.environ.get('CUDA_PATH')
cudnn_path = os.environ.get('CUDNN')

if cuda_path and cudnn_path:
    print(f"CUDA_PATH: {cuda_path}")
    print(f"CUDNN: {cudnn_path}")
    user_input = input("Is the PATH correct? (yes/no): (Exemple: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\ v**.*)")
    if user_input.lower() != "yes":
        print("Please correct the PATH and rerun the program.")
        exit(0)
else:
    print("Please set CUDA_PATH and CUDNN in the environment variables and rerun the program.")
    exit(0)

# Install the python dependencies
packages = [
    "bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui",
    "git+https://github.com/huggingface/transformers.git",
    "accelerate",
    "einops",
    "sentencepiece",
    "auto-gptq", #For CUDA 12.1: pip install auto-gptq OR For CUDA 11.8: pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
    "git+https://github.com/huggingface/peft.git",
    "git+https://github.com/huggingface/optimum.git",
]

for package in packages:
    try:
        subprocess.check_call(["pip", "install", "--no-cache-dir", "--upgrade" , package])
        print(f"Successfully installed/upgraded {package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install/upgrade {package}")




