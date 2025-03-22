import os
import platform
import subprocess
import torch

def check_cuda_cudnn():
    """Check if CUDA and cuDNN are installed and available."""
    
    print("System information:")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    # Check CUDA availability with PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available (PyTorch): {cuda_available}")
    
    if cuda_available:
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        
        # Get device information
        device_count = torch.cuda.device_count()
        print(f"GPU devices: {device_count}")
        
        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Check cuDNN
        cudnn_enabled = torch.backends.cudnn.enabled
        print(f"cuDNN enabled: {cudnn_enabled}")
        
        if cudnn_enabled:
            cudnn_version = torch.backends.cudnn.version()
            print(f"cuDNN version: {cudnn_version}")
    
    # Additional CUDA info using nvidia-smi
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        print("\nNVIDIA-SMI output:")
        print(nvidia_smi.decode('utf-8'))
    except:
        print("\nNVIDIA-SMI not available or failed to execute")

if __name__ == "__main__":
    check_cuda_cudnn()