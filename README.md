# CUDA 12.1 & PyTorch (CUDA 12.1) Installation Guide

## Prerequisites
- **Operating System**: Windows 10/11 (64-bit)
- **NVIDIA GPU**: Check [CUDA-Capable GPUs](https://developer.nvidia.com/cuda-gpus)
- **NVIDIA Driver**: Ensure the latest driver is installed ([Download](https://www.nvidia.com/download/index.aspx))

## Step 1: Install CUDA 12.1
1. **Download CUDA Toolkit 12.1** from the official [NVIDIA website](https://developer.nvidia.com/cuda-12-1-0-download-archive).
2. Select **Windows x86_64**, choose the **Local Installer**, and download the `.exe` file.
3. Run the installer and select **Express Installation**.
4. Restart your system after installation.
5. Verify CUDA installation by running:
   ```powershell
   nvcc --version
   ```
   Expected output (example):
   ```
   nvcc: NVIDIA (R) Cuda compiler driver
   Cuda compilation tools, release 12.1, V12.1.66
   ```

## Step 2: Install cuDNN (Optional but Recommended)
1. Download cuDNN for CUDA 12.1 from [NVIDIA Developer Zone](https://developer.nvidia.com/cudnn).
2. Extract the files and copy them to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`.

## Step 3: Install PyTorch with CUDA 12.1 Support
1. Uninstall any previous PyTorch version:
   ```powershell
   pip uninstall torch torchvision torchaudio
   ```
2. Install PyTorch with CUDA 12.1:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## Step 4: Verify PyTorch Installation
Run the following Python script:
```python
import torch
print("cuDNN Available:",torch.backends.cudnn.is_available())
print("cuDNN Version:",torch.backends.cudnn.version())
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Torch Version:", torch.__version__)
print("GPU Count:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")
```
Expected output:
```
cuDNN Available: 90100
cuDNN Version: True
CUDA Available: True
CUDA Version: 12.1
Torch Version: 2.5.1+cu121
GPU Count: 1
GPU Name: NVIDIA GeForce GTX XXXX
```

## Troubleshooting
### 1. `CUDA Available: False`
- Ensure you installed the correct PyTorch version (`cu121`).
- Check GPU detection with:
  ```powershell
  nvidia-smi
  ```
  If no GPU is listed, update your NVIDIA driver.
- Reinstall PyTorch with:
  ```powershell
  pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### 2. `nvcc --version` Shows CUDA 12.1, But `nvidia-smi` Shows a Different Version
- CUDA toolkit and driver versions can differ. If `nvidia-smi` shows an older CUDA version, update your driver.

### 3. PyTorch Still Not Using GPU
- Ensure CUDA paths are added to environment variables:
  - Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin` to `PATH`.
  - Restart the system.

### 4. `ImportError: DLL Load Failed`
- Ensure `torch` and `torchvision` versions match:
  ```powershell
  pip list | findstr torch
  ```
- If mismatch occurs, reinstall them using:
  ```powershell
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

## Additional Resources
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)

---
**Author:** Gurnoor Singh Saini

**Last Updated:** February 16, 2025