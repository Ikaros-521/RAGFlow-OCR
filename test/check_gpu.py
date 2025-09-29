#!/usr/bin/env python3
"""
检查GPU支持情况
"""

def check_gpu_support():
    print("🔍 检查GPU支持情况...")
    print("=" * 50)
    
    # 检查PyTorch
    try:
        import torch
        print("✅ PyTorch 已安装")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   设备 {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   ❌ CUDA 不可用")
    except ImportError:
        print("❌ PyTorch 未安装")
    
    print()
    
    # 检查ONNX Runtime
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime 已安装")
        providers = ort.get_available_providers()
        print("   可用执行提供程序:")
        for provider in providers:
            print(f"     - {provider}")
        
        if 'CUDAExecutionProvider' in providers:
            print("   ✅ CUDA 执行提供程序可用")
        else:
            print("   ❌ CUDA 执行提供程序不可用")
            
        if 'DmlExecutionProvider' in providers:
            print("   ✅ DirectML 执行提供程序可用 (Windows GPU)")
        else:
            print("   ❌ DirectML 执行提供程序不可用")
            
    except ImportError:
        print("❌ ONNX Runtime 未安装")
    
    print()
    
    # 检查CUDA版本
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU 驱动已安装")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ NVIDIA GPU 驱动未安装或不可用")
    except FileNotFoundError:
        print("❌ nvidia-smi 命令不可用")
    
    print()
    print("=" * 50)
    print("💡 启用GPU加速的建议:")
    print("1. 安装 onnxruntime-gpu: pip install onnxruntime-gpu")
    print("2. 安装 PyTorch CUDA 版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. 确保 NVIDIA 驱动和 CUDA 已正确安装")

if __name__ == "__main__":
    check_gpu_support()
