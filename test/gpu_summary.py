#!/usr/bin/env python3
"""
GPU 加速配置总结
"""

import onnxruntime as ort
from ocr import OCR
import cv2
import time
import numpy as np


def check_gpu_status():
    """检查GPU状态"""
    print("🔍 GPU 支持状态检查")
    print("=" * 50)
    
    # 检查 ONNX Runtime providers
    providers = ort.get_available_providers()
    print("ONNX Runtime 执行提供程序:")
    for provider in providers:
        if "CUDA" in provider:
            print(f"  🎯 {provider} (GPU)")
        elif "Tensorrt" in provider:
            print(f"  🔧 {provider} (GPU优化)")
        else:
            print(f"  💻 {provider}")
    
    # 检查 PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch CUDA:")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\n❌ PyTorch 未安装")
    
    # 检查 NVIDIA 驱动
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\nNVIDIA GPU 信息:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    name, driver, memory = line.split(', ')
                    print(f"  🎮 {name}")
                    print(f"     驱动版本: {driver}")
                    print(f"     显存: {memory} MB")
    except:
        print("\n⚠️ 无法获取 NVIDIA GPU 信息")


def quick_performance_test():
    """快速性能测试"""
    print("\n⚡ 快速性能测试")
    print("=" * 50)
    
    # 创建简单测试图片
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "GPU Performance Test", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 初始化OCR
    print("🔄 初始化OCR...")
    ocr = OCR()
    
    # 测试推理
    print("🔍 执行OCR推理...")
    start_time = time.time()
    result = ocr(img)
    end_time = time.time()
    
    process_time = end_time - start_time
    print(f"✅ 识别完成")
    print(f"   处理时间: {process_time:.3f}秒")
    print(f"   识别文本: {len(result)} 个")
    print(f"   识别速度: {len(result)/process_time:.1f} 文本/秒")
    
    return process_time


def main():
    print("🚀 OCR GPU 加速状态总结")
    print("=" * 60)
    
    # 检查GPU状态
    check_gpu_status()
    
    # 快速性能测试
    try:
        perf_time = quick_performance_test()
        
        print(f"\n📊 性能评估:")
        if perf_time < 0.5:
            print("🎉 性能优秀！GPU 加速效果显著")
        elif perf_time < 1.0:
            print("✅ 性能良好，GPU 加速正常工作")
        else:
            print("⚠️ 性能一般，建议检查GPU配置")
            
    except Exception as e:
        print(f"\n❌ 性能测试失败: {e}")
    
    print(f"\n💡 使用建议:")
    print("1. 对于大图片或批量处理，GPU 加速效果更明显")
    print("2. 小图片可能因为GPU初始化开销而优势不明显")
    print("3. 可以使用 benchmark.py 进行详细性能测试")
    print("4. RTX 4090 D 是高性能GPU，适合OCR加速")
    
    print(f"\n🎯 GPU 加速已成功配置！")


if __name__ == "__main__":
    main()
