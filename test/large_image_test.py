#!/usr/bin/env python3
"""
大图片GPU加速测试
"""

import cv2
import time
import numpy as np
import os
from ocr import OCR


def create_large_test_image():
    """创建大尺寸测试图片"""
    
    # 创建一个更大的图片 (2000x1500)
    img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255
    
    # 添加更多文本
    texts = [
        "Large Image GPU Performance Test",
        "NVIDIA GeForce RTX 4090 D Graphics Card",
        "ONNX Runtime CUDA Execution Provider",
        "Deep Learning Inference Acceleration", 
        "文本识别GPU加速性能测试",
        "中文字符识别准确率测试",
        "OCR光学字符识别技术",
        "人工智能文字识别系统",
        "Machine Learning Text Detection",
        "Computer Vision Applications",
        "Neural Network Optimization",
        "High Performance Computing",
        "Parallel Processing Technology",
        "Graphics Processing Unit",
        "Tensor Processing Operations"
    ]
    
    font_scale = 1.2
    thickness = 2
    
    # 在多个位置放置文本
    for i, text in enumerate(texts):
        x = 50 + (i % 3) * 600
        y = 100 + (i // 3) * 100
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    # 添加一些图形元素
    for i in range(10):
        x = 100 + i * 180
        y = 800 + (i % 2) * 200
        cv2.rectangle(img, (x, y), (x+150, y+80), (0, 0, 0), 2)
        cv2.putText(img, f"Box {i+1}", (x+10, y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 保存图片
    large_image_path = "large_test_image.jpg"
    cv2.imwrite(large_image_path, img)
    return large_image_path


def test_performance(image_path, iterations=3):
    """测试性能"""
    
    print(f"📖 读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片")
        return None
    
    h, w = image.shape[:2]
    print(f"📐 图片尺寸: {w}x{h}")
    
    # 初始化OCR
    print("🔄 初始化OCR...")
    ocr = OCR()
    
    # 预热
    print("🔥 预热...")
    try:
        ocr(image)
        print("✅ 预热完成")
    except Exception as e:
        print(f"❌ 预热失败: {e}")
        return None
    
    # 多次测试
    times = []
    text_counts = []
    
    print(f"⚡ 开始 {iterations} 次性能测试...")
    
    for i in range(iterations):
        print(f"   第 {i+1}/{iterations} 次测试...", end=" ")
        
        start_time = time.time()
        try:
            result = ocr(image)
            end_time = time.time()
            
            process_time = end_time - start_time
            times.append(process_time)
            text_counts.append(len(result))
            
            print(f"✅ {process_time:.3f}秒, {len(result)} 文本")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            continue
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_texts = sum(text_counts) / len(text_counts)
        speed = avg_texts / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 性能统计:")
        print(f"   平均时间: {avg_time:.3f}秒")
        print(f"   最快时间: {min_time:.3f}秒")
        print(f"   最慢时间: {max_time:.3f}秒")
        print(f"   平均文本: {avg_texts:.1f}")
        print(f"   识别速度: {speed:.1f} 文本/秒")
        print(f"   图片处理: {w*h/(avg_time*1000000):.1f} MPix/秒")
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'avg_texts': avg_texts,
            'speed': speed,
            'image_size': f"{w}x{h}",
            'mpix_per_sec': w*h/(avg_time*1000000)
        }
    
    return None


def main():
    print("🚀 大图片GPU加速性能测试")
    print("=" * 60)
    
    # 检查GPU状态
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print("ONNX Runtime 执行提供程序:")
    for provider in providers:
        icon = "🎯" if "CUDA" in provider else "🔧" if "Tensorrt" in provider else "💻"
        print(f"  {icon} {provider}")
    
    print()
    
    # 创建大图片
    print("🔄 创建大尺寸测试图片...")
    image_path = create_large_test_image()
    print(f"✅ 大图片已创建: {image_path}")
    
    try:
        # 测试性能
        result = test_performance(image_path, iterations=3)
        
        if result:
            print(f"\n🎉 测试完成!")
            print(f"📈 对于 {result['image_size']} 的图片:")
            print(f"   平均处理时间: {result['avg_time']:.3f}秒")
            print(f"   识别速度: {result['speed']:.1f} 文本/秒")
            print(f"   图片处理速度: {result['mpix_per_sec']:.1f} MPix/秒")
            
            # 性能评估
            if result['avg_time'] < 1.0:
                print("⚡ 性能优秀！")
            elif result['avg_time'] < 2.0:
                print("✅ 性能良好")
            else:
                print("⚠️ 性能一般，可能需要优化")
    
    finally:
        # 清理
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"🗑️ 已清理临时文件: {image_path}")


if __name__ == "__main__":
    main()
