#!/usr/bin/env python3
"""
OCR性能基准测试程序
测试不同尺寸图片的识别速度和准确率
"""

import cv2
import os
import time
import numpy as np
import argparse
from pathlib import Path
from ocr import OCR


def create_test_images():
    """创建不同尺寸的测试图片"""
    print("🔄 创建测试图片...")
    
    test_dir = "test_images"
    Path(test_dir).mkdir(exist_ok=True)
    
    # 测试图片配置
    test_configs = [
        {"size": (300, 100), "text": "Small Image Test", "name": "small"},
        {"size": (600, 200), "text": "Medium Image OCR Test", "name": "medium"},
        {"size": (1200, 400), "text": "Large Image OCR Recognition Test", "name": "large"},
        {"size": (2400, 800), "text": "Very Large Image OCR Recognition Performance Test", "name": "xlarge"},
    ]
    
    created_images = []
    
    for config in test_configs:
        # 创建白色背景
        img = np.ones((config["size"][1], config["size"][0], 3), dtype=np.uint8) * 255
        
        # 添加文本
        text = config["text"]
        font_scale = min(config["size"][0] / 400, config["size"][1] / 100)
        thickness = max(1, int(font_scale * 2))
        
        # 计算文本位置（居中）
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = (config["size"][0] - text_width) // 2
        y = (config["size"][1] + text_height) // 2
        
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # 保存图片
        image_path = os.path.join(test_dir, f"{config['name']}_test.jpg")
        cv2.imwrite(image_path, img)
        created_images.append(image_path)
        
        print(f"   ✅ 创建: {image_path} ({config['size'][0]}x{config['size'][1]})")
    
    return created_images


def benchmark_ocr(image_paths, iterations=3):
    """OCR性能基准测试"""
    
    print("🔄 初始化OCR...")
    try:
        ocr = OCR()
        print("✅ OCR初始化成功")
    except Exception as e:
        print(f"❌ OCR初始化失败: {e}")
        return None
    
    results = []
    
    for image_path in image_paths:
        print(f"\n📷 测试图片: {os.path.basename(image_path)}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            continue
        
        h, w = image.shape[:2]
        print(f"📐 图片尺寸: {w}x{h}")
        
        # 多次测试取平均值
        times = []
        text_counts = []
        
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
            
            result = {
                'image_path': image_path,
                'image_size': f"{w}x{h}",
                'iterations': len(times),
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'avg_texts': avg_texts,
                'speed': speed,
                'times': times,
                'text_counts': text_counts
            }
            
            results.append(result)
            
            print(f"   📊 平均时间: {avg_time:.3f}秒")
            print(f"   📊 最快时间: {min_time:.3f}秒")
            print(f"   📊 最慢时间: {max_time:.3f}秒")
            print(f"   📊 平均文本: {avg_texts:.1f}")
            print(f"   📊 识别速度: {speed:.1f} 文本/秒")
    
    return results


def print_benchmark_summary(results):
    """打印基准测试总结"""
    
    if not results:
        print("❌ 没有测试结果")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 OCR性能基准测试总结")
    print(f"{'='*80}")
    
    print(f"{'图片尺寸':<15} {'平均时间':<10} {'识别速度':<12} {'平均文本':<10} {'测试次数':<8}")
    print("-" * 80)
    
    for result in results:
        size = result['image_size']
        avg_time = result['avg_time']
        speed = result['speed']
        avg_texts = result['avg_texts']
        iterations = result['iterations']
        
        print(f"{size:<15} {avg_time:.3f}秒{'':<4} {speed:.1f}文本/秒{'':<4} {avg_texts:.1f}{'':<6} {iterations}{'':<4}")
    
    # 总体统计
    total_images = len(results)
    total_time = sum(r['avg_time'] for r in results)
    total_texts = sum(r['avg_texts'] for r in results)
    overall_speed = total_texts / total_time if total_time > 0 else 0
    
    print("-" * 80)
    print(f"总体统计:")
    print(f"  测试图片: {total_images}")
    print(f"  总处理时间: {total_time:.3f}秒")
    print(f"  总识别文本: {total_texts:.1f}")
    print(f"  整体速度: {overall_speed:.1f} 文本/秒")
    print(f"  平均每张: {total_time/total_images:.3f}秒")


def main():
    parser = argparse.ArgumentParser(description="OCR性能基准测试")
    parser.add_argument('--images', nargs='+', help='测试图片路径（可选）')
    parser.add_argument('--iterations', '-i', type=int, default=3, help='每个图片的测试次数 (默认: 3)')
    parser.add_argument('--create-test', action='store_true', help='创建测试图片')
    
    args = parser.parse_args()
    
    print("🚀 OCR性能基准测试程序")
    print("=" * 60)
    
    # 检查模型文件
    model_dir = "models"
    required_files = ["det.onnx", "rec.onnx", "ocr.res"]
    
    print("🔍 检查模型文件...")
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} 缺失")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ 缺少模型文件: {missing_files}")
        print("请先运行: python download_models.py")
        return
    
    # 确定测试图片
    if args.create_test:
        image_paths = create_test_images()
    elif args.images:
        image_paths = args.images
    else:
        # 使用默认测试图片
        test_dir = "test_images"
        if os.path.exists(test_dir):
            image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        else:
            print("❌ 没有找到测试图片")
            print("使用 --create-test 创建测试图片，或使用 --images 指定图片路径")
            return
    
    if not image_paths:
        print("❌ 没有可用的测试图片")
        return
    
    print(f"\n📷 将测试 {len(image_paths)} 张图片，每张测试 {args.iterations} 次")
    
    # 运行基准测试
    results = benchmark_ocr(image_paths, args.iterations)
    
    # 打印总结
    print_benchmark_summary(results)
    
    print(f"\n🎉 基准测试完成!")


if __name__ == "__main__":
    main()
