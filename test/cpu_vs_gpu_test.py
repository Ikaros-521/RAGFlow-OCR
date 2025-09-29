#!/usr/bin/env python3
"""
CPU vs GPU 性能对比测试
支持强制指定使用CPU或GPU进行对比
"""

import cv2
import time
import os
import numpy as np
import onnxruntime as ort
from ocr import OCR
import argparse


def force_cpu_mode():
    """强制使用CPU模式"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['ORT_DISABLE_CUDA'] = '1'
    os.environ['ORT_CUDA_PROVIDER_ONLY'] = '0'
    print("💻 环境变量设置: CUDA_VISIBLE__DEVICES=-1, ORT_DISABLE_CUDA=1")


def restore_gpu_mode():
    """恢复GPU模式"""
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
        del os.environ['CUDA_VISIBLE_DEVICES']
    if 'ORT_DISABLE_CUDA' in os.environ:
        del os.environ['ORT_DISABLE_CUDA']
    if 'ORT_CUDA_PROVIDER_ONLY' in os.environ:
        del os.environ['ORT_CUDA_PROVIDER_ONLY']
    print("🎯 环境变量设置: 恢复GPU支持")


def create_test_images():
    """创建不同尺寸的测试图片"""
    print("🔄 创建测试图片...")
    
    test_images = []
    
    # 小图片 (500x300)
    img1 = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img1, "Small Image Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite("test_small.jpg", img1)
    test_images.append(("test_small.jpg", "小图片 (500x300)", 500*300))
    
    # 中等图片 (1000x600)
    img2 = np.ones((600, 1000, 3), dtype=np.uint8) * 255
    texts2 = ["Medium Image Test", "OCR Performance", "CPU vs GPU Comparison"]
    for i, text in enumerate(texts2):
        cv2.putText(img2, text, (50, 150+i*80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite("test_medium.jpg", img2)
    test_images.append(("test_medium.jpg", "中等图片 (1000x600)", 1000*600))
    
    # 大图片 (1500x900)
    img3 = np.ones((900, 1500, 3), dtype=np.uint8) * 255
    texts3 = ["Large Image OCR Test", "GPU Acceleration Performance", "NVIDIA GeForce RTX 4090 D", 
              "Computer Vision Application", "Deep Learning Inference"]
    for i, text in enumerate(texts3):
        cv2.putText(img3, text, (50, 120+i*60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite("test_large.jpg", img3)
    test_images.append(("test_large.jpg", "大图片 (1500x900)", 1500*900))
    
    # 超大图片 (2000x1200)
    img4 = np.ones((1200, 2000, 3), dtype=np.uint8) * 255
    texts4 = ["Very Large Image OCR Test", "NVIDIA GeForce RTX 4090 D Graphics Card", 
              "ONNX Runtime CUDA Execution Provider", "High Performance Computing OCR",
              "Machine Learning Text Recognition", "GPU vs CPU Benchmark Analysis",
              "Computer Vision Deep Learning", "Optical Character Recognition System"]
    for i, text in enumerate(texts4):
        cv2.putText(img4, text, (50, 100+i*50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite("test_xlarge.jpg", img4)
    test_images.append(("test_xlarge.jpg", "超大图片 (2000x1200)", 2000*1200))
    
    # 4K图片 (3840x2160)
    img5 = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
    texts5 = ["4K Resolution OCR Performance Test", "NVIDIA GeForce RTX 4090 D Graphics Card",
              "ONNX Runtime CUDA Execution Provider", "High Performance Computing OCR Application",
              "Machine Learning Deep Learning Text Recognition", "Computer Vision AI Processing",
              "Graphics Processing Unit Parallel Computing", "Optical Character Recognition Neural Networks",
              "GPU Acceleration vs CPU Sequential Processing", "Ray Tracing Tensor Core Performance",
              "Memory Bandwidth Throughput Optimization", "Parallel Processing Pipeline Architecture",
              "Deep Neural Network Inference Acceleration", "Real-time Computer Vision Applications",
              "Video Processing Image Recognition Systems", "Training Dataset Evaluation Performance",
              "Algorithm Optimization Machine Learning Models", "Pattern Recognition Feature Extraction",
              "Edge Detection Contour Analysis Techniques", "Text Detection Bounding Box Localization",
              "Character Segmentation Recognition Accuracy", "Multi-language OCR Support Implementation",
              "Handwriting Recognition Signature Verification", "Document Analysis Layout Understanding",
              "Natural Language Processing Information Extraction", "Knowledge Base Construction Indexing"]
    font_scale = 2.5
    thickness = 3
    line_height = 70
    
    current_y = 120
    for i, text in enumerate(texts5):
        if current_y + line_height > 2000:  # 留出底部边距
            break
        cv2.putText(img5, text, (120, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        current_y += line_height
    
    # 添加一些图形元素增加复杂度
    for i in range(5):
        x = 120 + i * 740
        y = current_y + 80
        cv2.rectangle(img5, (x, y), (x+700, y+120), (0, 0, 0), 3)
        cv2.putText(img5, f"4K Region {i+1}", (x+30, y+75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    cv2.imwrite("test_4k.jpg", img5)
    test_images.append(("test_4k.jpg", "4K图片 (3840x2160)", 3840*2160))
    
    print(f"✅ 创建了 {len(test_images)} 张测试图片")
    return test_images


def test_mode_performance(image_path, mode="GPU", iterations=3):
    """测试指定模式的性能"""
    
    print(f"\n{'='*50}")
    print(f"🔥 {mode} 模式性能测试")
    print(f"🎯 图片: {os.path.basename(image_path)}")
    
    # 强制重新加载模型，确保使用正确的执行提供程序
    import gc
    import importlib
    try:
        # 清理已加载的模型缓存
        from ocr.ocr import loaded_models
        loaded_models.clear()
        
        # 重新加载ocr模块，强制重新初始化
        if 'ocr.ocr' in importlib.sys.modules:
            importlib.reload(importlib.import_module('ocr.ocr'))
        
        gc.collect()  # 强制垃圾回收
        print("🗑️ 强制重新加载模型模块")
    except Exception as e:
        print(f"⚠️ 重新加载模型时出现问题: {e}")
        gc.collect()
    
    # 设置执行模式（在重新加载之后设置）
    if mode == "CPU":
        force_cpu_mode()
        print("💻 强制使用 CPU 模式")
    else:
        restore_gpu_mode()
        print("🎯 使用 GPU 模式 (如果可用)")
    
    # 验证提供程序可用性
    print("🔍 验证当前模式环境...")
    providers = ort.get_available_providers()
    if mode == "CPU":
        print(f"   CPU模式: {[p for p in providers if 'CPU' in p]}")
    else:
        gpu_providers = [p for p in providers if 'CUDA' in p or 'Tensorrt' in p]
        print(f"   GPU提供程序: {gpu_providers}")
        if not gpu_providers:
            print("⚠️ 警告: 没有检测到GPU提供程序，将使用CPU")
    
    # 初始化OCR
    print("🔄 初始化OCR...")
    try:
        ocr = OCR()
        print("✅ OCR初始化成功")
    except Exception as e:
        print(f"❌ OCR初始化失败: {e}")
        return None
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None
    
    h, w = image.shape[:2]
    pixels = w * h
    print(f"📐 图片尺寸: {w}x{h} ({pixels/1000000:.1f} MPix)")
    
    # 预热
    print("🔥 预热中...")
    try:
        ocr(image)
        print("✅ 预热完成")
    except Exception as e:
        print(f"❌ 预热失败: {e}")
        return None
    
    # 性能测试
    print(f"⚡ 开始 {iterations} 次性能测试...")
    times = []
    text_counts = []
    
    for i in range(iterations):
        print(f"   第 {i+1}/{iterations} 次...", end=" ")
        
        start_time = time.time()
        try:
            result = ocr(image)
            end_time = time.time()
            
            process_time = end_time - start_time
            times.append(process_time)
            text_counts.append(len(result))
            
            print(f"✅ {process_time:.3f}秒 ({len(result)} 文本)")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            continue
    
    if not times:
        print("❌ 没有成功的测试结果")
        return None
    
    # 计算统计结果
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_texts = sum(text_counts) / len(text_counts)
    speed = avg_texts / avg_time if avg_time > 0 else 0
    megapix_per_sec = pixels / (avg_time * 1000000)
    
    result = {
        'mode': mode,
        'image_path': image_path,
        'image_size': f"{w}x{h}",
        'pixels': pixels,
        'iterations': len(times),
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_texts': avg_texts,
        'speed': speed,
        'megapixels_per_sec': megapix_per_sec,
        'times': times,
        'text_counts': text_counts
    }
    
    print(f"\n📊 {mode} 性能统计:")
    print(f"   平均时间: {avg_time:.3f}秒")
    print(f"   最快时间: {min_time:.3f}秒") 
    print(f"   最慢时间: {max_time:.3f}秒")
    print(f"   平均文本: {avg_texts:.1f}")
    print(f"   识别速度: {speed:.1f} 文本/秒")
    print(f"   图片处理: {megapix_per_sec:.1f} MPix/秒")
    
    return result


def print_comparison_results(results, test_images_info):
    """打印对比结果"""
    
    print(f"\n{'='*80}")
    print("📊 CPU vs GPU 性能对比总结")
    print(f"{'='*80}")
    
    # 按图片尺寸分组显示结果
    for img_path, img_desc, pixels in test_images_info:
        cpu_result = None
        gpu_result = None
        
        for result in results:
            if result and result['image_path'] == img_path:
                if result['mode'] == 'CPU':
                    cpu_result = result
                else:
                    gpu_result = result
        
        if cpu_result and gpu_result:
            print(f"\n🎯 {img_desc} ({pixels/1000000:.1f} MPix)")
            print("-" * 60)
            
            # 计算性能提升
            time_speedup = cpu_result['avg_time'] / gpu_result['avg_time'] if gpu_result['avg_time'] > 0 else 0
            speed_speedup = gpu_result['speed'] / cpu_result['speed'] if cpu_result['speed'] > 0 else 0
            pix_speedup = gpu_result['megapixels_per_sec'] / cpu_result['megapixels_per_sec'] if cpu_result['megapixels_per_sec'] > 0 else 0
            
            print(f"{'指标':<20} {'CPU':<12} {'GPU':<12} {'提升':<8}")
            print("-" * 60)
            print(f"{'平均处理时间':<20} {cpu_result['avg_time']:.3f}秒{'':<3} {gpu_result['avg_time']:.3f}秒{'':<3} {time_speedup:.1f}x")
            print(f"{'识别速度':<20} {cpu_result['speed']:.1f}文本/秒{'':<1} {gpu_result['speed']:.1f}文本/秒{'':<1} {speed_speedup:.1f}x")
            print(f"{'图片处理速度':<20} {cpu_result['megapixels_per_sec']:.1f}MPix/s{'':<2} {gpu_result['megapixels_per_sec']:.1f}MPix/s{'':<2} {pix_speedup:.1f}x")
            
            # 性能评估
            if time_speedup > 1.5:
                print("🎉 GPU 加速效果显著!")
            elif time_speedup > 1.1:
                print("✅ GPU 有一定加速效果")
            elif time_speedup > 0.9:
                print("⚖️ CPU 和 GPU 性能相近")
            else:
                print("⚠️ CPU 性能更好，可能需要GPU优化")


def cleanup_test_images(test_images_info):
    """清理测试图片"""
    print(f"\n🗑️ 清理测试文件...")
    for img_path, _, _ in test_images_info:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"   已删除: {img_path}")


def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU 性能对比测试")
    parser.add_argument('--iterations', '-i', type=int, default=3, help='每个测试的重复次数 (默认: 3)')
    parser.add_argument('--keep-images', action='store_true', help='保留测试图片')
    
    args = parser.parse_args()
    
    print("🚀 CPU vs GPU OCR 性能对比测试")
    print("=" * 60)
    
    # 检查环境
    print("🔍 检查环境支持...")
    providers = ort.get_available_providers()
    print("ONNX Runtime 执行提供程序:")
    for provider in providers:
        if "CUDA" in provider:
            print(f"  🎯 {provider}")
        elif "Tensorrt" in provider:
            print(f"  🔧 {provider}")
        else:
            print(f"  💻 {provider}")
    
    has_cuda = 'CUDAExecutionProvider' in providers
    if not has_cuda:
        print("⚠️ 警告: 没有检测到 CUDA 支持，GPU 测试可能使用 CPU")
    
    print()
    
    # 创建测试图片
    test_images_info = create_test_images()
    
    try:
        all_results = []
        
        # 测试每种图片的两种模式
        for img_path, img_desc, pixels in test_images_info:
            print(f"\n{'#'*60}")
            print(f"🎯 测试: {img_desc}")
            print(f"{'#'*60}")
            
            # CPU 测试
            cpu_result = test_mode_performance(img_path, "CPU", args.iterations)
            all_results.append(cpu_result)
            
            # 测试间隔，确保模型完全卸载
            print("⏳ 测试间隔，等待模型卸载...")
            import time
            time.sleep(1)
            
            # GPU 测试
            gpu_result = test_mode_performance(img_path, "GPU", args.iterations)
            all_results.append(gpu_result)
            
            # 恢复GPU模式（确保后续测试正常）
            restore_gpu_mode()
            
            # 每个图片测试完成后再次清理和重新加载
            print("🧹 本轮测试完成，重新加载模块...")
            import gc
            import importlib
            try:
                # 重新加载整个ocr模块
                import ocr
                if hasattr(ocr, '__dict__'):
                    ocr_dict = ocr.__dict__.copy()
                    for name in list(importlib.sys.modules.keys()):
                        if name.startswith('ocr.'):
                            del importlib.sys.modules[name]
                gc.collect()
            except Exception as e:
                print(f"⚠️ 重新加载OCR模块出现问题: {e}")
                gc.collect()
        
        # 显示对比结果
        print_comparison_results(all_results, test_images_info)
        
        # 清理
        if not args.keep_images:
            cleanup_test_images(test_images_info)
        
        print(f"\n🎉 性能对比测试完成!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 测试被用户中断")
        cleanup_test_images(test_images_info)
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        cleanup_test_images(test_images_info)
        raise


if __name__ == "__main__":
    main()
