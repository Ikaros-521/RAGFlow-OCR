#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR并发性能测试程序
测试多线程/多进程并发处理的性能，统计显存占用和处理速度
支持GPU显存监控和时间统计
"""

import cv2
import os
import time
import numpy as np
import argparse
import threading
import queue
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from ocr import OCR
import gc
import sys

# 设置控制台输出编码 - 解决Windows emoji显示问题
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class GPUMonitor:
    """GPU显存监控类"""
    
    def __init__(self):
        self.gpu_available = False
        self.initialized = False
        self.gpu_info = {}
        
        # self._initialize_gpu_monitor()
    
    def _initialize_gpu_monitor(self):
        """初始化GPU监控"""
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_info = {
                    'device_count': torch.cuda.device_count(),
                    'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    'device_memory': [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
                }
                print("GPU监控初始化成功")
            else:
                print("CUDA不可用，将使用模拟监控")
        except ImportError:
            print("PyTorch未安装，将使用模拟监控")
        
        self.initialized = True
    
    def get_memory_info(self, device_id=0):
        """获取GPU显存信息"""
        if not self.initialized:
            return None
            
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_id)
                    
                    # 重置最大内存使用量统计
                    torch.cuda.reset_peak_memory_stats(device_id)
                    
                    allocated = torch.cuda.memory_allocated(device_id)
                    cached = torch.cuda.memory_reserved(device_id)
                    max_allocated = torch.cuda.max_memory_allocated(device_id)
                    capacity = torch.cuda.get_device_properties(device_id).total_memory
                    
                    return {
                        'allocated': allocated,
                        'cached': cached,
                        'max_allocated': max_allocated,
                        'capacity': capacity,
                        'free': capacity - allocated,
                        'utilization': allocated / capacity if capacity > 0 else 0
                    }
            except Exception as e:
                print(f"获取GPU内存信息失败: {e}")
                return None
        else:
            # 模拟GPU内存信息
            return {
                'allocated': 0,
                'cached': 0,
                'max_allocated': 0,
                'capacity': 8589934592,  # 8GB模拟
                'free': 8589934592,
                'utilization': 0
            }
    
    def format_memory(self, bytes_value):
        """格式化内存大小显示"""
        if bytes_value == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(bytes_value)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
            
        return f"{size:.2f} {units[unit_index]}"


class ConcurrentOCRBenchmark:
    """并发OCR基准测试类"""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.test_images = []
        self.results = []
        
    def create_test_images(self, batch_size=10):
        """创建多个测试图片"""
        print("创建并发测试图片...")
        
        test_dir = "concurrent_test_images"
        Path(test_dir).mkdir(exist_ok=True)
        
        # 创建不同复杂度的测试图片 - 全部使用1080P分辨率
        image_configs = [
            {"size": (1920, 1080), "text": "1080P简单文本测试", "complexity": "simple"},
            {"size": (1920, 1080), "text": "OCR并发性能测试", "complexity": "medium"},
            {"size": (1920, 1080), "text": "高速并行处理测试", "complexity": "complex"},
            {"size": (1920, 1080), "text": "深度学习文本识别系统", "complexity": "complex"},
        ]
        
        base_images = []
        for i, config in enumerate(image_configs):
            img_path = os.path.join(test_dir, f"test_base_{i}.jpg")
            self._create_image(img_path, config)
            base_images.append(img_path)
        
        # 为每个基础图片创建多个变体用于并发测试
        for i, base_img_path in enumerate(base_images):
            for j in range(batch_size):
                source_img = cv2.imread(base_img_path)
                if source_img is not None:
                    # 添加噪声或小变换增加处理差异
                    noise = np.random.randint(0, 50, source_img.shape, dtype=np.uint8)
                    # 添加随机旋转
                    angle = np.random.uniform(-2.0, 2.0)
                    h, w = source_img.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    noisy_img = cv2.warpAffine(source_img, rotation_matrix, (w, h))
                    
                    variant_path = os.path.join(test_dir, f"test_{i}_{j}.jpg")
                    cv2.imwrite(variant_path, noisy_img)
                    self.test_images.append(variant_path)
        
        print(f"创建了 {len(self.test_images)} 张测试图片")
        return self.test_images
    
    def _create_image(self, path, config):
        """创建单张测试图片"""
        img = np.ones((config["size"][1], config["size"][0], 3), dtype=np.uint8) * 255
        
        # 添加主要文本 - 针对1080P优化
        font_scale = config["size"][0] / 1200  # 1080P合适的字体大小
        thickness = max(2, int(font_scale * 3))
        
        texts = [
            config["text"],
            f"批次: {config['complexity']}",
            "并发处理测试",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "1080P高分辨率OCR测试",
            "GPU显存占用统计",
            "高性能并发处理"
        ]
        
        # 在1080P分辨率下更好地分布文本
        start_y = 200
        line_height = 120
        
        for i, text in enumerate(texts):
            y = start_y + i * line_height
            # 确保不超出图片边界
            if y < config["size"][1] - 50:
                cv2.putText(img, text, (100, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 0, 0), thickness)
        
        # 添加边框 - 1080P分辨率下使用更粗的边框
        cv2.rectangle(img, (20, 20), (config["size"][0]-20, config["size"][1]-20), (0, 0, 0), 3)
        
        # 添加一些装饰性元素来增加复杂度
        # 添加一些矩形和圆形
        for i in range(3):
            x = 100 + i * 300
            y = 800 + i * 50
            cv2.rectangle(img, (x, y), (x+200, y+80), (100, 100, 100), 2)
            cv2.putText(img, f"Region {i+1}", (x+20, y+50), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale*0.6, (0, 0, 0), 2)
        
        cv2.imwrite(path, img)
    
    def single_ocr_test(self, image_path, test_id=0):
        """单次OCR测试"""
        start_time = time.time()
        
        # 监控GPU内存使用情况 - 开始
        gpu_info_start = self.gpu_monitor.get_memory_info()
        
        try:
            # 创建OCR实例 (每个线程独立)
            ocr = OCR()
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'test_id': test_id,
                    'image_path': image_path,
                    'success': False,
                    'error': '无法读取图片',
                    'processing_time': 0,
                    'text_count': 0,
                    'gpu_info_start': gpu_info_start,
                    'gpu_info_end': gpu_info_start
                }
            
            # 执行OCR识别
            result = ocr(image)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 监控GPU内存使用情况 - 结束
            gpu_info_end = self.gpu_monitor.get_memory_info()
            
            return {
                'test_id': test_id,
                'image_path': image_path,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'success': True,
                'processing_time': processing_time,
                'text_count': len(result),
                'texts': [text for _, (text, _) in result],
                'gpu_info_start': gpu_info_start,
                'gpu_info_end': gpu_info_end,
                'gpu_memory_increase': gpu_info_end['allocated'] - gpu_info_start['allocated'] if gpu_info_start and gpu_info_end else 0
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            gpu_info_end = self.gpu_monitor.get_memory_info()
            
            return {
                'test_id': test_id,
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'text_count': 0,
                'gpu_info_start': gpu_info_start,
                'gpu_info_end': gpu_info_end,
                'gpu_memory_increase': 0
            }
    
    def run_concurrent_test(self, max_workers=4, mode='thread', test_images=None):
        """运行并发测试"""
        if test_images is None:
            test_images = self.test_images
            
        if not test_images:
            print("没有测试图片")
            return []
        
        print(f"\n开始并发测试 (模式: {mode}, 工作线程: {max_workers}, 图片数: {len(test_images)})")
        print("=" * 60)
        
        # 记录开始时的GPU状态
        gpu_start = self.gpu_monitor.get_memory_info()
        test_start_time = time.time()
        
        results = []
        
        if mode == 'thread':
            # 线程池并发
            results = self._run_thread_concurrent(test_images, max_workers)
        elif mode == 'process':
            # 进程池并发
            results = self._run_process_concurrent(test_images, max_workers)
        else:
            # 顺序执行
            results = self._run_sequential(test_images)
        
        test_end_time = time.time()
        total_time = test_end_time - test_start_time
        
        # 记录结束时的GPU状态
        gpu_end = self.gpu_monitor.get_memory_info()
        
        # 统计结果
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if successful_results:
            total_processing_time = sum(r['processing_time'] for r in successful_results)
            avg_processing_time = total_processing_time / len(successful_results)
            avg_text_count = sum(r['text_count'] for r in successful_results) / len(successful_results)
            total_texts = sum(r['text_count'] for r in successful_results)
            
            # GPU内存使用统计
            # gpu_memory_peak = max(r.get('gpu_info_end', {}).get('allocated', 0) for r in successful_results)
            # gpu_memory_increase = sum(r.get('gpu_memory_increase', 0) for r in successful_results)
            
            concurrent_efficiency = total_processing_time / total_time if total_time > 0 else 0
            
            benchmark_result = {
                'mode': mode,
                'max_workers': max_workers,
                'total_images': len(test_images),
                'successful_count': len(successful_results),
                'failed_count': len(failed_results),
                'total_time': total_time,
                'avg_processing_time': avg_processing_time,
                'total_processing_time': total_processing_time,
                'avg_text_count': avg_text_count,
                'total_texts': total_texts,
                'throughput': len(successful_results) / total_time if total_time > 0 else 0,
                'text_throughput': total_texts / total_time if total_time > 0 else 0,
                'concurrent_efficiency': concurrent_efficiency,
                # 'gpu_info': {
                #     'start': gpu_start,
                #     'end': gpu_end,
                #     'peak_memory': gpu_memory_peak,
                #     'total_memory_increase': gpu_memory_increase
                # },
                'results': results
            }
            
            self.results.append(benchmark_result)
            self._print_concurrent_summary(benchmark_result)
            
        return results
    
    def _run_thread_concurrent(self, test_images, max_workers):
        """使用线程池并发执行"""
        print("使用线程池并发执行...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_image = {
                executor.submit(self.single_ocr_test, img_path, i): img_path 
                for i, img_path in enumerate(test_images)
            }
            
            # 收集结果
            for i, future in enumerate(as_completed(future_to_image)):
                img_path = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"   完成 {i+1}/{len(test_images)}: {os.path.basename(img_path)} "
                          f"({result['processing_time']:.3f}s, {result['text_count']} 文本)")
                except Exception as e:
                    print(f"   失败 {i+1}/{len(test_images)}: {os.path.basename(img_path)} - {e}")
                    results.append({
                        'test_id': i,
                        'image_path': img_path,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0,
                        'text_count': 0
                    })
        
        return results
    
    def _run_process_concurrent(self, test_images, max_workers):
        """使用进程池并发执行"""
        print("使用进程池并发执行...")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_image = {
                executor.submit(self._process_ocr_test_wrapper, img_path, i): img_path 
                for i, img_path in enumerate(test_images)
            }
            
            # 收集结果
            for i, future in enumerate(as_completed(future_to_image)):
                img_path = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"   完成 {i+1}/{len(test_images)}: {os.path.basename(img_path)} "
                          f"({result['processing_time']:.3f}s, {result['text_count']} 文本)")
                except Exception as e:
                    print(f"   失败 {i+1}/{len(test_images)}: {os.path.basename(img_path)} - {e}")
                    results.append({
                        'test_id': i,
                        'image_path': img_path,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0,
                        'text_count': 0
                    })
        
        return results
    
    def _run_sequential(self, test_images):
        """顺序执行"""
        print("顺序执行...")
        
        results = []
        for i, img_path in enumerate(test_images):
            result = self.single_ocr_test(img_path, i)
            results.append(result)
            print(f"   完成 {i+1}/{len(test_images)}: {os.path.basename(img_path)} "
                  f"({result['processing_time']:.3f}s, {result['text_count']} 文本)")
        
        return results
    
    @staticmethod
    def _process_ocr_test_wrapper(image_path, test_id):
        """进程池包装函数"""
        # 在进程中重新创建benchmark实例
        benchmark = ConcurrentOCRBenchmark()
        return benchmark.single_ocr_test(image_path, test_id)
    
    def _print_concurrent_summary(self, result):
        """打印并发测试总结"""
        print(f"\n{'='*80}")
        print(f"{result['mode']} 并发测试总结 (工作线程: {result['max_workers']})")
        print(f"{'='*80}")
        
        print(f"测试概况:")
        print(f"   总图片数: {result['total_images']}")
        print(f"   成功处理: {result['successful_count']}")
        print(f"   失败处理: {result['failed_count']}")
        print(f"   成功率: {result['successful_count']/result['total_images']*100:.1f}%")
        
        print(f"\n时间统计:")
        print(f"   总测试时间: {result['total_time']:.3f}秒")
        print(f"   总处理时间: {result['total_processing_time']:.3f}秒")
        print(f"   平均单张处理: {result['avg_processing_time']:.3f}秒")
        print(f"   并发效率: {result['concurrent_efficiency']:.2f}")
        
        print(f"\n性能指标:")
        print(f"   吞吐量: {result['throughput']:.2f} 图片/秒")
        print(f"   文本吞吐量: {result['text_throughput']:.1f} 文本/秒")
        print(f"   平均识别文本: {result['avg_text_count']:.1f}")
        print(f"   总识别文本: {result['total_texts']}")
        
        # gpu_info = result['gpu_info']
        # if gpu_info['start'] and gpu_info['end']:
        #     print(f"\nGPU显存使用:")
        #     print(f"   测试开始时: {self.gpu_monitor.format_memory(gpu_info['start']['allocated'])}")
        #     print(f"   测试结束时: {self.gpu_monitor.format_memory(gpu_info['end']['allocated'])}")
        #     print(f"   峰值使用量: {self.gpu_monitor.format_memory(gpu_info['peak_memory'])}")
        #     print(f"   显存增长: {self.gpu_monitor.format_memory(gpu_info['total_memory_increase'])}")
        #     print(f"   显存利用率: {gpu_info['end']['utilization']*100:.1f}%")
        
        print(f"{'='*80}")
    
    def save_results(self, filename="concurrent_benchmark_results.json"):
        """保存测试结果"""
        if not self.results:
            print("没有测试结果可保存")
            return
        
        # 将结果转换为可序列化的格式
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            # 确保所有数据都可以序列化
            if 'results' in serializable_result:
                serializable_results.append(serializable_result)
            else:
                serializable_results.append(serializable_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"测试结果已保存到: {filename}")
    
    def cleanup_test_images(self):
        """清理测试图片"""
        test_dir = "concurrent_test_images"
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print(f"已清理测试图片目录: {test_dir}")


def main():
    parser = argparse.ArgumentParser(description="OCR并发性能测试")
    parser.add_argument('--workers', '-w', type=int, default=4, help='并发工作线程数 (默认: 4)')
    parser.add_argument('--mode', choices=['thread', 'process', 'sequential'], default='thread', 
                       help='并发模式: thread(线程池), process(进程池), sequential(顺序)')
    parser.add_argument('--images', type=int, default=20, help='测试图片数量 (默认: 20)')
    parser.add_argument('--keep-images', action='store_true', help='保留测试图片')
    parser.add_argument('--save-results', action='store_true', help='保存详细结果到JSON文件')
    
    args = parser.parse_args()
    
    print("OCR并发性能基准测试")
    print("=" * 60)
    
    # 检查模型文件
    model_dir = "models"
    required_files = ["det.onnx", "rec.onnx", "ocr.res"]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing_files:
        print(f"缺少模型文件: {missing_files}")
        print("请先运行: python download_models.py")
        return
    
    # 创建基准测试实例
    benchmark = ConcurrentOCRBenchmark()
    
    try:
        # 创建测试图片
        test_images = benchmark.create_test_images(args.images // 4)  # 4种基础类型
        
        if len(test_images) > args.images:
            test_images = test_images[:args.images]
        
        # 运行并发测试
        results = benchmark.run_concurrent_test(
            max_workers=args.workers,
            mode=args.mode,
            test_images=test_images
        )
        
        # 保存结果
        if args.save_results:
            benchmark.save_results(f"concurrent_{args.mode}_{args.workers}workers_results.json")
        
        print(f"\n并发测试完成!")
        
    except KeyboardInterrupt:
        print(f"\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        raise
    finally:
        # 清理
        if not args.keep_images:
            benchmark.cleanup_test_images()


if __name__ == "__main__":
    main()
