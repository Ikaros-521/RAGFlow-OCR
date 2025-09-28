#!/usr/bin/env python3
"""
批量OCR处理示例
"""

import cv2
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import OCR


def process_images(input_dir, output_dir):
    """批量处理图像"""
    # 初始化OCR
    print("初始化OCR...")
    ocr = OCR()
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"在 {input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 批量处理
    total_time = 0
    results = []
    
    for i, image_file in enumerate(image_files):
        print(f"处理 {i+1}/{len(image_files)}: {image_file.name}")
        
        # 读取图像
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  跳过: 无法读取图像")
            continue
        
        # OCR识别
        start_time = time.time()
        result = ocr(image)
        process_time = time.time() - start_time
        total_time += process_time
        
        # 保存结果
        output_file = Path(output_dir) / f"{image_file.stem}_ocr.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"图像: {image_file.name}\n")
            f.write(f"处理时间: {process_time:.2f}秒\n")
            f.write(f"识别到 {len(result)} 个文本区域:\n")
            f.write("-" * 40 + "\n")
            
            for j, (bbox, (text, score)) in enumerate(result):
                f.write(f"{j+1}. {text} (置信度: {score:.3f})\n")
        
        # 保存可视化结果
        result_image = image.copy()
        for bbox, (text, score) in result:
            pts = np.array(bbox, np.int32)
            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
            x, y = int(bbox[0][0]), int(bbox[0][1])
            cv2.putText(result_image, f"{text[:10]}...", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        output_image = Path(output_dir) / f"{image_file.stem}_result.jpg"
        cv2.imwrite(str(output_image), result_image)
        
        results.append({
            'file': image_file.name,
            'text_count': len(result),
            'time': process_time
        })
        
        print(f"  完成: 识别到 {len(result)} 个文本，耗时 {process_time:.2f}秒")
    
    # 输出统计信息
    print(f"\n批量处理完成!")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均每张图像: {total_time/len(results):.2f}秒")
    print(f"结果保存在: {output_dir}")


def main():
    input_dir = "input_images"  # 输入图像目录
    output_dir = "output_results"  # 输出结果目录
    
    if not os.path.exists(input_dir):
        print(f"请创建 {input_dir} 目录并放入要处理的图像文件")
        return
    
    process_images(input_dir, output_dir)


if __name__ == "__main__":
    import numpy as np
    main()
