#!/usr/bin/env python3
"""
高级OCR使用示例
"""

import cv2
import sys
import os
import json
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import OCR


class OCRProcessor:
    """OCR处理器类"""
    
    def __init__(self, model_dir=None, device_id=0):
        self.ocr = OCR(model_dir)
        self.device_id = device_id
        
    def process_image(self, image_path, save_visualization=True, output_dir="output"):
        """处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # OCR识别
        start_time = time.time()
        result = self.ocr(image, device_id=self.device_id)
        process_time = time.time() - start_time
        
        # 解析结果
        texts = []
        for bbox, (text, score) in result:
            texts.append({
                'text': text,
                'confidence': score,
                'bbox': bbox
            })
        
        # 保存结果
        if save_visualization:
            self._save_visualization(image, result, image_path, output_dir)
        
        return {
            'image_path': image_path,
            'texts': texts,
            'text_count': len(texts),
            'process_time': process_time
        }
    
    def _save_visualization(self, image, result, image_path, output_dir):
        """保存可视化结果"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        result_image = image.copy()
        for bbox, (text, score) in result:
            # 绘制边界框
            pts = np.array(bbox, np.int32)
            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
            
            # 添加文本标签
            x, y = int(bbox[0][0]), int(bbox[0][1])
            label = f"{text[:15]}... ({score:.2f})" if len(text) > 15 else f"{text} ({score:.2f})"
            cv2.putText(result_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存图像
        output_path = Path(output_dir) / f"{Path(image_path).stem}_visualization.jpg"
        cv2.imwrite(str(output_path), result_image)
    
    def extract_text_only(self, image_path):
        """仅提取文本内容"""
        result = self.process_image(image_path, save_visualization=False)
        return [item['text'] for item in result['texts']]
    
    def save_json_result(self, result, output_path):
        """保存JSON格式结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    # 创建OCR处理器
    processor = OCRProcessor(device_id=0)
    
    # 测试图像路径
    test_image = "test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"请将测试图像命名为 {test_image} 并放在当前目录")
        return
    
    print("=== 高级OCR使用示例 ===")
    
    # 1. 完整处理
    print("\n1. 完整处理...")
    result = processor.process_image(test_image, output_dir="output")
    
    print(f"处理时间: {result['process_time']:.2f}秒")
    print(f"识别到 {result['text_count']} 个文本区域")
    
    # 2. 仅提取文本
    print("\n2. 仅提取文本内容...")
    texts = processor.extract_text_only(test_image)
    print("提取的文本:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    # 3. 保存JSON结果
    print("\n3. 保存JSON结果...")
    processor.save_json_result(result, "output/ocr_result.json")
    print("JSON结果已保存到: output/ocr_result.json")
    
    # 4. 性能测试
    print("\n4. 性能测试...")
    times = []
    for i in range(5):
        start_time = time.time()
        processor.extract_text_only(test_image)
        process_time = time.time() - start_time
        times.append(process_time)
        print(f"  第{i+1}次: {process_time:.3f}秒")
    
    avg_time = sum(times) / len(times)
    print(f"平均处理时间: {avg_time:.3f}秒")
    
    print("\n=== 处理完成 ===")


if __name__ == "__main__":
    import numpy as np
    main()
