#!/usr/bin/env python3
"""
基础OCR使用示例
"""

import cv2
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import OCR


def main():
    # 初始化OCR
    print("初始化OCR...")
    ocr = OCR()
    
    # 读取测试图像
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"请将测试图像命名为 {image_path} 并放在当前目录")
        return
    
    print(f"读取图像: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("无法读取图像文件")
        return
    
    # 执行OCR
    print("执行OCR识别...")
    result = ocr(image)
    
    # 输出结果
    print(f"\n识别到 {len(result)} 个文本区域:")
    print("-" * 50)
    
    for i, (bbox, (text, score)) in enumerate(result):
        print(f"文本 {i+1}:")
        print(f"  内容: {text}")
        print(f"  置信度: {score:.3f}")
        print(f"  位置: {bbox}")
        print()
    
    # 保存结果图像
    output_path = "ocr_result.jpg"
    result_image = image.copy()
    
    for bbox, (text, score) in result:
        # 绘制边界框
        pts = np.array(bbox, np.int32)
        cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
        
        # 添加文本标签
        x, y = int(bbox[0][0]), int(bbox[0][1])
        cv2.putText(result_image, f"{text[:10]}...", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, result_image)
    print(f"结果图像已保存到: {output_path}")


if __name__ == "__main__":
    import numpy as np
    main()
