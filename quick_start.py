#!/usr/bin/env python3
"""
快速开始脚本 - 演示OCR基本用法
"""

import cv2
import numpy as np
import os
from ocr import OCR


def main():
    print("=== OCR快速开始示例 ===\n")
    
    # 1. 检查模型文件
    print("1. 检查模型文件...")
    model_dir = "models"
    required_files = ["det.onnx", "rec.onnx", "ocr.res"]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} 缺失")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n缺少模型文件: {missing_files}")
        print("请先运行: python download_models.py")
        return
    
    # 2. 创建测试图像
    print("\n2. 创建测试图像...")
    test_image = create_test_image()
    cv2.imwrite("quick_test.jpg", test_image)
    print("   ✓ 测试图像已创建: quick_test.jpg")
    
    # 3. 初始化OCR
    print("\n3. 初始化OCR...")
    try:
        ocr = OCR()
        print("   ✓ OCR初始化成功")
    except Exception as e:
        print(f"   ✗ OCR初始化失败: {e}")
        return
    
    # 4. 执行OCR识别
    print("\n4. 执行OCR识别...")
    try:
        result = ocr(test_image)
        print(f"   ✓ 识别完成，找到 {len(result)} 个文本区域")
    except Exception as e:
        print(f"   ✗ OCR识别失败: {e}")
        return
    
    # 5. 显示结果
    print("\n5. 识别结果:")
    print("   " + "-" * 50)
    for i, (bbox, (text, score)) in enumerate(result):
        print(f"   {i+1}. 文本: {text}")
        print(f"      置信度: {score:.3f}")
        print(f"      位置: {bbox}")
        print()
    
    # 6. 保存可视化结果
    print("6. 保存可视化结果...")
    result_image = test_image.copy()
    for bbox, (text, score) in result:
        # 绘制边界框
        pts = np.array(bbox, np.int32)
        cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
        
        # 添加文本标签
        x, y = int(bbox[0][0]), int(bbox[0][1])
        cv2.putText(result_image, f"{text} ({score:.2f})", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite("quick_result.jpg", result_image)
    print("   ✓ 可视化结果已保存: quick_result.jpg")
    
    print("\n=== 快速开始完成 ===")
    print("现在你可以:")
    print("1. 查看 quick_test.jpg - 原始测试图像")
    print("2. 查看 quick_result.jpg - 识别结果可视化")
    print("3. 运行 examples/ 目录下的示例了解更多用法")
    print("4. 查看 README.md 了解详细文档")


def create_test_image():
    """创建包含多种文本的测试图像"""
    # 创建白色背景
    img = np.ones((300, 800, 3), dtype=np.uint8) * 255
    
    # 添加不同类型的文本
    texts = [
        ("Hello World!", (50, 50), 1.0),
        ("OCR Test 2025", (50, 100), 0.8),
        ("中文识别测试", (50, 150), 1.2),
        ("123456789", (400, 50), 1.0),
        ("Special: @#$%^&*", (400, 100), 0.8),
        ("Mixed 中英文123", (400, 150), 1.0),
        ("Long text example for testing OCR accuracy", (50, 200), 0.6),
        ("多行文本\n第二行\n第三行", (400, 200), 0.8)
    ]
    
    for text, pos, scale in texts:
        # 处理多行文本
        lines = text.split('\n')
        y_offset = 0
        for line in lines:
            cv2.putText(img, line, (pos[0], pos[1] + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
            y_offset += int(30 * scale)
    
    return img


if __name__ == "__main__":
    main()
