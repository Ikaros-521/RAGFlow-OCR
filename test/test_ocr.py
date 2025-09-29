#!/usr/bin/env python3
"""
OCR功能测试脚本
"""

import cv2
import numpy as np
import time
import os
from ocr import OCR


def create_test_image():
    """创建测试图像"""
    # 创建白色背景
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    
    # 添加测试文本
    texts = [
        ("Hello World!", (50, 50)),
        ("OCR Test", (50, 100)),
        ("中文测试", (50, 150)),
        ("123456789", (300, 50)),
        ("Special chars: @#$%", (300, 100))
    ]
    
    for text, pos in texts:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img


def test_basic_functionality():
    """测试基础功能"""
    print("=== 基础功能测试 ===")
    
    # 创建测试图像
    test_image = create_test_image()
    cv2.imwrite("test_image.jpg", test_image)
    print("✓ 创建测试图像")
    
    # 初始化OCR
    try:
        ocr = OCR()
        print("✓ OCR初始化成功")
    except Exception as e:
        print(f"✗ OCR初始化失败: {e}")
        return False
    
    # 执行OCR
    try:
        start_time = time.time()
        result = ocr(test_image)
        process_time = time.time() - start_time
        print(f"✓ OCR执行成功，耗时: {process_time:.3f}秒")
    except Exception as e:
        print(f"✗ OCR执行失败: {e}")
        return False
    
    # 检查结果
    if result and len(result) > 0:
        print(f"✓ 识别到 {len(result)} 个文本区域")
        for i, (bbox, (text, score)) in enumerate(result):
            print(f"  {i+1}. {text} (置信度: {score:.3f})")
    else:
        print("✗ 未识别到任何文本")
        return False
    
    return True


def test_performance():
    """性能测试"""
    print("\n=== 性能测试 ===")
    
    ocr = OCR()
    test_image = create_test_image()
    
    # 多次测试
    times = []
    for i in range(5):
        start_time = time.time()
        result = ocr(test_image)
        process_time = time.time() - start_time
        times.append(process_time)
        print(f"第{i+1}次: {process_time:.3f}秒")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"平均时间: {avg_time:.3f}秒")
    print(f"最快时间: {min_time:.3f}秒")
    print(f"最慢时间: {max_time:.3f}秒")
    
    return avg_time < 5.0  # 期望平均处理时间小于5秒


def test_different_images():
    """测试不同图像"""
    print("\n=== 不同图像测试 ===")
    
    ocr = OCR()
    
    # 测试不同尺寸的图像
    sizes = [(100, 300), (200, 600), (400, 800)]
    
    for h, w in sizes:
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.putText(img, f"Size: {w}x{h}", (10, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        try:
            result = ocr(img)
            print(f"✓ {w}x{h} 图像处理成功，识别到 {len(result)} 个文本")
        except Exception as e:
            print(f"✗ {w}x{h} 图像处理失败: {e}")
            return False
    
    return True


def test_error_handling():
    """错误处理测试"""
    print("\n=== 错误处理测试 ===")
    
    ocr = OCR()
    
    # 测试空图像
    try:
        result = ocr(None)
        print("✓ 空图像处理正常")
    except Exception as e:
        print(f"✗ 空图像处理异常: {e}")
    
    # 测试无效图像
    try:
        invalid_img = np.array([])
        result = ocr(invalid_img)
        print("✓ 无效图像处理正常")
    except Exception as e:
        print(f"✗ 无效图像处理异常: {e}")
    
    return True


def main():
    """主测试函数"""
    print("开始OCR功能测试...\n")
    
    # 检查模型文件
    model_dir = "models"
    required_files = ["det.onnx", "rec.onnx", "ocr.res"]
    
    print("=== 模型文件检查 ===")
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            print("请先运行: python download_models.py")
            return
    
    # 运行测试
    tests = [
        ("基础功能", test_basic_functionality),
        ("性能测试", test_performance),
        ("不同图像", test_different_images),
        ("错误处理", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 通过")
            else:
                print(f"✗ {test_name} 失败")
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！OCR功能正常。")
    else:
        print("❌ 部分测试失败，请检查配置。")
    
    # 清理测试文件
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")


if __name__ == "__main__":
    main()
