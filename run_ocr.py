#!/usr/bin/env python3
"""
简化的OCR运行程序
快速测试单张图片的OCR识别
"""

import cv2
import os
import time
import sys
from pathlib import Path
from ocr import OCR


def run_ocr(image_path):
    """运行OCR识别"""
    
    # 检查图片文件
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    # 检查模型文件
    model_dir = "models"
    required_files = ["det.onnx", "rec.onnx", "ocr.res"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    
    if missing_files:
        print(f"❌ 缺少模型文件: {missing_files}")
        print("请先运行: python download_models.py")
        return
    
    # 初始化OCR
    print("🔄 初始化OCR...")
    try:
        ocr = OCR()
        print("✅ OCR初始化成功")
    except Exception as e:
        print(f"❌ OCR初始化失败: {e}")
        return
    
    # 读取图片
    print(f"📖 读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    print(f"📐 图片尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 执行OCR识别
    print("🔍 开始OCR识别...")
    start_time = time.time()
    
    try:
        result = ocr(image)
        end_time = time.time()
        process_time = end_time - start_time
        
        print(f"✅ OCR识别完成!")
        print(f"⏱️  处理时间: {process_time:.3f}秒")
        print(f"📊 识别到 {len(result)} 个文本区域")
        
        if len(result) > 0:
            speed = len(result) / process_time
            print(f"🚀 识别速度: {speed:.1f} 文本/秒")
        
    except Exception as e:
        print(f"❌ OCR识别失败: {e}")
        return
    
    # 显示识别结果
    print(f"\n📝 识别结果:")
    print("-" * 50)
    
    for i, (bbox, (text, score)) in enumerate(result):
        print(f"{i+1:2d}. {text}")
        print(f"    置信度: {score:.3f}")
        print(f"    位置: {bbox}")
        print()
    
    # 保存结果图片
    if len(result) > 0:
        print("🎨 保存可视化结果...")
        
        result_image = image.copy()
        for bbox, (text, score) in result:
            # 绘制边界框
            pts = np.array(bbox, np.int32)
            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
            
            # 添加文本标签
            x, y = int(bbox[0][0]), int(bbox[0][1])
            label = f"{text[:20]}..." if len(text) > 20 else text
            cv2.putText(result_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存结果
        output_path = f"{Path(image_path).stem}_ocr_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"✅ 结果图片已保存: {output_path}")
        
        # 保存文本结果
        text_path = f"{Path(image_path).stem}_ocr_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"OCR识别结果 - {image_path}\n")
            f.write(f"处理时间: {process_time:.3f}秒\n")
            f.write(f"识别到 {len(result)} 个文本区域\n")
            f.write("-" * 40 + "\n")
            
            for i, (bbox, (text, score)) in enumerate(result):
                f.write(f"{i+1}. {text} (置信度: {score:.3f})\n")
        
        print(f"✅ 文本结果已保存: {text_path}")
    
    print(f"\n🎉 OCR识别完成!")


def main():
    if len(sys.argv) != 2:
        print("使用方法: python run_ocr.py <图片路径>")
        print("示例: python run_ocr.py test.jpg")
        return
    
    image_path = sys.argv[1]
    run_ocr(image_path)


if __name__ == "__main__":
    import numpy as np
    main()
