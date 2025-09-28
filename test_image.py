#!/usr/bin/env python3
"""
图片OCR测试程序
支持传入图片路径，进行OCR识别并统计速度
"""

import cv2
import os
import sys
import time
import argparse
import json
from pathlib import Path
from ocr import OCR


def test_single_image(image_path, output_dir="output", save_visualization=True):
    """测试单张图片的OCR识别"""
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return None
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化OCR
    print("🔄 初始化OCR...")
    try:
        ocr = OCR()
        print("✅ OCR初始化成功")
    except Exception as e:
        print(f"❌ OCR初始化失败: {e}")
        return None
    
    # 读取图片
    print(f"📖 读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None
    
    print(f"📐 图片尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 执行OCR识别
    print("🔍 开始OCR识别...")
    start_time = time.time()
    
    try:
        result = ocr(image)
        end_time = time.time()
        process_time = end_time - start_time
        print(f"✅ OCR识别完成，耗时: {process_time:.3f}秒")
    except Exception as e:
        print(f"❌ OCR识别失败: {e}")
        return None
    
    # 解析结果
    texts = []
    for i, (bbox, (text, score)) in enumerate(result):
        texts.append({
            'id': i + 1,
            'text': text,
            'confidence': float(score),
            'bbox': bbox
        })
    
    # 统计信息
    stats = {
        'image_path': image_path,
        'image_size': f"{image.shape[1]}x{image.shape[0]}",
        'text_count': len(texts),
        'process_time': process_time,
        'speed': f"{len(texts)/process_time:.1f} 文本/秒" if process_time > 0 else "0 文本/秒",
        'texts': texts
    }
    
    # 输出结果
    print(f"\n📊 识别结果统计:")
    print(f"   文本数量: {stats['text_count']}")
    print(f"   处理时间: {stats['process_time']:.3f}秒")
    print(f"   识别速度: {stats['speed']}")
    
    print(f"\n📝 识别到的文本:")
    print("-" * 60)
    for text_info in texts:
        print(f"{text_info['id']:2d}. {text_info['text']} (置信度: {text_info['confidence']:.3f})")
    
    # 保存可视化结果
    if save_visualization:
        print(f"\n🎨 保存可视化结果...")
        result_image = image.copy()
        
        for text_info in texts:
            bbox = text_info['bbox']
            text = text_info['text']
            score = text_info['confidence']
            
            # 绘制边界框
            pts = np.array(bbox, np.int32)
            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
            
            # 添加文本标签
            x, y = int(bbox[0][0]), int(bbox[0][1])
            label = f"{text[:15]}... ({score:.2f})" if len(text) > 15 else f"{text} ({score:.2f})"
            cv2.putText(result_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存可视化图片
        output_image_path = os.path.join(output_dir, f"{Path(image_path).stem}_result.jpg")
        cv2.imwrite(output_image_path, result_image)
        print(f"✅ 可视化结果已保存: {output_image_path}")
    
    # 保存JSON结果
    json_path = os.path.join(output_dir, f"{Path(image_path).stem}_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON结果已保存: {json_path}")
    
    return stats


def test_multiple_images(image_paths, output_dir="output", save_visualization=True):
    """测试多张图片的OCR识别"""
    
    print(f"🔄 开始批量测试 {len(image_paths)} 张图片...")
    
    # 初始化OCR
    print("🔄 初始化OCR...")
    try:
        ocr = OCR()
        print("✅ OCR初始化成功")
    except Exception as e:
        print(f"❌ OCR初始化失败: {e}")
        return None
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    total_time = 0
    total_texts = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"📷 处理第 {i+1}/{len(image_paths)} 张图片: {os.path.basename(image_path)}")
        
        # 检查图片文件
        if not os.path.exists(image_path):
            print(f"❌ 图片文件不存在: {image_path}")
            continue
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            continue
        
        print(f"📐 图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 执行OCR识别
        start_time = time.time()
        try:
            result = ocr(image)
            end_time = time.time()
            process_time = end_time - start_time
            total_time += process_time
            
            # 解析结果
            texts = []
            for j, (bbox, (text, score)) in enumerate(result):
                texts.append({
                    'id': j + 1,
                    'text': text,
                    'confidence': float(score),
                    'bbox': bbox
                })
            
            total_texts += len(texts)
            
            # 统计信息
            stats = {
                'image_path': image_path,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'text_count': len(texts),
                'process_time': process_time,
                'speed': f"{len(texts)/process_time:.1f} 文本/秒" if process_time > 0 else "0 文本/秒",
                'texts': texts
            }
            
            results.append(stats)
            
            print(f"✅ 识别完成: {len(texts)} 个文本, 耗时: {process_time:.3f}秒")
            
            # 保存可视化结果
            if save_visualization:
                result_image = image.copy()
                for text_info in texts:
                    bbox = text_info['bbox']
                    text = text_info['text']
                    score = text_info['confidence']
                    
                    pts = np.array(bbox, np.int32)
                    cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
                    
                    x, y = int(bbox[0][0]), int(bbox[0][1])
                    label = f"{text[:15]}... ({score:.2f})" if len(text) > 15 else f"{text} ({score:.2f})"
                    cv2.putText(result_image, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                output_image_path = os.path.join(output_dir, f"{Path(image_path).stem}_result.jpg")
                cv2.imwrite(output_image_path, result_image)
            
        except Exception as e:
            print(f"❌ OCR识别失败: {e}")
            continue
    
    # 输出总体统计
    print(f"\n{'='*60}")
    print(f"📊 批量测试完成!")
    print(f"   处理图片: {len(results)}/{len(image_paths)}")
    print(f"   总文本数: {total_texts}")
    print(f"   总耗时: {total_time:.3f}秒")
    print(f"   平均速度: {total_texts/total_time:.1f} 文本/秒" if total_time > 0 else "0 文本/秒")
    print(f"   平均每张: {total_time/len(results):.3f}秒" if len(results) > 0 else "0秒")
    
    # 保存批量结果
    batch_result = {
        'total_images': len(image_paths),
        'processed_images': len(results),
        'total_texts': total_texts,
        'total_time': total_time,
        'average_speed': f"{total_texts/total_time:.1f} 文本/秒" if total_time > 0 else "0 文本/秒",
        'average_per_image': f"{total_time/len(results):.3f}秒" if len(results) > 0 else "0秒",
        'results': results
    }
    
    batch_json_path = os.path.join(output_dir, "batch_results.json")
    with open(batch_json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_result, f, ensure_ascii=False, indent=2)
    print(f"✅ 批量结果已保存: {batch_json_path}")
    
    return batch_result


def main():
    parser = argparse.ArgumentParser(description="图片OCR测试程序")
    parser.add_argument('images', nargs='+', help='图片文件路径（支持多张图片）')
    parser.add_argument('--output', '-o', default='output', help='输出目录 (默认: output)')
    parser.add_argument('--no-viz', action='store_true', help='不保存可视化结果')
    
    args = parser.parse_args()
    
    print("🚀 OCR图片测试程序")
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
    
    # 处理图片
    if len(args.images) == 1:
        # 单张图片
        result = test_single_image(
            args.images[0], 
            args.output, 
            not args.no_viz
        )
    else:
        # 多张图片
        result = test_multiple_images(
            args.images, 
            args.output, 
            not args.no_viz
        )
    
    if result:
        print(f"\n🎉 测试完成! 结果保存在: {args.output}")
    else:
        print(f"\n❌ 测试失败!")


if __name__ == "__main__":
    import numpy as np
    main()
