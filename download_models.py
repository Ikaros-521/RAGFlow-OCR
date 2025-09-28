#!/usr/bin/env python3
#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import argparse
from huggingface_hub import snapshot_download


def download_models():
    """下载OCR模型文件"""
    print("开始下载OCR模型文件...")
    
    # 设置模型目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    try:
        # 从HuggingFace下载模型
        print(f"从HuggingFace下载模型到: {model_dir}")
        snapshot_download(
            repo_id="InfiniFlow/deepdoc",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        print("模型下载完成！")
        
        # 检查必要的文件
        required_files = ["det.onnx", "rec.onnx", "ocr.res"]
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                print(f"✓ {file} 已下载")
            else:
                print(f"✗ {file} 未找到")
                
    except Exception as e:
        print(f"下载模型时出错: {e}")
        print("请检查网络连接或尝试设置镜像源:")
        print("export HF_ENDPOINT=https://hf-mirror.com")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="下载OCR模型文件")
    parser.add_argument('--force', action='store_true', help='强制重新下载模型')
    args = parser.parse_args()
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # 检查模型是否已存在
    if not args.force and os.path.exists(os.path.join(model_dir, "det.onnx")):
        print("模型文件已存在，使用 --force 参数强制重新下载")
        return
    
    success = download_models()
    if success:
        print("\n模型下载完成！现在可以使用OCR功能了。")
        print("示例用法:")
        print("from ocr import OCR")
        print("import cv2")
        print("ocr = OCR()")
        print("image = cv2.imread('test.jpg')")
        print("result = ocr(image)")
    else:
        print("\n模型下载失败，请检查网络连接后重试。")


if __name__ == "__main__":
    main()
