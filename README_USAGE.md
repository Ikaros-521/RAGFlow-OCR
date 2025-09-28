# OCR使用指南

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型
```bash
python download_models.py
```

### 3. 测试单张图片
```bash
python run_ocr.py your_image.jpg
```

## 详细使用

### 基础测试程序

#### 1. 简单测试 (`run_ocr.py`)
最简单的使用方式，测试单张图片：

```bash
python run_ocr.py test.jpg
```

**输出示例：**
```
🔄 初始化OCR...
✅ OCR初始化成功
📖 读取图片: test.jpg
📐 图片尺寸: 800x600
🔍 开始OCR识别...
✅ OCR识别完成!
⏱️  处理时间: 0.245秒
📊 识别到 5 个文本区域
🚀 识别速度: 20.4 文本/秒

📝 识别结果:
--------------------------------------------------
 1. Hello World
    置信度: 0.987
    位置: [[100, 50], [200, 50], [200, 80], [100, 80]]

 2. OCR Test
    置信度: 0.956
    位置: [[100, 100], [180, 100], [180, 130], [100, 130]]
```

#### 2. 完整测试程序 (`test_image.py`)
支持单张或多张图片，提供详细统计：

```bash
# 测试单张图片
python test_image.py image.jpg

# 测试多张图片
python test_image.py image1.jpg image2.jpg image3.jpg

# 指定输出目录
python test_image.py image.jpg --output results

# 不保存可视化结果
python test_image.py image.jpg --no-viz
```

**输出示例：**
```
📊 识别结果统计:
   文本数量: 5
   处理时间: 0.245秒
   识别速度: 20.4 文本/秒

📝 识别到的文本:
------------------------------------------------------------
 1. Hello World (置信度: 0.987)
 2. OCR Test (置信度: 0.956)
 3. 中文测试 (置信度: 0.923)
 4. 123456789 (置信度: 0.998)
 5. Special chars: @#$% (置信度: 0.845)
```

#### 3. 性能基准测试 (`benchmark.py`)
测试不同尺寸图片的性能：

```bash
# 创建测试图片并运行基准测试
python benchmark.py --create-test

# 测试指定图片
python benchmark.py --images test1.jpg test2.jpg

# 每个图片测试5次
python benchmark.py --create-test --iterations 5
```

**输出示例：**
```
📊 OCR性能基准测试总结
================================================================================
图片尺寸         平均时间     识别速度       平均文本     测试次数
--------------------------------------------------------------------------------
300x100         0.089秒     11.2文本/秒    1.0         3
600x200         0.156秒     12.8文本/秒    2.0         3
1200x400        0.298秒     13.4文本/秒    4.0         3
2400x800        0.567秒     14.1文本/秒    8.0         3
--------------------------------------------------------------------------------
总体统计:
  测试图片: 4
  总处理时间: 1.110秒
  总识别文本: 15.0
  整体速度: 13.5 文本/秒
  平均每张: 0.278秒
```

## 输出文件

### 1. 可视化结果图片
- 文件名：`原图片名_result.jpg`
- 内容：原图片 + 绿色边界框 + 文本标签

### 2. 文本结果文件
- 文件名：`原图片名_ocr_text.txt`
- 内容：识别的文本列表和置信度

### 3. JSON结果文件
- 文件名：`原图片名_result.json`
- 内容：完整的识别结果，包括位置、置信度等

### 4. 批量测试结果
- 文件名：`batch_results.json`
- 内容：多张图片的汇总统计

## 性能优化建议

### 1. GPU加速
确保安装了GPU版本的onnxruntime：
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 2. 图片预处理
对于大图片，可以先调整尺寸：
```python
import cv2

def resize_image(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image
```

### 3. 批量处理
对于多张图片，复用OCR实例：
```python
from ocr import OCR

ocr = OCR()  # 只初始化一次
results = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    result = ocr(image)
    results.append(result)
```

## 故障排除

### 1. 模型文件缺失
```
❌ 缺少模型文件: ['det.onnx', 'rec.onnx', 'ocr.res']
请先运行: python download_models.py
```

### 2. 内存不足
- 减小图片尺寸
- 使用CPU模式
- 增加系统内存

### 3. 识别精度低
- 确保图片清晰
- 调整图片对比度
- 检查文本方向

### 4. 速度慢
- 使用GPU加速
- 减小图片尺寸
- 批量处理优化

## 支持的图片格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## 环境要求

- Python 3.8+
- 内存: 至少4GB RAM
- 存储: 至少2GB可用空间
- 可选: NVIDIA GPU (用于加速)
