# RAGFlow OCR

基于RAGFlow的OCR功能独立拆分版本，支持文本检测和识别。

## 功能特性

- 文本检测 (Text Detection)
- 文本识别 (Text Recognition) 
- 支持GPU加速
- 支持批量处理
- 支持多种图像格式

## 安装

```bash
pip install -r requirements.txt
```

## 模型下载

首次使用需要下载模型文件：

```bash
python download_models.py
```

## 快速开始

```python
from ocr import OCR
import cv2

# 初始化OCR
ocr = OCR()

# 读取图像
image = cv2.imread('test.jpg')

# 执行OCR
result = ocr(image)

# 输出结果
for bbox, (text, score) in result:
    print(f"文本: {text}, 置信度: {score:.3f}, 位置: {bbox}")
```

## 高级用法

```python
# 指定模型路径
ocr = OCR(model_dir="./custom_models")

# 指定设备ID
result = ocr(image, device_id=0)

# 批量处理
images = [cv2.imread(f'img_{i}.jpg') for i in range(5)]
results = [ocr(img) for img in images]
```

## 输出格式

OCR返回格式为：
```python
[(bbox, (text, score)), ...]
```

其中：
- `bbox`: 文本边界框坐标 `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`
- `text`: 识别的文本内容
- `score`: 识别置信度 (0-1)

## 依赖

- onnxruntime
- opencv-python
- numpy
- Pillow
- huggingface_hub (用于模型下载)
