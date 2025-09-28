# RAGFlow OCR

基于RAGFlow的OCR功能独立拆分版本，支持文本检测和识别。

## 功能特性

- ✅ 文本检测 (Text Detection)
- ✅ 文本识别 (Text Recognition) 
- ✅ 支持GPU加速
- ✅ 支持批量处理
- ✅ 支持多种图像格式
- ✅ 自动文本方向校正
- ✅ 高精度识别

## 安装

### 1. 克隆项目
```bash
git clone <repository-url>
cd standalone_ocr
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载模型
```bash
python download_models.py
```

## 快速开始

### 基础使用
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

### 高级使用
```python
# 指定模型路径
ocr = OCR(model_dir="./custom_models")

# 指定设备ID (GPU)
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

## 配置选项

可以通过环境变量配置：

```bash
# 设置并行设备数量 (0=CPU, >0=GPU数量)
export PARALLEL_DEVICES=1

# 设置HuggingFace镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

## 示例

### 1. 基础使用示例
```bash
cd examples
python basic_usage.py
```

### 2. 批量处理示例
```bash
cd examples
python batch_processing.py
```

### 3. 高级使用示例
```bash
cd examples
python advanced_usage.py
```

## 性能优化

### GPU加速
```python
# 使用GPU加速
ocr = OCR()
result = ocr(image, device_id=0)  # 使用第一个GPU
```

### 批量处理
```python
# 批量处理提高效率
images = [cv2.imread(f'img_{i}.jpg') for i in range(10)]
results = []
for img in images:
    result = ocr(img)
    results.append(result)
```

## 依赖

- onnxruntime>=1.15.0
- opencv-python>=4.8.0
- numpy>=1.21.0
- Pillow>=9.0.0
- huggingface_hub>=0.16.0

## 故障排除

### 1. 模型下载失败
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com
python download_models.py
```

### 2. GPU内存不足
```python
# 使用CPU
ocr = OCR()
result = ocr(image, device_id=None)
```

### 3. 识别精度低
- 确保图像清晰度足够
- 调整图像大小
- 检查文本方向

## 许可证

Apache License 2.0

## 贡献

欢迎提交Issue和Pull Request！
