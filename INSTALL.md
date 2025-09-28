# 安装指南

## 系统要求

- Python 3.8+
- 操作系统: Windows, Linux, macOS
- 内存: 至少4GB RAM
- 存储: 至少2GB可用空间

## 安装步骤

### 1. 环境准备

确保已安装Python 3.8或更高版本：
```bash
python --version
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

首次使用需要下载模型文件：
```bash
python download_models.py
```

如果下载速度慢，可以设置镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_models.py
```

### 4. 验证安装

```python
from ocr import OCR
import cv2
import numpy as np

# 创建测试图像
test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(test_image, "Hello OCR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# 测试OCR
ocr = OCR()
result = ocr(test_image)
print("安装成功！")
```

## 可选配置

### GPU支持

如果需要GPU加速，请安装CUDA版本的onnxruntime：

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 开发模式安装

```bash
pip install -e .
```

## 故障排除

### 1. 模型下载失败

**问题**: 网络连接问题导致模型下载失败

**解决方案**:
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
python download_models.py

# 或者手动下载模型文件到 models/ 目录
```

### 2. 内存不足

**问题**: 处理大图像时内存不足

**解决方案**:
- 减小图像尺寸
- 使用CPU模式
- 增加系统内存

### 3. GPU不可用

**问题**: GPU加速不工作

**解决方案**:
```python
# 强制使用CPU
ocr = OCR()
result = ocr(image, device_id=None)
```

### 4. 依赖冲突

**问题**: 与其他包的依赖冲突

**解决方案**:
```bash
# 创建虚拟环境
python -m venv ocr_env
source ocr_env/bin/activate  # Linux/macOS
# 或
ocr_env\Scripts\activate  # Windows

pip install -r requirements.txt
```

## 性能优化

### 1. 图像预处理

```python
import cv2

# 调整图像大小
def resize_image(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

# 使用
image = cv2.imread('large_image.jpg')
image = resize_image(image)
result = ocr(image)
```

### 2. 批量处理优化

```python
# 批量处理时复用OCR实例
ocr = OCR()
results = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    result = ocr(image)
    results.append(result)
```

## 更新

更新到最新版本：
```bash
git pull
pip install -r requirements.txt --upgrade
python download_models.py --force
```
