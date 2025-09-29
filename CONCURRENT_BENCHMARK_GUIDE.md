# OCR并发性能测试使用指南

## 概述

本指南介绍如何使用新创建的并发OCR性能测试工具，这些工具可以：
- 测试多线程/多进程并发处理的性能
- 统计GPU显存占用情况
- 分析处理速度和耗时时长
- 生成详细的性能报告和图表

## 工具组件

1. **`concurrent_benchmark.py`** - 主要的并发测试程序
2. **`concurrent_performance_analyzer.py`** - 性能分析器，用于生成报告和图表
3. **更新的 `requirements.txt`** - 包含了新的依赖包

## 安装依赖

```bash
# 安装新增的依赖
pip install matplotlib>=3.6.0 seaborn>=0.12.0 pandas>=1.5.0
```

## 使用方法

### 1. 运行并发测试

#### 基本用法
```bash
# 使用默认参数运行测试（线程池模式，4个工作线程，20张图片）
python concurrent_benchmark.py

# 指定工作线程数
python concurrent_benchmark.py --workers 8

# 使用进程池模式
python concurrent_benchmark.py --mode process --workers 4

# 指定测试图片数量
python concurrent_benchmark.py --images 50 --workers 8

# 保留测试图片用于检查
python concurrent_benchmark.py --keep-images
```

#### 高级选项
```bash
# 顺序执行（用于对比基准）
python concurrent_benchmark.py --mode sequential --images 10

# 保存详细结果到JSON文件
python concurrent_benchmark.py --save-results --workers 8 --images 30

# 组合参数
python concurrent_benchmark.py --mode thread --workers 6 --images 25 --save-results
```

### 2. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--workers` | int | 4 | 并发工作线程数 |
| `--mode` | str | thread | 并发模式：thread(线程池)、process(进程池)、sequential(顺序) |
| `--images` | int | 20 | 测试图片数量 |
| `--keep-images` | flag | False | 是否保留测试图片 |
| `--save-results` | flag | False | 是否保存详细结果到JSON文件 |

### 3. 性能分析

运行测试后，使用分析器生成详细报告：

```bash
# 分析最新的测试结果
python concurrent_performance_analyzer.py concurrent_thread_4workers_results.json

# 指定输出目录
python concurrent_performance_analyzer.py concurrent_process_8workers_results.json --output my_reports

# 分析多个测试结果文件
python concurrent_performance_analyzer.py results/batch_test_results.json --output analysis_20240115
```

## 测试流程

### 第一阶段：创建测试图片
程序会自动创建以下类型的测试图片：
- 简单图片 (800x400)
- 中等复杂度图片 (1200x600) 
- 复杂图片 (1600x800)
- 高复杂度图片 (2000x1000)

每种类型会生成多个变体用于并发测试。

### 第二阶段：并发性能测试
- **GPU显存监控**：实时监控GPU显存使用情况
- **并发执行**：支持线程池、进程池和顺序执行
- **时间统计**：精确测量处理时间和吞吐量
- **错误处理**：记录失败的测试并提供详细错误信息

### 第三阶段：性能分析
- **多维度对比**：比较不同模式和工作线程配置的性能
- **可扩展性分析**：分析性能随并发数变化的趋势
- **GPU利用分析**：评估GPU显存使用效率
- **可视化报告**：生成图表和详细的Markdown报告

## 监控指标

### 时间指标
- **总测试时间**：整个测试流程的总耗时
- **平均处理时间**：单张图片的平均OCR处理时间
- **吞吐量**：每秒处理的图片数量
- **并发效率**：实际处理时间占总时间的比例

### GPU指标
- **GPU显存使用**：开始、结束和峰值显存使用量
- **显存增长**：测试过程中显存使用的增长量
- **GPU利用率**：GPU计算资源的利用率
- **每图显存使用**：平均每张图片占用的显存

### 性能指标
- **成功率**：成功处理的图片比例
- **识别文本数**：每张图片识别的文本区域数量
- **文本吞吐量**：每秒识别的文本数量

## 输出文件

### 测试结果文件
- `concurrent_[mode]_[workers]workers_results.json` - 详细的测试结果数据

### 分析报告文件
分析器会在 `benchmark_reports/` 目录生成：

1. **图表文件**
   - `summary_comparison.png` - 总体性能对比
   - `radar_comparison.png` - 性能指标雷达图
   - `scalability_analysis.png` - 可扩展性分析
   - `gpu_usage_analysis.png` - GPU使用情况
   - `gpu_memory_trend.png` - GPU内存使用趋势

2. **文本报告**
   - `performance_report.md` - 详细的Markdown格式报告

## 最佳实践

### 1. 测试环境准备
```bash
# 确保GPU驱动程序正常
python test/check_gpu.py

# 验证模型文件完整
python download_models.py

# 测试单张图片OCR功能
python run_ocr.py sample.jpg
```

### 2. 测试策略
```bash
# 1. 先进行小规模测试验证环境
python concurrent_benchmark.py --mode sequential --images 5

# 2. 测试不同并发模式
python concurrent_benchmark.py --mode thread --workers 2 --images 10 --save-results
python concurrent_benchmark.py --mode thread --workers 4 --images 10 --save-results
python concurrent_benchmark.py --mode process --workers 4 --images 10 --save-results

# 3. 大规模测试最佳配置
python concurrent_benchmark.py --mode thread --workers 8 --images 100 --save-results

# 4. 生成对比分析
python concurrent_performance_analyzer.py concurrent_thread_4workers_results.json
```

### 3. 性能优化建议

**高吞吐量场景**：
- 使用线程池模式
- 增加工作线程数（建议4-8个）
- 监控GPU利用率，避免过载

**GPU显存受限场景**：
- 减少并发数量
- 使用进程池模式分散显存占用
- 监控显存增长，避免溢出

**CPU受限场景**：
- 减少并发数量或使用线程池
- 检查CPU核心数和利用率

## 故障排除

### 常见问题

1. **GPU显存不足**
   ```
   错误信息：CUDA out of memory
   解决方案：减少并发数量或图片尺寸
   ```

2. **模型加载失败**
   ```
   错误信息：模型文件不存在
   解决方案：运行 python download_models.py
   ```

3. **并发性能不佳**
   ```
   可能原因：
   - GPU利用率不足，增加并发数量
   - I/O瓶颈，使用SSD存储
   - 内存不足，减少并发数量
   ```

### 调试模式
```bash
# 单线程测试用于调试
python concurrent_benchmark.py --mode sequential --images 3 --keep-images

# 检查GPU状态
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

## 高级配置

### 自定义测试图片
如果需要使用自己的测试图片：
1. 创建目录 `my_test_images/`
2. 放置测试图片到该目录
3. 修改 `concurrent_benchmark.py` 中的图片创建逻辑

### 扩展分析功能
可以修改 `concurrent_performance_analyzer.py` 添加：
- 自定义图表类型
- 额外的性能指标
- 对比不同GPU型号的性能

## 注意事项

1. **测试一致性**：每次测试都使用相同的图片集以确保结果可比较
2. **环境准备**：测试前确保GPU工作正常、模型文件完整
3. **资源监控**：大规模测试时注意监控系统资源使用情况
4. **结果验证**：结合实际情况分析结果，避免仅依赖数值比较

通过本工具可以全面评估OCR系统在并发场景下的性能表现，为生产环境的参数调优提供数据支持。
