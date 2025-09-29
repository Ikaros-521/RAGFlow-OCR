#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发OCR性能分析器
分析并发测试结果，生成详细的性能报告和对比图表
支持显存使用趋势分析和性能提升计算
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import sys

# 设置控制台输出编码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


class ConcurrentPerformanceAnalyzer:
    """并发性能分析器"""
    
    def __init__(self):
        self.results_data = []
        self.plt_style = 'seaborn-v0_8'
        
        # 设置图表样式
        try:
            plt.style.use(self.plt_style)
        except:
            plt.style.use('default')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_results(self, json_file):
        """加载测试结果"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.results_data = data
                else:
                    self.results_data = [data]
            print(f"✅ 已加载 {len(self.results_data)} 个测试结果")
            return True
        except Exception as e:
            print(f"❌ 加载结果文件失败: {e}")
            return False
    
    def format_memory(self, bytes_value):
        """格式化内存大小"""
        if bytes_value == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(bytes_value)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
            
        return f"{size:.2f} {units[unit_index]}"
    
    def analyze_performance(self):
        """分析性能数据"""
        if not self.results_data:
            print("❌ 没有数据可分析")
            return None
        
        analysis = {
            'summary': self._get_summary_stats(),
            'comparison': self._compare_modes(),
            'scalability': self._analyze_scalability(),
            'gpu_usage': self._analyze_gpu_usage(),
            'throughput_analysis': self._analyze_throughput()
        }
        
        return analysis
    
    def _get_summary_stats(self):
        """获取总体统计信息"""
        summary = {
            'total_tests': len(self.results_data),
            'modes_tested': list(set(r.get('mode', 'unknown') for r in self.results_data)),
            'total_images_processed': sum(r.get('total_images', 0) for r in self.results_data),
            'total_successful': sum(r.get('successful_count', 0) for r in self.results_data),
            'total_failed': sum(r.get('failed_count', 0) for r in self.results_data),
            'avg_throughput': np.mean([r.get('throughput', 0) for r in self.results_data]),
            'max_throughput': max(r.get('throughput', 0) for r in self.results_data),
            'avg_success_rate': np.mean([r.get('successful_count', 0)/max(r.get('total_images', 1), 1) for r in self.results_data])
        }
        return summary
    
    def _compare_modes(self):
        """对比不同并发模式"""
        mode_stats = {}
        
        for result in self.results_data:
            mode = result.get('mode', 'unknown')
            workers = result.get('max_workers', 1)
            
            key = f"{mode}_{workers}"
            
            if key not in mode_stats:
                mode_stats[key] = {
                    'mode': mode,
                    'workers': workers,
                    'throughputs': [],
                    'processing_times': [],
                    'success_rates': [],
                    'gpu_efficiencies': [],
                    'total_times': []
                }
            
            mode_stats[key]['throughputs'].append(result.get('throughput', 0))
            mode_stats[key]['processing_times'].append(result.get('avg_processing_time', 0))
            mode_stats[key]['success_rates'].append(result.get('successful_count', 0) / max(result.get('total_images', 1), 1))
            mode_stats[key]['total_times'].append(result.get('total_time', 0))
            
            # GPU效率分析
            if gpu_info := result.get('gpu_info', {}).get('end'):
                mode_stats[key]['gpu_efficiencies'].append(gpu_info.get('utilization', 0))
        
        # 计算平均值
        comparison = {}
        for key, stats in mode_stats.items():
            comparison[key] = {
                'mode': stats['mode'],
                'workers': stats['workers'],
                'avg_throughput': np.mean(stats['throughputs']),
                'std_throughput': np.std(stats['throughputs']),
                'avg_processing_time': np.mean(stats['processing_times']),
                'avg_success_rate': np.mean(stats['success_rates']),
                'avg_total_time': np.mean(stats['total_times']),
                'avg_gpu_efficiency': np.mean(stats['gpu_efficiencies']) if stats['gpu_efficiencies'] else 0
            }
        
        return comparison
    
    def _analyze_scalability(self):
        """分析可扩展性"""
        scalability_data = {}
        
        for result in self.results_data:
            mode = result.get('mode', 'unknown')
            workers = result.get('max_workers', 1)
            
            if mode not in scalability_data:
                scalability_data[mode] = []
            
            scalability_data[mode].append({
                'workers': workers,
                'throughput': result.get('throughput', 0),
                'total_time': result.get('total_time', 0),
                'efficiency': result.get('concurrent_efficiency', 0)
            })
        
        # 按工作线程数排序
        for mode in scalability_data:
            scalability_data[mode].sort(key=lambda x: x['workers'])
        
        return scalability_data
    
    def _analyze_gpu_usage(self):
        """分析GPU使用情况"""
        gpu_data = []
        
        for result in self.results_data:
            gpu_info = result.get('gpu_info', {})
            if not gpu_info:
                continue
            
            start_mem = gpu_info.get('start', {})
            end_mem = gpu_info.get('end', {})
            peak_mem = gpu_info.get('peak_memory', 0)
            
            gpu_data.append({
                'mode': result.get('mode', 'unknown'),
                'workers': result.get('max_workers', 1),
                'start_allocated': start_mem.get('allocated', 0) if start_mem else 0,
                'end_allocated': end_mem.get('allocated', 0) if end_mem else 0,
                'peak_allocated': peak_mem,
                'memory_increase': gpu_info.get('total_memory_increase', 0),
                'total_images': result.get('total_images', 0),
                'utilization': end_mem.get('utilization', 0) if end_mem else 0
            })
        
        return gpu_data
    
    def _analyze_throughput(self):
        """分析吞吐量"""
        throughput_data = []
        
        for result in self.results_data:
            throughput_data.append({
                'mode': result.get('mode', 'unknown'),
                'workers': result.get('max_workers', 1),
                'throughput': result.get('throughput', 0),
                'text_throughput': result.get('text_throughput', 0),
                'success_rate': result.get('successful_count', 0) / max(result.get('total_images', 1), 1),
                'total_images': result.get('total_images', 0)
            })
        
        return throughput_data
    
    def create_performance_report(self, output_dir="benchmark_reports"):
        """创建性能分析报告"""
        if not self.results_data:
            print("❌ 没有数据可以生成报告")
            return
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"📊 生成性能分析报告到: {output_dir}")
        
        # 分析数据
        analysis = self.analyze_performance()
        
        # 生成各种图表
        self._create_summary_plots(analysis, output_dir)
        self._create_comparison_plots(analysis, output_dir)
        self._create_scalability_plots(analysis, output_dir)
        self._create_gpu_usage_plots(analysis, output_dir)
        
        # 生成文本报告
        self._create_text_report(analysis, output_dir)
        
        print(f"✅ 性能分析报告生成完成!")
    
    def _create_summary_plots(self, analysis, output_dir):
        """创建总结图表"""
        # 总体性能对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OCR并发性能总体分析', fontsize=16, fontweight='bold')
        
        # 吞吐量对比
        comparison = analysis['comparison']
        modes = [f"{data['mode']}_{data['workers']}w" for data in comparison.values()]
        throughputs = [data['avg_throughput'] for data in comparison.values()]
        
        axes[0, 0].bar(modes, throughputs, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('平均吞吐量对比')
        axes[0, 0].set_ylabel('图片处理/秒')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 成功率对比
        success_rates = [data['avg_success_rate'] * 100 for data in comparison.values()]
        axes[0, 1].bar(modes, success_rates, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('成功率对比')
        axes[0, 1].set_ylabel('成功率 (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # GPU效率对比
        gpu_efficiency = [data['avg_gpu_efficiency'] * 100 for data in comparison.values()]
        axes[1, 0].bar(modes, gpu_efficiency, color='orange', alpha=0.7)
        axes[1, 0].set_title('GPU利用率对比')
        axes[1, 0].set_ylabel('利用率 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 处理时间对比
        processing_times = [data['avg_processing_time'] for data in comparison.values()]
        axes[1, 1].bar(modes, processing_times, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('平均处理时间对比')
        axes[1, 1].set_ylabel('时间 (秒)')
        axes[1, 1].tick_params(axis='xes', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_plots(self, analysis, output_dir):
        """创建对比图表"""
        comparison = analysis['comparison']
        
        # 性能指标雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['avg_throughput', 'avg_success_rate', 'avg_gpu_efficiency']
        metric_labels = ['吞吐量', '成功率', 'GPU效率']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合圆圈
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (key, data) in enumerate(comparison.items()):
            values = []
            for metric in metrics:
                if metric == 'avg_throughput':
                    # 归一化吞吐量 (假设最大值为100)
                    values.append(min(data[metric] / 100, 1))
                elif metric == 'avg_success_rate':
                    values.append(data[metric])
                elif metric == 'avg_gpu_efficiency':
                    values.append(data[metric])
            
            values += values[:1]  # 闭合圆圈
            ax.plot(angles, values, 'o-', linewidth=2, label=key, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('性能指标雷达图对比')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radar_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scalability_plots(self, analysis, output_dir):
        """创建可扩展性图表"""
        scalability = analysis['scalability']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('可扩展性分析', fontsize=16, fontweight='bold')
        
        # 按模式分组绘制
        for mode, modes in scalability.items():
            workers = [d['workers'] for d in modes]
            throughputs = [d['throughput'] for d in modes]
            times = [d['total_time'] for d in modes]
            efficiencies = [d['efficiency'] for d in modes]
            
            # 吞吐量扩展
            axes[0, 0].plot(workers, throughputs, 'o-', label=f'{mode}', linewidth=2)
            
            # 总耗时
            axes[0, 1].plot(workers, times, 's-', label=f'{mode}', linewidth=2)
            
            # 效率
            axes[1, 0].plot(workers, efficiencies, '^-', label=f'{mode}', linewidth=2)
            
            # 扩展倍数 (相对于单线程)
            if len(throughputs) > 0:
                speedup = [t/throughputs[0] if throughputs[0] > 0 else 0 for t in throughputs]
                axes[1, 1].plot(workers, speedup, 'd-', label=f'{mode}', linewidth=2)
        
        axes[0, 0].set_title('吞吐量随线程数变化')
        axes[0, 0].set_xlabel('工作线程数')
        axes[0, 0].set_ylabel('吞吐量 (图片/秒)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('总耗时随线程数变化')
        axes[0, 1].set_xlabel('工作线程数')
        axes[0, 1].set_ylabel('总耗时 (秒)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('并发效率随线程数变化')
        axes[1, 0].set_xlabel('工作线程数')
        axes[1, 0].set_ylabel('效率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('加速比随线程数变化')
        axes[1, 1].set_xlabel('工作线程数')
        axes[1, 1].set_ylabel('加速比')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scalability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gpu_usage_plots(self, analysis, output_dir):
        """创建GPU使用情况图表"""
        gpu_data = analysis['gpu_usage']
        
        if not gpu_data:
            print("⚠️ 没有GPU使用数据，跳过GPU图表生成")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GPU使用情况分析', fontsize=16, fontweight='bold')
        
        # 准备数据
        df = pd.DataFrame(gpu_data)
        
        # GPU内存使用对比
        mode_workers = [f"{row['mode']}_{row['workers']}" for _, row in df.iterrows()]
        
        axes[0, 0].bar(mode_workers, df['peak_allocated'], color='lightblue', alpha=0.7)
        axes[0, 0].set_title('峰值GPU内存使用')
        axes[0, 0].set_ylabel('内存 (字节)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # GPU内存增长
        axes[0, 1].bar(mode_workers, df['memory_increase'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('GPU内存增长')
        axes[0, 1].set_ylabel('内存增长 (字节)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # GPU利用率
        axes[1, 0].bar(mode_workers, df['utilization'] * 100, color='orange', alpha=0.7)
        axes[1, 0].set_title('GPU利用率')
        axes[1, 0].set_ylabel('利用率 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 每张图片的平均GPU内存使用
        avg_mem_per_image = df['peak_allocated'] / df['total_images']
        axes[1, 1].bar(mode_workers, avg_mem_per_image, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('每张图片平均GPU内存使用')
        axes[1, 1].set_ylabel('内存/图片 (字节)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gpu_usage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # GPU内存使用趋势图
        if len(gpu_data) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            workers_list = list(set(d['workers'] for d in gpu_data))
            
            for mode in set(d['mode'] for d in gpu_data):
                mode_data = [d for d in gpu_data if d['mode'] == mode]
                mode_data.sort(key=lambda x: x['workers'])
                
                workers = [d['workers'] for d in mode_data]
                peak_mem = [d['peak_allocated'] for d in mode_data]
                
                ax.plot(workers, peak_mem, 'o-', label=f'{mode}', linewidth=2)
            
            ax.set_title('GPU内存使用随线程数变化')
            ax.set_xlabel('工作线程数')
            ax.set_ylabel('峰值GPU内存使用 (字节)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/gpu_memory_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_text_report(self, analysis, output_dir):
        """创建文本报告"""
        report_file = f"{output_dir}/performance_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# OCR并发性能分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 概述
            f.write("## 测试概述\n\n")
            summary = analysis['summary']
            f.write(f"- 总测试次数: {summary['total_tests']}\n")
            f.write(f"- 测试模式: {', '.join(summary['modes_tested'])}\n")
            f.write(f"- 总处理图片数: {summary['total_images_processed']}\n")
            f.write(f"- 成功率: {summary['avg_success_rate']*100:.1f}%\n")
            f.write(f"- 平均吞吐量: {summary['avg_throughput']:.2f} 图片/秒\n")
            f.write(f"- 最大吞吐量: {summary['max_throughput']:.2f} 图片/秒\n\n")
            
            # 模式对比
            f.write("## 并发模式对比\n\n")
            f.write("| 模式 | 线程数 | 平均吞吐量 | 平均成功率 | GPU效率 |\n")
            f.write("|------|--------|------------|------------|----------|\n")
            
            comparison = analysis['comparison']
            for key, data in comparison.items():
                f.write(f"| {data['mode']} | {data['workers']} | "
                       f"{data['avg_throughput']:.2f} | "
                       f"{data['avg_success_rate']*100:.1f}% | "
                       f"{data['avg_gpu_efficiency']*100:.1f}% |\n")
            
            # GPU使用分析
            f.write("\n## GPU使用情况分析\n\n")
            gpu_data = analysis['gpu_usage']
            if gpu_data:
                f.write("| 配置 | 平均GPU利用率 | 峰值内存使用 | 内存增长 |\n")
                f.write("|------|---------------|--------------|----------|\n")
                
                for gpu_info in gpu_data:
                    f.write(f"| {gpu_info['mode']}_{gpu_info['workers']}w | "
                           f"{gpu_info['utilization']*100:.1f}% | "
                           f"{self.format_memory(gpu_info['peak_allocated'])} | "
                           f"{self.format_memory(gpu_info['memory_increase'])} |\n")
            else:
                f.write("无GPU使用数据\n")
            
            # 可扩展性分析
            f.write("\n## 可扩展性分析\n\n")
            scalability = analysis['scalability']
            
            for mode, data in scalability.items():
                f.write(f"### {mode} 模式\n\n")
                f.write("| 线程数 | 吞吐量 | 加速比 | 效率 |\n")
                f.write("|--------|--------|--------|------|\n")
                
                base_throughput = data[0]['throughput'] if data else 1
                
                for d in data:
                    speedup = d['throughput'] / base_throughput if base_throughput > 0 else 1
                    f.write(f"| {d['workers']} | {d['throughput']:.2f} | "
                           f"{speedup:.2f}x | {d['efficiency']:.2f} |\n")
                
                f.write("\n")
            
            # 建议和结论
            f.write("## 性能建议\n\n")
            
            best_throughput = max(comparison.values(), key=lambda x: x['avg_throughput'])
            best_efficiency = max(comparison.values(), key=lambda x: x['avg_gpu_efficiency'])
            
            f.write(f"### 推荐配置\n")
            f.write(f"- **最高吞吐量**: {best_throughput['mode']} 模式 ({best_throughput['workers']} 线程)\n")
            f.write(f"- **最高GPU效率**: {best_efficiency['mode']} 模式 ({best_efficiency['workers']} 线程)\n\n")
            
            f.write("### 性能优化建议\n")
            
            # 基于数据的建议
            thread_mode_results = [d for d in comparison.values() if d['mode'] == 'thread']
            process_mode_results = [d for d in comparison.values() if d['mode'] == 'process']
            
            if thread_mode_results and process_mode_results:
                avg_thread_throughput = np.mean([d['avg_throughput'] for d in thread_mode_results])
                avg_process_throughput = np.mean([d['avg_throughput'] for d in process_mode_results])
                
                if avg_thread_throughput > avg_process_throughput:
                    f.write("- 推荐使用线程并发模式，相比进程模式性能更佳\n")
                else:
                    f.write("- 推荐使用进程并发模式，相比线程模式性能更佳\n")
            
            # GPU使用建议
            if gpu_data:
                avg_gpu_utilization = np.mean([g['utilization'] for g in gpu_data])
                if avg_gpu_utilization < 0.3:
                    f.write("- GPU利用率较低，可考虑增加并发数量或模型优化\n")
                elif avg_gpu_utilization > 0.8:
                    f.write("- GPU利用率较高，建议适当减少并发数量避免显存溢出\n")
                else:
                    f.write("- GPU利用率适中，当前配置较为合理\n")
        
        print(f"✅ 文本报告已生成: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="OCR并发性能分析器")
    parser.add_argument('json_file', help='测试结果JSON文件名')
    parser.add_argument('--output', '-o', default='benchmark_reports', help='输出报告目录')
    
    args = parser.parse_args()
    
    print("📊 OCR并发性能分析器")
    print("=" * 50)
    
    # 创建分析器
    analyzer = ConcurrentPerformanceAnalyzer()
    
    # 加载结果
    if not analyzer.load_results(args.json_file):
        return
    
    # 生成分析报告
    analyzer.create_performance_report(args.output)
    
    print(f"🎉 性能分析完成! 查看 {args.output} 目录获取详细报告")


if __name__ == "__main__":
    main()
