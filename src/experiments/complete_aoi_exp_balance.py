# src/experiments/complete_aoi_exp_balance.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
from typing import Dict, List, Tuple
import json
from datetime import datetime
import platform

warnings.filterwarnings('ignore')

# 导入必要的模块
from ..models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..algorithms.tlbo import TLBO
from ..algorithms.tlbo_hho import TLBOHHO
from ..algorithms.ga import GA
from ..algorithms.gwo import GWO

# 导入之前的系统创建函数
from .realistic_system_setup import (
    create_realistic_edge_computing_system,
    plot_device_capabilities,
    get_task_type
)

# 定义全局颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 全局变量控制是否使用英文标签
USE_ENGLISH_LABELS = True

# 标签字典
LABELS = {
    'zh': {
        'convergence_title': '算法收敛曲线对比（平衡权重）',
        'iterations': '迭代次数',
        'fitness': '适应度值',
        'avg_energy': '平均能耗 (J)',
        'avg_delay': '平均延迟 (s)',
        'avg_aoi': '平均AoI (s)',
        'aoi_violation': 'AoI违规率 (%)',
        'value': '值',
        'energy_dist': '能耗分布',
        'delay_dist': '延迟分布',
        'aoi_dist': 'AoI分布',
        'energy': '能耗',
        'delay': '延迟',
        'aoi': 'AoI',
        'aoi_compliance': 'AoI合规率',
        'performance_title': '算法综合性能对比（平衡权重）',
        'algorithm': '算法',
        'avg_fitness': '平均适应度',
        'performance_comparison': '算法性能对比（平衡权重）'
    },
    'en': {
        'convergence_title': 'Algorithm Convergence Comparison (Balanced Weights)',
        'iterations': 'Iterations',
        'fitness': 'Fitness Value',
        'avg_energy': 'Average Energy (J)',
        'avg_delay': 'Average Delay (s)',
        'avg_aoi': 'Average AoI (s)',
        'aoi_violation': 'AoI Violation Rate (%)',
        'value': 'Value',
        'energy_dist': 'Energy Distribution',
        'delay_dist': 'Delay Distribution',
        'aoi_dist': 'AoI Distribution',
        'energy': 'Energy',
        'delay': 'Delay',
        'aoi': 'AoI',
        'aoi_compliance': 'AoI Compliance',
        'performance_title': 'Algorithm Performance Comparison (Balanced Weights)',
        'algorithm': 'Algorithm',
        'avg_fitness': 'Average Fitness',
        'performance_comparison': 'Algorithm Performance Comparison (Balanced Weights)'
    }
}

# 设置中文字体的函数
def setup_chinese_font():
    """设置支持中文的字体"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':
        # macOS系统
        font_list = ['PingFang SC', 'Heiti SC', 'Songti SC', 'STHeiti', 'STSong']
    else:
        # Linux系统
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                    'Noto Sans CJK', 'DejaVu Sans', 'Liberation Sans']
    
    # 尝试设置字体
    font_set = False
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            # 测试是否能正常显示中文
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', fontsize=12)
            plt.close(fig)
            font_set = True
            print(f"成功设置中文字体: {font}")
            break
        except:
            continue
    
    if not font_set:
        print("警告: 无法找到合适的中文字体，图表中的中文可能无法正常显示")
        print("请安装中文字体，如: SimHei, Microsoft YaHei等")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False

# 定义全局颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 标签字典
LABELS = {
    'zh': {
        'convergence_title': '算法收敛曲线对比（平衡权重）',
        'iterations': '迭代次数',
        'fitness': '适应度值',
        'avg_energy': '平均能耗 (J)',
        'avg_delay': '平均延迟 (s)',
        'avg_aoi': '平均AoI (s)',
        'aoi_violation': 'AoI违规率 (%)',
        'value': '值',
        'energy_dist': '能耗分布',
        'delay_dist': '延迟分布',
        'aoi_dist': 'AoI分布',
        'energy': '能耗',
        'delay': '延迟',
        'aoi': 'AoI',
        'aoi_compliance': 'AoI合规率',
        'performance_title': '算法综合性能对比（平衡权重）',
        'algorithm': '算法',
        'avg_fitness': '平均适应度',
        'performance_comparison': '算法性能对比（平衡权重）'
    },
    'en': {
        'convergence_title': 'Algorithm Convergence Comparison (Balanced Weights)',
        'iterations': 'Iterations',
        'fitness': 'Fitness Value',
        'avg_energy': 'Average Energy (J)',
        'avg_delay': 'Average Delay (s)',
        'avg_aoi': 'Average AoI (s)',
        'aoi_violation': 'AoI Violation Rate (%)',
        'value': 'Value',
        'energy_dist': 'Energy Distribution',
        'delay_dist': 'Delay Distribution',
        'aoi_dist': 'AoI Distribution',
        'energy': 'Energy',
        'delay': 'Delay',
        'aoi': 'AoI',
        'aoi_compliance': 'AoI Compliance',
        'performance_title': 'Algorithm Performance Comparison (Balanced Weights)',
        'algorithm': 'Algorithm',
        'avg_fitness': 'Average Fitness',
        'performance_comparison': 'Algorithm Performance Comparison (Balanced Weights)'
    }
}

def get_label(key):
    """获取标签文本"""
    lang = 'en' if USE_ENGLISH_LABELS else 'zh'
    return LABELS[lang].get(key, key)

# 设置中文字体
setup_chinese_font()


def create_aoi_aware_system(num_devices=20, num_edge_servers=4, num_cloud_servers=2, num_tasks=40):
    """创建考虑AoI的边缘计算系统"""
    # 首先创建基础系统
    system = create_realistic_edge_computing_system(
        num_devices=num_devices,
        num_edge_servers=num_edge_servers,
        num_cloud_servers=num_cloud_servers,
        num_tasks=num_tasks
    )
    
    # 为所有任务添加AoI相关参数
    for task in system.tasks:
        task_type = get_task_type(task)
        
        if task_type == 'realtime_sensitive':
            # 实时敏感任务需要频繁更新
            task.update_interval = np.random.uniform(0.1, 0.3)  # 100-300ms
            task.max_aoi = np.random.uniform(0.5, 1.0)  # 最大可接受AoI: 0.5-1秒
        elif task_type == 'compute_intensive':
            # 计算密集型任务的更新间隔可以稍长
            task.update_interval = np.random.uniform(0.5, 1.0)  # 0.5-1秒
            task.max_aoi = np.random.uniform(2.0, 3.0)  # 最大可接受AoI: 2-3秒
        elif task_type == 'data_intensive':
            # 数据密集型任务
            task.update_interval = np.random.uniform(0.8, 1.5)  # 0.8-1.5秒
            task.max_aoi = np.random.uniform(3.0, 5.0)  # 最大可接受AoI: 3-5秒
        else:  # lightweight
            # 轻量级任务
            task.update_interval = np.random.uniform(0.3, 0.6)  # 300-600ms
            task.max_aoi = np.random.uniform(1.0, 2.0)  # 最大可接受AoI: 1-2秒
    
    return system


def run_aoi_experiment(system_config=None):
    """运行考虑AoI的平衡权重实验"""
    
    # 1. 创建或加载系统
    if system_config is None:
        print("="*100)
        print("1. Creating AoI-aware Edge Computing System")
        print("="*100)
        system = create_aoi_aware_system(
            num_devices=20, 
            num_edge_servers=4, 
            num_cloud_servers=2, 
            num_tasks=40
        )
    else:
        system = SystemModel.from_config(system_config)
        # 为配置添加AoI参数
        for task in system.tasks:
            if not hasattr(task, 'update_interval') or task.update_interval is None:
                task_type = get_task_type(task)
                if task_type == 'realtime_sensitive':
                    task.update_interval = np.random.uniform(0.1, 0.3)
                    task.max_aoi = np.random.uniform(0.5, 1.0)
                elif task_type == 'compute_intensive':
                    task.update_interval = np.random.uniform(0.5, 1.0)
                    task.max_aoi = np.random.uniform(2.0, 3.0)
                elif task_type == 'data_intensive':
                    task.update_interval = np.random.uniform(0.8, 1.5)
                    task.max_aoi = np.random.uniform(3.0, 5.0)
                else:
                    task.update_interval = np.random.uniform(0.3, 0.6)
                    task.max_aoi = np.random.uniform(1.0, 2.0)
    
    # 2. 系统AoI特性分析
    print("\n" + "="*100)
    print("2. System AoI Characteristics Analysis")
    print("\n" + "="*100)
    analyze_aoi_characteristics(system)
    
    # 3. 运行平衡权重的算法对比实验
    print("\n" + "="*100)
    print("3. Running Algorithm Comparison (Balanced Weights: Energy, Delay, and AoI)")
    print("="*100)
    
    # 使用平衡权重
    results = run_algorithm_comparison_balanced(
        system, 
        max_iter=150, 
        population_size=50, 
        n_runs=5
    )
    
    # 4. 分析和可视化结果
    print("\n" + "="*100)
    print("4. Analyzing Experimental Results")
    print("="*100)
    
    analyze_balanced_results(results, system)
    
    # 5. 生成实验图表
    print("\n" + "="*100)
    print("5. Generating Result Figures")
    print("="*100)
    
    plot_balanced_results(results, system)
    
    print("\nAoI experiment completed! All results have been saved to results/aoi/ directory")
    
    return system, results


def run_algorithm_comparison_balanced(system, max_iter=150, population_size=50, n_runs=5):
    """运行平衡权重的算法对比实验"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 针对AoI优化的权重
    w_energy, w_delay, w_aoi = 0.25, 0.35, 0.40
    
    # 初始化算法
    algorithms = {
        'TLBO': TLBO(
            system, delay_model, energy_model, aoi_model,
            max_iter=max_iter, population_size=population_size,
            w_energy=w_energy, w_delay=w_delay, w_aoi=w_aoi, verbose=False
        ),
        'TLBOHHO': TLBOHHO(
            system, delay_model, energy_model, aoi_model,
            max_iter=max_iter, population_size=population_size,
            w_energy=w_energy, w_delay=w_delay, w_aoi=w_aoi, 
            hho_prob=0.7,  # 增加HHO使用概率
            verbose=False
        ),
        'GA': GA(
            system, delay_model, energy_model, aoi_model,
            max_iter=max_iter, population_size=population_size,
            w_energy=w_energy, w_delay=w_delay, w_aoi=w_aoi, verbose=False
        ),
        'GWO': GWO(
            system, delay_model, energy_model, aoi_model,
            max_iter=max_iter, population_size=population_size,
            w_energy=w_energy, w_delay=w_delay, w_aoi=w_aoi, verbose=False
        )
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"  Running {name} algorithm...")
        
        best_fitness_history = np.zeros((n_runs, max_iter + 1))
        best_solutions = []
        best_fitness_values = []
        detailed_metrics = []
        
        for run in range(n_runs):
            print(f"    Run {run + 1}/{n_runs}", end=' ... ')
            
            # 重置系统状态
            for task in system.tasks:
                task.execution_location = None
                task.execution_node_id = None
                task.allocated_resource = None
                task.delay = None
                task.energy = None
                if hasattr(task, 'aoi'):
                    task.aoi = None
            
            # 运行算法
            best_solution, best_fitness, history = algorithm.optimize()
            
            # 记录历史
            if len(history) <= max_iter + 1:
                best_fitness_history[run, :len(history)] = history
                if len(history) < max_iter + 1:
                    best_fitness_history[run, len(history):] = history[-1]
            else:
                best_fitness_history[run, :] = history[:max_iter + 1]
            
            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            
            # 计算详细指标
            system.apply_solution(best_solution)
            metrics = calculate_detailed_metrics(system, delay_model, energy_model, aoi_model)
            detailed_metrics.append(metrics)
            
            print(f"Completed, Fitness: {best_fitness:.6f}")
        
        # 计算统计信息
        avg_fitness_history = np.mean(best_fitness_history, axis=0)
        std_fitness_history = np.std(best_fitness_history, axis=0)
        
        # 记录结果
        results[name] = {
            'best_solutions': best_solutions,
            'best_fitness_values': best_fitness_values,
            'avg_fitness_history': avg_fitness_history,
            'std_fitness_history': std_fitness_history,
            'mean_fitness': np.mean(best_fitness_values),
            'std_fitness': np.std(best_fitness_values),
            'best_fitness': np.min(best_fitness_values),
            'detailed_metrics': detailed_metrics
        }
        
        print(f"    {name} completed: Average fitness {results[name]['mean_fitness']:.6f} ± {results[name]['std_fitness']:.6f}")
    
    return results


def calculate_detailed_metrics(system, delay_model, energy_model, aoi_model):
    """计算详细的性能指标"""
    metrics = {
        'total_energy': 0,
        'total_delay': 0,
        'total_aoi': 0,
        'avg_energy': 0,
        'avg_delay': 0,
        'avg_aoi': 0,
        'delay_violations': 0,
        'aoi_violations': 0,
        'task_allocation': {'device': 0, 'edge': 0, 'cloud': 0}
    }
    
    valid_aoi_tasks = 0
    
    for task in system.tasks:
        # 计算能耗和延迟
        if task.energy is None:
            task.energy = energy_model.calculate_total_energy(task)
        if task.delay is None:
            task.delay = delay_model.calculate_total_delay(task)
        
        metrics['total_energy'] += task.energy
        metrics['total_delay'] += task.delay
        
        # 计算AoI
        if hasattr(task, 'update_interval') and task.update_interval is not None:
            try:
                task.aoi = aoi_model.calculate_average_aoi(task)
                metrics['total_aoi'] += task.aoi
                valid_aoi_tasks += 1
                
                # 检查AoI违规
                if hasattr(task, 'max_aoi') and task.max_aoi is not None and task.aoi > task.max_aoi:
                    metrics['aoi_violations'] += 1
            except:
                pass
        
        # 检查延迟违规
        if task.delay > task.max_delay:
            metrics['delay_violations'] += 1
        
        # 统计任务分配
        if task.execution_location == 'device':
            metrics['task_allocation']['device'] += 1
        elif task.execution_location == 'edge':
            metrics['task_allocation']['edge'] += 1
        elif task.execution_location == 'cloud':
            metrics['task_allocation']['cloud'] += 1
    
    # 计算平均值
    n_tasks = len(system.tasks)
    metrics['avg_energy'] = metrics['total_energy'] / n_tasks if n_tasks > 0 else 0
    metrics['avg_delay'] = metrics['total_delay'] / n_tasks if n_tasks > 0 else 0
    metrics['avg_aoi'] = metrics['total_aoi'] / valid_aoi_tasks if valid_aoi_tasks > 0 else 0
    
    return metrics


def analyze_aoi_characteristics(system):
    """分析系统的AoI特性"""
    print("\nTask AoI Characteristics Analysis:")
    print("-" * 60)
    
    task_types = {}
    for task in system.tasks:
        task_type = get_task_type(task)
        if task_type not in task_types:
            task_types[task_type] = {
                'count': 0,
                'update_intervals': [],
                'max_aois': []
            }
        
        task_types[task_type]['count'] += 1
        if hasattr(task, 'update_interval') and task.update_interval is not None:
            task_types[task_type]['update_intervals'].append(task.update_interval)
        if hasattr(task, 'max_aoi') and task.max_aoi is not None:
            task_types[task_type]['max_aois'].append(task.max_aoi)
    
    print(f"{'Task Type':<20} {'Count':<8} {'Avg Update Interval(s)':<20} {'Avg Max AoI(s)':<15}")
    print("-" * 60)
    
    for task_type, data in task_types.items():
        avg_interval = np.mean(data['update_intervals']) if data['update_intervals'] else 0
        avg_max_aoi = np.mean(data['max_aois']) if data['max_aois'] else 0
        print(f"{task_type:<20} {data['count']:<8} {avg_interval:<20.3f} {avg_max_aoi:<15.3f}")


def analyze_balanced_results(results, system):
    """分析平衡权重的实验结果"""
    print("\nAlgorithm Performance Comparison (Balanced Weights):")
    print("-" * 90)
    print(f"{'Algorithm':<10} {'Avg Fitness':<12} {'Avg Energy(J)':<14} {'Avg Delay(s)':<14} {'Avg AoI(s)':<12} {'AoI Violation(%)':<16}")
    print("-" * 90)
    
    for alg_name, data in results.items():
        # 计算平均指标
        metrics_list = data['detailed_metrics']
        avg_energy = np.mean([m['avg_energy'] for m in metrics_list])
        avg_delay = np.mean([m['avg_delay'] for m in metrics_list])
        avg_aoi = np.mean([m['avg_aoi'] for m in metrics_list])
        
        # 计算AoI违规率
        total_tasks = len(system.tasks)
        aoi_tasks = sum(1 for t in system.tasks if hasattr(t, 'update_interval') and t.update_interval is not None)
        avg_aoi_violations = np.mean([m['aoi_violations'] for m in metrics_list])
        aoi_violation_rate = (avg_aoi_violations / aoi_tasks * 100) if aoi_tasks > 0 else 0
        
        print(f"{alg_name:<10} {data['mean_fitness']:<12.6f} {avg_energy:<14.6f} "
              f"{avg_delay:<14.6f} {avg_aoi:<12.6f} {aoi_violation_rate:<16.2f}")


def plot_balanced_results(results, system, save_dir='results/aoi/'):
    """生成平衡权重的实验结果图表（英文版本）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. 任务类型分布及其特征（包含AoI）
    plot_task_type_characteristics(system, save_dir)
    
    # 2. 算法收敛曲线对比
    plot_convergence_comparison(results, save_dir)
    
    # 3. 总体性能指标
    plot_overall_performance_metrics(results, system, save_dir)
    
    # 4. 能耗、延迟和AoI对比
    plot_detailed_performance_comparison(results, system, save_dir)
    
    # 5. 任务分配对比
    plot_task_allocation_comparison(results, system, save_dir)
    
    print(f"\nAll figures have been saved to {save_dir}:")
    print("  - task_type_characteristics.png")
    print("  - convergence_comparison.png")
    print("  - overall_performance_metrics.png")
    print("  - performance_comparison.png")
    print("  - task_allocation_comparison.png")


def plot_task_type_characteristics(system, save_dir):
    """绘制任务类型分布及其特征（包含AoI）"""
    # 统计任务类型
    task_types = {}
    for task in system.tasks:
        task_type = get_task_type(task)
        if task_type not in task_types:
            task_types[task_type] = {
                'count': 0,
                'workloads': [],  # 计算工作量 = data_size * computation_complexity
                'data_sizes': [],
                'deadlines': [],
                'update_intervals': [],
                'max_aois': []
            }
        
        task_types[task_type]['count'] += 1
        
        # 计算工作量（CPU cycles）
        workload = task.data_size * task.computation_complexity
        task_types[task_type]['workloads'].append(workload / 1e9)  # 转换为G cycles
        
        # 数据大小
        task_types[task_type]['data_sizes'].append(task.data_size / (1024 * 1024 * 8))  # 转换为MB (比特转字节)
        
        # 截止时间
        task_types[task_type]['deadlines'].append(task.max_delay)
        
        # AoI相关参数
        if hasattr(task, 'update_interval') and task.update_interval is not None:
            task_types[task_type]['update_intervals'].append(task.update_interval)
        if hasattr(task, 'max_aoi') and task.max_aoi is not None:
            task_types[task_type]['max_aois'].append(task.max_aoi)
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准备数据
    labels = list(task_types.keys())
    counts = [task_types[t]['count'] for t in labels]
    
    # 替换标签为英文
    label_mapping = {
        'compute_intensive': 'Compute\nIntensive',
        'data_intensive': 'Data\nIntensive',
        'realtime_sensitive': 'Real-time\nSensitive',
        'lightweight': 'Lightweight'
    }
    english_labels = [label_mapping.get(label, label) for label in labels]
    
    # 1. 任务类型分布（饼图）
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    wedges, texts, autotexts = ax1.pie(counts, labels=english_labels, colors=colors_pie, 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Task Type Distribution', fontsize=14, fontweight='bold')
    
    # 2. 工作负载分布（箱线图）
    workload_data = [task_types[t]['workloads'] for t in labels]
    bp1 = ax2.boxplot(workload_data, labels=english_labels, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors_pie):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Workload (G cycles)', fontsize=12)
    ax2.set_title('Workload Distribution by Task Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 更新间隔分布（箱线图）
    update_interval_data = [task_types[t]['update_intervals'] for t in labels]
    bp2 = ax3.boxplot(update_interval_data, labels=english_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors_pie):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Update Interval (s)', fontsize=12)
    ax3.set_title('Update Interval Distribution by Task Type', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 任务特征统计表
    ax4.axis('tight')
    ax4.axis('off')
    
    # 创建统计表格数据
    table_data = []
    headers = ['Task Type', 'Count', 'Avg Workload\n(G cycles)', 'Avg Data Size\n(MB)', 
               'Avg Update\nInterval (s)', 'Avg Max\nAoI (s)']
    
    for i, task_type in enumerate(labels):
        row_data = [
            english_labels[i],
            task_types[task_type]['count'],
            f"{np.mean(task_types[task_type]['workloads']):.2f}" if task_types[task_type]['workloads'] else "N/A",
            f"{np.mean(task_types[task_type]['data_sizes']):.2f}" if task_types[task_type]['data_sizes'] else "N/A",
            f"{np.mean(task_types[task_type]['update_intervals']):.3f}" if task_types[task_type]['update_intervals'] else "N/A",
            f"{np.mean(task_types[task_type]['max_aois']):.3f}" if task_types[task_type]['max_aois'] else "N/A"
        ]
        table_data.append(row_data)
    
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.18, 0.12, 0.18, 0.18, 0.17, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Task Characteristics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_type_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_comparison(results, save_dir):
    """绘制算法收敛曲线对比"""
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (algo_name, data) in enumerate(results.items()):
        avg_history = data['avg_fitness_history']
        std_history = data['std_fitness_history']
        
        x = np.arange(len(avg_history))
        color = colors[idx % len(colors)]
        
        plt.plot(x, avg_history, label=algo_name, color=color, 
                linewidth=2.5, linestyle=linestyles[idx % len(linestyles)])
        plt.fill_between(x, avg_history - std_history, avg_history + std_history, 
                        alpha=0.15, color=color)
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Fitness Value', fontsize=14)
    plt.title('Algorithm Convergence Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 添加局部放大图
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(plt.gca(), width="40%", height="40%", loc='center right',
                      bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gca().transAxes)
    
    # 放大最后50次迭代
    start_idx = max(0, len(x) - 50)
    for idx, (algo_name, data) in enumerate(results.items()):
        avg_history = data['avg_fitness_history']
        color = colors[idx % len(colors)]
        axins.plot(x[start_idx:], avg_history[start_idx:], color=color, 
                  linewidth=2, linestyle=linestyles[idx % len(linestyles)])
    
    axins.grid(True, alpha=0.3)
    axins.set_xlim(x[start_idx], x[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_overall_performance_metrics(results, system, save_dir):
    """绘制总体性能指标"""
    # 准备数据
    algorithms = list(results.keys())
    metrics_data = []
    
    for algo_name in algorithms:
        metrics_list = results[algo_name]['detailed_metrics']
        metrics_data.append({
            'fitness': results[algo_name]['mean_fitness'],
            'fitness_std': results[algo_name]['std_fitness'],
            'energy': np.mean([m['avg_energy'] for m in metrics_list]),
            'delay': np.mean([m['avg_delay'] for m in metrics_list]),
            'aoi': np.mean([m['avg_aoi'] for m in metrics_list]),
            'violations': np.mean([m['aoi_violations'] for m in metrics_list])
        })
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # 1. 适应度值对比（带误差条）
    fitness_values = [m['fitness'] for m in metrics_data]
    fitness_stds = [m['fitness_std'] for m in metrics_data]
    
    bars1 = ax1.bar(x, fitness_values, width, yerr=fitness_stds, 
                    capsize=5, color='skyblue', alpha=0.8)
    ax1.set_xlabel('Algorithms', fontsize=12)
    ax1.set_ylabel('Average Fitness Value', fontsize=12)
    ax1.set_title('Average Fitness Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    
    # 添加数值标签
    for i, (bar, val, std) in enumerate(zip(bars1, fitness_values, fitness_stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 平均能耗对比
    energy_values = [m['energy'] for m in metrics_data]
    bars2 = ax2.bar(x, energy_values, width, color='lightgreen', alpha=0.8)
    ax2.set_xlabel('Algorithms', fontsize=12)
    ax2.set_ylabel('Average Energy Consumption (J)', fontsize=12)
    ax2.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    
    for bar, val in zip(bars2, energy_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_values)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. 平均延迟对比
    delay_values = [m['delay'] for m in metrics_data]
    bars3 = ax3.bar(x, delay_values, width, color='lightcoral', alpha=0.8)
    ax3.set_xlabel('Algorithms', fontsize=12)
    ax3.set_ylabel('Average Delay (s)', fontsize=12)
    ax3.set_title('Response Time Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms)
    
    for bar, val in zip(bars3, delay_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(delay_values)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. 平均AoI对比
    aoi_values = [m['aoi'] for m in metrics_data]
    bars4 = ax4.bar(x, aoi_values, width, color='lightyellow', edgecolor='orange', 
                    linewidth=2, alpha=0.8)
    ax4.set_xlabel('Algorithms', fontsize=12)
    ax4.set_ylabel('Average AoI (s)', fontsize=12)
    ax4.set_title('Age of Information Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms)
    
    for bar, val in zip(bars4, aoi_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(aoi_values)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics_detail.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 归一化性能雷达图（单独的图）
    categories = ['Energy\nEfficiency', 'Response\nTime', 'AoI\nPerformance', 'Reliability']
    
    fig_radar = plt.figure(figsize=(8, 8))
    ax_radar = fig_radar.add_subplot(111, projection='polar')
    
    # 获取AoI任务数
    aoi_tasks = sum(1 for t in system.tasks if hasattr(t, 'update_interval') and t.update_interval is not None)
    
    # 归一化数据（越小越好的指标取反）
    max_energy = max(m['energy'] for m in metrics_data)
    max_delay = max(m['delay'] for m in metrics_data)
    max_aoi = max(m['aoi'] for m in metrics_data)
    max_violations = max(m['violations'] for m in metrics_data)
    
    for idx, (algo_name, metrics) in enumerate(zip(algorithms, metrics_data)):
        values = [
            1 - metrics['energy'] / max_energy if max_energy > 0 else 1,
            1 - metrics['delay'] / max_delay if max_delay > 0 else 1,
            1 - metrics['aoi'] / max_aoi if max_aoi > 0 else 1,
            1 - metrics['violations'] / max_violations if max_violations > 0 else 1
        ]
        values += values[:1]  # 闭合
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=algo_name, 
                     color=colors[idx % len(colors)])
        ax_radar.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=12)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Overall Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax_radar.grid(True)
    
    plt.tight_layout()
    fig_radar.savefig(os.path.join(save_dir, 'overall_performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_radar)


def plot_detailed_performance_comparison(results, system, save_dir):
    """绘制详细的性能对比（能耗、延迟、AoI）"""
    algorithms = list(results.keys())
    
    # 收集每次运行的详细数据
    energy_data = []
    delay_data = []
    aoi_data = []
    
    for algo_name in algorithms:
        metrics_list = results[algo_name]['detailed_metrics']
        energy_data.append([m['total_energy'] for m in metrics_list])
        delay_data.append([m['total_delay'] for m in metrics_list])
        aoi_data.append([m['total_aoi'] for m in metrics_list])
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 总能耗对比（箱线图）
    bp1 = ax1.boxplot(energy_data, labels=algorithms, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Total Energy Consumption (J)', fontsize=12)
    ax1.set_title('Total Energy Consumption Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加平均值标记
    for i, data in enumerate(energy_data):
        ax1.plot(i+1, np.mean(data), 'r*', markersize=10)
    
    # 2. 总延迟对比（箱线图）
    bp2 = ax2.boxplot(delay_data, labels=algorithms, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Total Delay (s)', fontsize=12)
    ax2.set_title('Total Delay Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, data in enumerate(delay_data):
        ax2.plot(i+1, np.mean(data), 'r*', markersize=10)
    
    # 3. 总AoI对比（箱线图）
    bp3 = ax3.boxplot(aoi_data, labels=algorithms, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Total AoI (s)', fontsize=12)
    ax3.set_title('Total Age of Information Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, data in enumerate(aoi_data):
        ax3.plot(i+1, np.mean(data), 'r*', markersize=10)
    
    # 添加整体标题
    fig.suptitle('Performance Comparison: Energy, Delay, and AoI', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_task_allocation_comparison(results, system, save_dir):
    """绘制任务分配对比图"""
    algorithms = list(results.keys())
    allocations = []
    
    # 使用最优解进行任务分配分析
    for algo_name in algorithms:
        best_idx = np.argmin(results[algo_name]['best_fitness_values'])
        best_solution = results[algo_name]['best_solutions'][best_idx]
        
        system.apply_solution(best_solution)
        
        allocation = {'device': 0, 'edge': 0, 'cloud': 0}
        for task in system.tasks:
            if task.execution_location == 'device':
                allocation['device'] += 1
            elif task.execution_location == 'edge':
                allocation['edge'] += 1
            elif task.execution_location == 'cloud':
                allocation['cloud'] += 1
        
        allocations.append(allocation)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # 1. 堆积柱状图
    device_counts = [alloc['device'] for alloc in allocations]
    edge_counts = [alloc['edge'] for alloc in allocations]
    cloud_counts = [alloc['cloud'] for alloc in allocations]
    
    p1 = ax1.bar(x, device_counts, width, label='Device', color='#87CEEB', alpha=0.9)
    p2 = ax1.bar(x, edge_counts, width, bottom=device_counts, label='Edge', 
                 color='#98FB98', alpha=0.9)
    p3 = ax1.bar(x, cloud_counts, width, 
                bottom=np.array(device_counts) + np.array(edge_counts), 
                label='Cloud', color='#FFB6C1', alpha=0.9)
    
    ax1.set_xlabel('Algorithms', fontsize=12)
    ax1.set_ylabel('Number of Tasks', fontsize=12)
    ax1.set_title('Task Allocation Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    ax1.legend(loc='upper right')
    
    # 添加数值标签
    for i, (device, edge, cloud) in enumerate(zip(device_counts, edge_counts, cloud_counts)):
        if device > 0:
            ax1.text(i, device/2, str(device), ha='center', va='center', fontweight='bold')
        if edge > 0:
            ax1.text(i, device + edge/2, str(edge), ha='center', va='center', fontweight='bold')
        if cloud > 0:
            ax1.text(i, device + edge + cloud/2, str(cloud), ha='center', va='center', fontweight='bold')
    
    # 2. 百分比堆积图
    total_tasks = len(system.tasks)
    device_pcts = [d/total_tasks*100 for d in device_counts]
    edge_pcts = [e/total_tasks*100 for e in edge_counts]
    cloud_pcts = [c/total_tasks*100 for c in cloud_counts]
    
    p4 = ax2.bar(x, device_pcts, width, label='Device', color='#87CEEB', alpha=0.9)
    p5 = ax2.bar(x, edge_pcts, width, bottom=device_pcts, label='Edge', 
                 color='#98FB98', alpha=0.9)
    p6 = ax2.bar(x, cloud_pcts, width, 
                bottom=np.array(device_pcts) + np.array(edge_pcts), 
                label='Cloud', color='#FFB6C1', alpha=0.9)
    
    ax2.set_xlabel('Algorithms', fontsize=12)
    ax2.set_ylabel('Percentage of Tasks (%)', fontsize=12)
    ax2.set_title('Task Allocation Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right')
    
    # 添加百分比标签
    for i, (device, edge, cloud) in enumerate(zip(device_pcts, edge_pcts, cloud_pcts)):
        if device > 5:  # 只显示大于5%的标签
            ax2.text(i, device/2, f'{device:.1f}%', ha='center', va='center', fontsize=10)
        if edge > 5:
            ax2.text(i, device + edge/2, f'{edge:.1f}%', ha='center', va='center', fontsize=10)
        if cloud > 5:
            ax2.text(i, device + edge + cloud/2, f'{cloud:.1f}%', ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_allocation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories, fontsize=12)
    # ax.set_ylim(0, 1)
    # ax.set_title('算法综合性能对比（平衡权重）', fontsize=14, pad=20)
    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    # ax.grid(True)
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'balanced_radar_comparison.png'), dpi=300, bbox_inches='tight')
    # plt.close()
    
    # print(f"\n图表已保存到 {save_dir} 目录：")
    # print("  - balanced_convergence.png")
    # print("  - balanced_metrics_comparison.png")
    # print("  - balanced_distribution_comparison.png")
    # print("  - balanced_radar_comparison.png")


def run_complete_aoi_experiment():
    """运行完整的AoI实验（主函数）"""
    system, results = run_aoi_experiment()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = 'results/aoi/'
    os.makedirs(save_dir, exist_ok=True)
    
    result_file = os.path.join(save_dir, f'aoi_experiment_results_{timestamp}.json')
    
    json_results = {}
    for algo_name, algo_results in results.items():
        json_results[algo_name] = {
            'fitness': algo_results['best_fitness_values'],
            'mean_fitness': float(algo_results['mean_fitness']),
            'std_fitness': float(algo_results['std_fitness']),
            'best_fitness': float(algo_results['best_fitness']),
            'avg_energy': float(np.mean([m['avg_energy'] for m in algo_results['detailed_metrics']])),
            'avg_delay': float(np.mean([m['avg_delay'] for m in algo_results['detailed_metrics']])),
            'avg_aoi': float(np.mean([m['avg_aoi'] for m in algo_results['detailed_metrics']])),
            'aoi_violations': float(np.mean([m['aoi_violations'] for m in algo_results['detailed_metrics']]))
        }
    
    with open(result_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nExperimental results saved to: {result_file}")
    
    return results


# 主函数
def main():
    """Main function - Run balanced weight AoI experiment"""
    run_complete_aoi_experiment()


if __name__ == "__main__":
    main()