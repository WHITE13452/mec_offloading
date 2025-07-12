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

# 设置全局字体大小（比文章10号字体小半号，约9.5号）
FONT_SIZE_TITLE = 13  # 标题字体
FONT_SIZE_LABEL = 11  # 轴标签字体
FONT_SIZE_TICK = 10   # 刻度字体
FONT_SIZE_LEGEND = 10 # 图例字体
FONT_SIZE_TEXT = 10   # 文本标注字体

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
        'convergence_title': 'Algorithm Convergence Comparison',
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
        'performance_title': 'Algorithm Performance Comparison',
        'algorithm': 'Algorithm',
        'avg_fitness': 'Average Fitness',
        'performance_comparison': 'Algorithm Performance Comparison',
        'task_type_dist': 'Task Type Distribution',
        'workload_dist': 'Workload Distribution by Task Type',
        'workload': 'Workload (G cycles)',
        'update_interval': 'Update Interval (s)',
        'update_interval_dist': 'Update Interval Distribution',
        'energy_comparison': 'Energy Consumption Comparison',
        'total_energy': 'Total Energy Consumption (J)',
        'response_time': 'Response Time Comparison',
        'total_delay': 'Total Delay (s)',
        'aoi_comparison': 'Age of Information Comparison',
        'total_aoi': 'Total AoI (s)',
        'task_allocation': 'Task Allocation Comparison',
        'num_tasks': 'Number of Tasks',
        'device': 'Device',
        'edge': 'Edge',
        'cloud': 'Cloud'
    }
}

def get_label(key):
    """获取标签文本"""
    lang = 'en' if USE_ENGLISH_LABELS else 'zh'
    return LABELS[lang].get(key, key)

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


def save_experiment_results(results, system, save_dir='results/aoi/'):
    """保存实验结果到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备保存的数据
    save_data = {
        'timestamp': timestamp,
        'system_config': {
            'num_devices': len(system.devices),
            'num_edge_servers': len(system.edge_servers),
            'num_cloud_servers': len(system.cloud_servers),
            'num_tasks': len(system.tasks)
        },
        'algorithm_results': {}
    }
    
    # 保存每个算法的结果
    for algo_name, algo_results in results.items():
        save_data['algorithm_results'][algo_name] = {
            'fitness_values': algo_results['best_fitness_values'],
            'mean_fitness': float(algo_results['mean_fitness']),
            'std_fitness': float(algo_results['std_fitness']),
            'best_fitness': float(algo_results['best_fitness']),
            'avg_fitness_history': algo_results['avg_fitness_history'].tolist(),
            'std_fitness_history': algo_results['std_fitness_history'].tolist(),
            'detailed_metrics': algo_results['detailed_metrics']
        }
    
    # 保存到文件
    result_file = os.path.join(save_dir, f'experiment_results_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(save_data, f, indent=4)
    
    print(f"\nExperiment results saved to: {result_file}")
    return result_file


def load_experiment_results(result_file):
    """从JSON文件加载实验结果"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # 转换numpy数组
    results = {}
    for algo_name, algo_data in data['algorithm_results'].items():
        results[algo_name] = {
            'best_fitness_values': algo_data['fitness_values'],
            'mean_fitness': algo_data['mean_fitness'],
            'std_fitness': algo_data['std_fitness'],
            'best_fitness': algo_data['best_fitness'],
            'avg_fitness_history': np.array(algo_data['avg_fitness_history']),
            'std_fitness_history': np.array(algo_data['std_fitness_history']),
            'detailed_metrics': algo_data['detailed_metrics']
        }
    
    return results, data['system_config']


def run_aoi_experiment(system_config=None, load_results_from=None):
    """运行考虑AoI的平衡权重实验"""
    
    if load_results_from:
        # 从文件加载结果
        print(f"Loading results from: {load_results_from}")
        results, system_config_loaded = load_experiment_results(load_results_from)
        
        # 创建系统用于绘图
        system = create_aoi_aware_system(
            num_devices=system_config_loaded['num_devices'],
            num_edge_servers=system_config_loaded['num_edge_servers'],
            num_cloud_servers=system_config_loaded['num_cloud_servers'],
            num_tasks=system_config_loaded['num_tasks']
        )
    else:
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
        
        # 保存实验结果
        save_experiment_results(results, system)
    
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
    """生成平衡权重的实验结果图表（修改版）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. 任务类型分布（单独的饼图）
    plot_task_type_distribution(system, save_dir)
    
    # 2. 更新间隔分布（单独的箱线图）
    plot_update_interval_distribution(system, save_dir)
    
    # 3. 算法收敛曲线对比
    plot_convergence_comparison(results, save_dir)
    
    # 4. 平均适应度对比（带误差条的柱状图）
    plot_fitness_comparison(results, save_dir)
    
    # 5. 性能指标对比（能耗、延迟、AoI的箱线图）
    plot_performance_boxplots(results, system, save_dir)
    
    # 6. 任务分配对比（简化版，只保留数值柱状图）
    plot_task_allocation_simple(results, system, save_dir)
    
    print(f"\nAll figures have been saved to {save_dir}:")
    print("  - task_type_distribution.png")
    print("  - update_interval_distribution.png")
    print("  - convergence_comparison.png")
    print("  - fitness_comparison.png")
    print("  - performance_boxplots.png")
    print("  - task_allocation_comparison.png")


def plot_task_type_distribution(system, save_dir):
    """绘制任务类型分布（单独的饼图）"""
    # 统计任务类型
    task_types = {}
    for task in system.tasks:
        task_type = get_task_type(task)
        if task_type not in task_types:
            task_types[task_type] = 0
        task_types[task_type] += 1
    
    # 准备数据
    labels = list(task_types.keys())
    counts = list(task_types.values())
    
    # 替换标签为英文
    label_mapping = {
        'compute_intensive': 'Compute\nIntensive',
        'data_intensive': 'Data\nIntensive',
        'realtime_sensitive': 'Real-time\nSensitive',
        'lightweight': 'Lightweight'
    }
    english_labels = [label_mapping.get(label, label) for label in labels]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    wedges, texts, autotexts = ax.pie(counts, labels=english_labels, colors=colors_pie, 
                                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': FONT_SIZE_TEXT})
    
    # 设置标签字体大小
    for text in texts:
        text.set_fontsize(FONT_SIZE_LABEL)
    
    ax.set_title(get_label('task_type_dist'), fontsize=FONT_SIZE_TITLE, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_type_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_update_interval_distribution(system, save_dir):
    """绘制更新间隔分布（单独的箱线图）"""
    # 统计任务类型的更新间隔
    task_types = {}
    for task in system.tasks:
        task_type = get_task_type(task)
        if task_type not in task_types:
            task_types[task_type] = []
        
        if hasattr(task, 'update_interval') and task.update_interval is not None:
            task_types[task_type].append(task.update_interval)
    
    # 准备数据
    labels = list(task_types.keys())
    update_interval_data = [task_types[t] for t in labels]
    
    # 替换标签为英文
    label_mapping = {
        'compute_intensive': 'Compute\nIntensive',
        'data_intensive': 'Data\nIntensive',
        'realtime_sensitive': 'Real-time\nSensitive',
        'lightweight': 'Lightweight'
    }
    english_labels = [label_mapping.get(label, label) for label in labels]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_box = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    bp = ax.boxplot(update_interval_data, labels=english_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(get_label('update_interval'), fontsize=FONT_SIZE_LABEL)
    ax.set_title(get_label('update_interval_dist'), fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'update_interval_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_comparison(results, save_dir):
    """绘制算法收敛曲线对比"""
    plt.figure(figsize=(10, 6))
    
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
    
    plt.xlabel(get_label('iterations'), fontsize=FONT_SIZE_LABEL)
    plt.ylabel(get_label('fitness'), fontsize=FONT_SIZE_LABEL)
    plt.title(get_label('convergence_title'), fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
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
    axins.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK-1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_fitness_comparison(results, save_dir):
    """绘制平均适应度对比（带误差条的柱状图）"""
    # 准备数据
    algorithms = list(results.keys())
    fitness_values = [results[alg]['mean_fitness'] for alg in algorithms]
    fitness_stds = [results[alg]['std_fitness'] for alg in algorithms]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(algorithms))
    width = 0.6
    
    bars = ax.bar(x, fitness_values, width, yerr=fitness_stds, 
                   capsize=5, color='skyblue', alpha=0.8)
    
    ax.set_xlabel(get_label('algorithm'), fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(get_label('avg_fitness'), fontsize=FONT_SIZE_LABEL)
    ax.set_title(get_label('performance_title'), fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis='y', which='major', labelsize=FONT_SIZE_TICK)
    
    # 添加数值标签
    for i, (bar, val, std) in enumerate(zip(bars, fitness_values, fitness_stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=FONT_SIZE_TEXT)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fitness_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_boxplots(results, system, save_dir):
    """绘制性能指标对比（能耗、延迟、AoI的箱线图）"""
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
    
    # 创建三个单独的图
    metrics = [
        (energy_data, get_label('total_energy'), get_label('energy_comparison'), 'energy_comparison.png'),
        (delay_data, get_label('total_delay'), get_label('response_time'), 'delay_comparison.png'),
        (aoi_data, get_label('total_aoi'), get_label('aoi_comparison'), 'aoi_comparison.png')
    ]
    
    for data, ylabel, title, filename in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
        
        # 添加平均值标记
        for i, d in enumerate(data):
            ax.plot(i+1, np.mean(d), 'r*', markersize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def plot_task_allocation_simple(results, system, save_dir):
    """绘制任务分配对比图（简化版，只保留数值柱状图）"""
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
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # 堆积柱状图
    device_counts = [alloc['device'] for alloc in allocations]
    edge_counts = [alloc['edge'] for alloc in allocations]
    cloud_counts = [alloc['cloud'] for alloc in allocations]
    
    p1 = ax.bar(x, device_counts, width, label=get_label('device'), color='#87CEEB', alpha=0.9)
    p2 = ax.bar(x, edge_counts, width, bottom=device_counts, label=get_label('edge'), 
                 color='#98FB98', alpha=0.9)
    p3 = ax.bar(x, cloud_counts, width, 
                bottom=np.array(device_counts) + np.array(edge_counts), 
                label=get_label('cloud'), color='#FFB6C1', alpha=0.9)
    
    ax.set_xlabel(get_label('algorithm'), fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(get_label('num_tasks'), fontsize=FONT_SIZE_LABEL)
    ax.set_title(get_label('task_allocation'), fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis='y', which='major', labelsize=FONT_SIZE_TICK)
    ax.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND)
    
    # 添加数值标签
    for i, (device, edge, cloud) in enumerate(zip(device_counts, edge_counts, cloud_counts)):
        if device > 0:
            ax.text(i, device/2, str(device), ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE_TEXT)
        if edge > 0:
            ax.text(i, device + edge/2, str(edge), ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE_TEXT)
        if cloud > 0:
            ax.text(i, device + edge + cloud/2, str(cloud), ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE_TEXT)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_allocation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run_complete_aoi_experiment():
    """运行完整的AoI实验（主函数）"""
    # 可以选择从文件加载结果或重新运行实验
    # load_results_from = 'results/aoi/experiment_results_20241209_143025.json'  # 如果要加载之前的结果
    load_results_from = 'results/aoi/experiment_results_20250712_015516.json'  # 设置为None表示重新运行实验
    
    system, results = run_aoi_experiment(load_results_from=load_results_from)
    
    return results


# 主函数
def main():
    """Main function - Run balanced weight AoI experiment"""
    run_complete_aoi_experiment()


if __name__ == "__main__":
    main()