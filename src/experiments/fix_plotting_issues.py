import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 定义全局颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 全局变量控制是否使用英文标签
USE_ENGLISH_LABELS = True

# 设置全局字体大小 - 为收敛曲线图特别调整
FONT_SIZE_TITLE_LARGE = 16    # 收敛图标题字体（更大）
FONT_SIZE_LABEL_LARGE = 14    # 收敛图轴标签字体（更大）
FONT_SIZE_TICK_LARGE = 12     # 收敛图刻度字体（更大）
FONT_SIZE_LEGEND_LARGE = 12   # 收敛图图例字体（更大）

# 其他图的字体大小
FONT_SIZE_TITLE = 13  # 标题字体
FONT_SIZE_LABEL = 11  # 轴标签字体
FONT_SIZE_TICK = 10   # 刻度字体
FONT_SIZE_LEGEND = 10 # 图例字体
FONT_SIZE_TEXT = 10   # 文本标注字体

# 标签字典
LABELS = {
    'en': {
        'convergence_title': 'Algorithm Convergence Comparison',
        'iterations': 'Iterations',
        'fitness': 'Fitness Value',
        'algorithm': 'Algorithm',
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


def plot_convergence_comparison_large(results, save_dir):
    """绘制算法收敛曲线对比 - 增大版本"""
    plt.figure(figsize=(12, 8))  # 增大图表尺寸
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']
    
    # 调整线宽
    for idx, (algo_name, data) in enumerate(results.items()):
        avg_history = data['avg_fitness_history']
        std_history = data['std_fitness_history']
        
        x = np.arange(len(avg_history))
        color = colors[idx % len(colors)]
        
        plt.plot(x, avg_history, label=algo_name, color=color, 
                linewidth=3, linestyle=linestyles[idx % len(linestyles)])  # 增加线宽
        plt.fill_between(x, avg_history - std_history, avg_history + std_history, 
                        alpha=0.15, color=color)
    
    plt.xlabel(get_label('iterations'), fontsize=FONT_SIZE_LABEL_LARGE)
    plt.ylabel(get_label('fitness'), fontsize=FONT_SIZE_LABEL_LARGE)
    plt.title(get_label('convergence_title'), fontsize=FONT_SIZE_TITLE_LARGE, fontweight='bold', pad=15)
    plt.legend(fontsize=FONT_SIZE_LEGEND_LARGE, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LARGE)
    
    # 调整y轴范围，使曲线更清晰
    y_min = min([min(data['avg_fitness_history']) for data in results.values()])
    y_max = max([max(data['avg_fitness_history'][:20]) for data in results.values()])  # 只看前20次迭代的最大值
    plt.ylim(y_min * 0.9, y_max * 1.1)
    
    # 添加局部放大图
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(plt.gca(), width="45%", height="45%", loc='center right',
                      bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gca().transAxes)
    
    # 放大最后50次迭代
    start_idx = max(0, len(x) - 50)
    for idx, (algo_name, data) in enumerate(results.items()):
        avg_history = data['avg_fitness_history']
        color = colors[idx % len(colors)]
        axins.plot(x[start_idx:], avg_history[start_idx:], color=color, 
                  linewidth=2.5, linestyle=linestyles[idx % len(linestyles)])
    
    axins.grid(True, alpha=0.3)
    axins.set_xlim(x[start_idx], x[-1])
    axins.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LARGE-2)
    
    # 添加放大框标记
    from matplotlib.patches import Rectangle
    rect = Rectangle((x[start_idx], y_min * 0.9), 
                    x[-1] - x[start_idx], 
                    (y_max * 1.1 - y_min * 0.9) * 0.1,
                    facecolor='none', edgecolor='gray', linestyle='--', alpha=0.5)
    plt.gca().add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_task_allocation_simple_fixed(results, system_config, save_dir):
    """绘制任务分配对比图（修复版）"""
    algorithms = list(results.keys())
    
    # 创建默认分配数据（用于从加载的结果重建）
    allocations = []
    
    for algo_name in algorithms:
        # 使用最优解的索引
        best_idx = np.argmin(results[algo_name]['best_fitness_values'])
        
        # 从详细指标中获取任务分配信息
        if 'detailed_metrics' in results[algo_name] and len(results[algo_name]['detailed_metrics']) > best_idx:
            allocation = results[algo_name]['detailed_metrics'][best_idx]['task_allocation']
        else:
            # 如果没有详细指标，使用默认值
            print(f"Warning: No detailed metrics for {algo_name}, using default allocation")
            total_tasks = system_config.get('num_tasks', 40)
            allocation = {
                'device': int(total_tasks * 0.7),  # 默认70%本地
                'edge': int(total_tasks * 0.2),    # 默认20%边缘
                'cloud': int(total_tasks * 0.1)     # 默认10%云端
            }
        
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


def load_and_plot_results(result_file, save_dir='results/aoi/'):
    """从文件加载结果并重新绘图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载结果
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
    
    system_config = data['system_config']
    
    print("Plotting convergence comparison (large version)...")
    plot_convergence_comparison_large(results, save_dir)
    
    print("Plotting task allocation comparison (fixed version)...")
    plot_task_allocation_simple_fixed(results, system_config, save_dir)
    
    print(f"\nFigures saved to {save_dir}:")
    print("  - convergence_comparison.png (enlarged version)")
    print("  - task_allocation_comparison.png (fixed version)")


# 示例使用
if __name__ == "__main__":
    # 替换为您的实验结果文件路径
    result_file = 'results/aoi/experiment_results_20241209_143025.json'
    
    # 重新绘制图表
    load_and_plot_results(result_file)