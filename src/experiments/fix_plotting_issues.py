import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

FONT_SIZE_TITLE = 26    # 标题字体 (增大)
FONT_SIZE_LABEL = 24    # 轴标签字体 (增大)
FONT_SIZE_TICK = 22     # 刻度字体 (增大)
FONT_SIZE_LEGEND = 22   # 图例字体 (增大)
FONT_SIZE_TEXT = 22     # 文本标注字体 (增大)

# 全局变量控制是否使用英文标签
USE_ENGLISH_LABELS = True

# 根据提供的RGB值设置颜色
COLORS = [
    (136/255, 179/255, 214/255),   # 浅蓝色 R:100, G:143, B:255
    (252/255, 163/255, 17/255),   # 浅黄色 R:255, G:213, B:128
    (228/255, 144/255, 117/255),   # 粉蓝色 R:176, G:224, B:230
    (169/255, 84/255, 59/255),   # 浅黄色 R:255, G:213, B:128
]

# 不同的填充样式，适合黑白打印
HATCHES = ['/', '\\', 'x', '+', 'o', 'O', '.', '*']

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

def setup_plot_style():
    """设置绘图样式为Times New Roman字体"""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern字体作为数学公式字体
    plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    plt.rcParams['figure.titlesize'] = FONT_SIZE_TITLE
    plt.rcParams['figure.dpi'] = 300

def get_label(key):
    """获取标签文本"""
    lang = 'en' if USE_ENGLISH_LABELS else 'zh'
    return LABELS[lang].get(key, key)

def plot_convergence_comparison_large(results, save_dir):
    """绘制算法收敛曲线对比 - 适合黑白打印"""
    setup_plot_style()  # 应用Times New Roman字体
    
    plt.figure(figsize=(12, 8))
    
    # 自定义线型和标记，提高黑白打印可区分性
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    for idx, (algo_name, data) in enumerate(results.items()):
        avg_history = data['avg_fitness_history']
        std_history = data['std_fitness_history']
        
        x = np.arange(len(avg_history))
        color = COLORS[idx % len(COLORS)]
        
        # 使用标记点增强黑白打印识别度，但不是每个点都标记
        mark_every = max(1, len(x) // 15)  # 每15个点标记一次
        
        plt.plot(x, avg_history, 
                label=algo_name, 
                color=color,
                linestyle=linestyles[idx % len(linestyles)],
                marker=markers[idx % len(markers)],
                markevery=mark_every,
                linewidth=2.5)
        
        # 添加带纹理的填充区域
        plt.fill_between(x, avg_history - std_history, avg_history + std_history, 
                        alpha=0.15, color=color, hatch=HATCHES[idx % len(HATCHES)])
    
    plt.xlabel(get_label('iterations'), fontsize=FONT_SIZE_LABEL)
    plt.ylabel(get_label('fitness'), fontsize=FONT_SIZE_LABEL)
    plt.title(get_label('convergence_title'), fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    plt.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    # 调整y轴范围，使曲线更清晰
    y_min = min([min(data['avg_fitness_history']) for data in results.values()])
    y_max = max([max(data['avg_fitness_history'][:20]) for data in results.values()])  # 只看前20次迭代的最大值
    plt.ylim(y_min * 0.9, y_max * 1.1)
    
    # 添加局部放大图
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(plt.gca(), width="30%", height="40%", loc='center right',
                      bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gca().transAxes)
    
    # 放大最后50次迭代
    start_idx = max(0, len(x) - 50)
    for idx, (algo_name, data) in enumerate(results.items()):
        avg_history = data['avg_fitness_history']
        color = COLORS[idx % len(COLORS)]
        
        # 在放大图中也使用标记和线型
        axins.plot(x[start_idx:], avg_history[start_idx:], 
                  color=color, 
                  linestyle=linestyles[idx % len(linestyles)],
                  marker=markers[idx % len(markers)],
                  markevery=max(1, len(x[start_idx:]) // 5),
                  linewidth=1.5)
    
    axins.grid(True, alpha=0.3)
    axins.set_xlim(x[start_idx], x[-1])
    axins.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK-2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_task_allocation_simple_fixed(results, system_config, save_dir):
    """绘制任务分配对比图（堆叠柱状图）- 适合黑白打印"""
    setup_plot_style()  # 应用Times New Roman字体
    
    algorithms = list(results.keys())
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
                'cloud': int(total_tasks * 0.1)    # 默认10%云端
            }
        
        allocations.append(allocation)
    
    # 创建图表 - 增加顶部空间容纳图例
    fig, ax = plt.subplots(figsize=(8, 7.5))  # 增加高度以容纳顶部图例
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # 堆积柱状图数据
    device_counts = [alloc['device'] for alloc in allocations]
    edge_counts = [alloc['edge'] for alloc in allocations]
    cloud_counts = [alloc['cloud'] for alloc in allocations]
    
    # 使用不同颜色和纹理的柱状图
    p1 = ax.bar(x, device_counts, width, 
               label=get_label('device'), 
               color=COLORS[0], 
               edgecolor='black',
               hatch=HATCHES[0])
               
    p2 = ax.bar(x, edge_counts, width, 
               bottom=device_counts, 
               label=get_label('edge'), 
               color=COLORS[1],
               edgecolor='black', 
               hatch=HATCHES[1])
               
    p3 = ax.bar(x, cloud_counts, width, 
               bottom=np.array(device_counts) + np.array(edge_counts), 
               label=get_label('cloud'), 
               color=COLORS[2],
               edgecolor='black',
               hatch=HATCHES[2])
    
    ax.set_xlabel(get_label('algorithm'), fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(get_label('num_tasks'), fontsize=FONT_SIZE_LABEL)
    ax.set_title(get_label('task_allocation'), fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis='y', which='major', labelsize=FONT_SIZE_TICK)
    
    # 修改图例位置 - 放在图表上方，并水平排列
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.20),  # 将y坐标从1.15增加到1.20
        ncol=3,
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        framealpha=1.0,
        fancybox=True,
        shadow=False,
    )
    
    # 添加数值标签（加粗以增强黑白打印效果）
    for i, (device, edge, cloud) in enumerate(zip(device_counts, edge_counts, cloud_counts)):
        if device > 0:
            ax.text(i, device/2, str(device), ha='center', va='center', 
                   fontweight='bold', fontsize=FONT_SIZE_TEXT)
        if edge > 0:
            ax.text(i, device + edge/2, str(edge), ha='center', va='center', 
                   fontweight='bold', fontsize=FONT_SIZE_TEXT)
        if cloud > 0:
            ax.text(i, device + edge + cloud/2, str(cloud), ha='center', va='center', 
                   fontweight='bold', fontsize=FONT_SIZE_TEXT)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 使用更大的上边距确保图例不被截断
    plt.subplots_adjust(top=0.8)  # 增加顶部边距
    
    # 保存图表
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
    
    print("Plotting convergence comparison with enhanced styling...")
    plot_convergence_comparison_large(results, save_dir)
    
    print("Plotting task allocation comparison with hatching for black-white printing...")
    plot_task_allocation_simple_fixed(results, system_config, save_dir)
    
    print(f"\nFigures saved to {save_dir}:")
    print("  - convergence_comparison.png (Times New Roman font, enhanced for b/w printing)")
    print("  - task_allocation_comparison.png (Times New Roman font, enhanced for b/w printing)")


# 示例使用
if __name__ == "__main__":
    # 替换为您的实验结果文件路径
    result_file = 'results/aoi/experiment_results_20241209_143025.json'
    
    # 重新绘制图表
    load_and_plot_results(result_file)