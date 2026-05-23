"""
第四章实验 - RDHO五目标优化实验
对比算法: RDHO, RIME, DBO, TLBO-HHO
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import time

# 模型导入
from ..models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel

# 算法导入
from ..algorithms.rdho import RDHO
from ..algorithms.rime import RIME
from ..algorithms.dbo import DBO
from ..algorithms.tlbo_hho import TLBOHHO
from ..algorithms.cwtssa import CWTSSA  # 已注释
# from ..algorithms.mssa import MSSA

# 系统配置
from .realistic_system_setup import create_realistic_edge_computing_system, get_task_type

# ==================== 绘图样式设置（与第三章保持一致）====================

# 全局变量控制是否使用英文标签
USE_ENGLISH_LABELS = True

# 设置全局字体和字号 - 增大字号以匹配文档样式
FONT_SIZE_TITLE = 26    # 标题字体
FONT_SIZE_LABEL = 24    # 轴标签字体
FONT_SIZE_TICK = 22     # 刻度字体
FONT_SIZE_LEGEND = 22   # 图例字体
FONT_SIZE_TEXT = 22     # 文本标注字体

# 根据提供的RGB值设置颜色（与第三章保持一致）
COLORS = [
    (19/255, 33/255, 60/255),    # 深蓝色 R:019, G:033, B:060
    (252/255, 163/255, 17/255),  # 黄色 R:252, G:163, B:017
    (136/255, 179/255, 214/255), # 浅蓝色 R:136, G:179, B:214
    (200/255, 97/255, 52/255),   # 棕红色 R:200, G:097, B:052
    (100/255, 149/255, 237/255), # 矢车菊蓝 (第5种算法MSSA)
]

# 不同的填充样式，适合黑白打印
HATCHES = ['/', '\\', 'x', '+', 'o', 'O', '.', '*']

# 线型和标记符号
LINESTYLES = ['-', '--', '-.', ':', '-']
MARKERS = ['o', 's', '^', 'D', 'v']


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


def create_five_objective_system(num_devices=20, num_edge_servers=4,
                                  num_cloud_servers=2, num_tasks=40):
    """
    创建支持五目标优化的边缘计算系统

    扩展任务属性:
    - update_interval: 更新间隔
    - max_aoi: 最大可接受AoI
    - delay_sensitivity: 时延敏感度系数 β
    - energy_budget: 能耗预算
    """
    system = create_realistic_edge_computing_system(
        num_devices=num_devices,
        num_edge_servers=num_edge_servers,
        num_cloud_servers=num_cloud_servers,
        num_tasks=num_tasks
    )

    # 为任务添加五目标相关参数
    for task in system.tasks:
        task_type = get_task_type(task)

        # AoI参数
        if task_type == 'realtime_sensitive':
            task.update_interval = np.random.uniform(0.1, 0.3)
            task.max_aoi = np.random.uniform(0.5, 1.0)
            task.delay_sensitivity = np.random.uniform(0.8, 1.0)  # 高敏感度
        elif task_type == 'compute_intensive':
            task.update_interval = np.random.uniform(0.5, 1.0)
            task.max_aoi = np.random.uniform(2.0, 3.0)
            task.delay_sensitivity = np.random.uniform(0.3, 0.5)
        elif task_type == 'data_intensive':
            task.update_interval = np.random.uniform(0.8, 1.5)
            task.max_aoi = np.random.uniform(3.0, 5.0)
            task.delay_sensitivity = np.random.uniform(0.4, 0.6)
        else:  # lightweight
            task.update_interval = np.random.uniform(0.3, 0.6)
            task.max_aoi = np.random.uniform(1.0, 2.0)
            task.delay_sensitivity = np.random.uniform(0.5, 0.7)

        # 能耗预算（基于任务复杂度估算）
        task.energy_budget = task.data_size * task.computation_complexity * 1e-27 * 2

    return system


def run_five_objective_experiment(system=None, max_iter=150, population_size=50,
                                   n_runs=5, save_results=True):
    """
    运行五目标优化实验

    比较算法:
    1. RDHO (本章算法 - RIME-DBO混合优化器)
    2. RIME (基础RIME算法)
    3. DBO (基础DBO算法)
    4. TLBO-HHO (第三章算法)

    评价指标:
    - 综合适应度 (Fitness)
    - 总能耗 (Energy)
    - 最大时延 (Delay)
    - 平均AoI
    - 平均QoE
    - 公平性指数 (Fairness)
    - 约束满足率 (CSR)
    - 收敛速度
    - 运行时间
    """
    if system is None:
        system = create_five_objective_system()

    # 创建模型
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    qoe_model = QoEModel(system, delay_model, energy_model, aoi_model)
    fairness_model = FairnessModel(system, qoe_model)

    # 五目标权重设置
    weights = {
        'w_energy': 0.15,
        'w_delay': 0.15,
        'w_aoi': 0.20,
        'w_qoe': 0.25,
        'w_fairness': 0.25
    }

    # 定义算法（五目标优化）
    algorithms = {
        'RDHO': RDHO(
            system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
            max_iter=max_iter, population_size=population_size,
            **weights,
            producer_ratio=0.2,      # 生产者比例
            follower_ratio=0.7,      # 跟随者比例
            scout_ratio=0.1,         # 侦察者比例
            elite_ratio=0.1,         # 精英解比例
            base_penalty=1.0,        # 动态惩罚基础值
            penalty_alpha=2.0,       # 惩罚增长指数
            verbose=False
        ),
        'RIME': RIME(
            system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
            max_iter=max_iter, population_size=population_size,
            **weights,
            verbose=False
        ),
        'DBO': DBO(
            system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
            max_iter=max_iter, population_size=population_size,
            **weights,
            rolling_ratio=0.2,       # 滚球蜣螂比例
            breeding_ratio=0.2,      # 繁殖蜣螂比例
            foraging_ratio=0.4,      # 觅食蜣螂比例
            verbose=False
        ),
        'TLBO-HHO': TLBOHHO(
            system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
            max_iter=max_iter, population_size=population_size,
            **weights,
            verbose=False
        ),
        'CWTSSA': CWTSSA(
            system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
            max_iter=max_iter, population_size=population_size,
            **weights,
            PD=0.2, SD=0.1, ST=0.8,      # SSA参数
            w_max=0.9, w_min=0.4,         # 自适应权重参数
            mutation_prob=0.2,            # t分布变异概率
            mutation_scale=0.5,           # 变异幅度
            verbose=False
        ),
        # 'MSSA': MSSA(
        #     system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
        #     max_iter=max_iter, population_size=population_size,
        #     **weights,
        #     PD=0.2, SD=0.1, ST=0.8,       # SSA参数
        #     w_max=0.9, w_min=0.4,          # 惯性权重参数
        #     mutation_rate=0.1,             # 变异概率
        #     mutation_scale=0.5,            # 变异幅度
        #     termination_window=20,         # 终止检查窗口
        #     termination_threshold=1e-6,    # 终止阈值
        #     verbose=False
        # ),
    }

    results = {}

    for name, algorithm in algorithms.items():
        print(f"\n{'='*60}")
        print(f"运行算法: {name}")
        print(f"{'='*60}")

        run_results = {
            'fitness_history': [],
            'best_fitness': [],
            'energy': [],
            'delay': [],
            'aoi': [],
            'qoe': [],
            'fairness': [],
            'csr': [],
            'runtime': []
        }

        for run in range(n_runs):
            print(f"  运行 {run+1}/{n_runs}...", end=' ')

            # 重置任务状态
            for task in system.tasks:
                task.execution_location = None
                task.execution_node_id = None
                task.allocated_resource = None
                task.delay = None
                task.energy = None
                task.aoi = None

            # 运行优化
            start_time = time.time()
            best_solution, best_fitness, history = algorithm.optimize()
            runtime = time.time() - start_time

            # 计算详细指标
            metrics = calculate_detailed_metrics(
                best_solution, system, delay_model, energy_model,
                aoi_model, qoe_model, fairness_model
            )

            # 记录结果
            run_results['fitness_history'].append(history)
            run_results['best_fitness'].append(best_fitness)
            run_results['energy'].append(metrics['total_energy'])
            run_results['delay'].append(metrics['max_delay'])
            run_results['aoi'].append(metrics['avg_aoi'])
            run_results['qoe'].append(metrics['avg_qoe'])
            run_results['fairness'].append(metrics['fairness'])
            run_results['csr'].append(metrics['csr'])
            run_results['runtime'].append(runtime)

            print(f"Fitness: {best_fitness:.6f}, QoE: {metrics['avg_qoe']:.4f}, "
                  f"Fairness: {metrics['fairness']:.4f}")

        # 计算统计量
        # 处理可能不同长度的收敛历史（如MSSA使用早停策略）
        fitness_histories = run_results['fitness_history']
        max_len = max(len(h) for h in fitness_histories)
        padded_histories = []
        for h in fitness_histories:
            if len(h) < max_len:
                # 用最后一个值填充到相同长度
                h = list(h) + [h[-1]] * (max_len - len(h))
            padded_histories.append(h)

        results[name] = {
            'mean_fitness': np.mean(run_results['best_fitness']),
            'std_fitness': np.std(run_results['best_fitness']),
            'mean_energy': np.mean(run_results['energy']),
            'mean_delay': np.mean(run_results['delay']),
            'mean_aoi': np.mean(run_results['aoi']),
            'mean_qoe': np.mean(run_results['qoe']),
            'mean_fairness': np.mean(run_results['fairness']),
            'mean_csr': np.mean(run_results['csr']),
            'mean_runtime': np.mean(run_results['runtime']),
            'convergence': np.mean(padded_histories, axis=0).tolist(),
            'raw_results': run_results
        }

    # 保存结果
    if save_results:
        save_experiment_results(results, system)

    return results


def calculate_detailed_metrics(solution, system, delay_model, energy_model,
                               aoi_model, qoe_model, fairness_model):
    """计算详细性能指标"""
    try:
        # 应用解到系统
        apply_solution_to_system(solution, system)

        # 计算各项指标
        total_energy = 0
        delays = []
        aois = []
        qoes = []
        constraint_violations = 0

        for i, task in enumerate(system.tasks):
            # 能耗
            energy = energy_model.calculate_total_energy(task)
            total_energy += energy

            # 时延
            delay = delay_model.calculate_total_delay(task)
            delays.append(delay)

            # AoI
            aoi = aoi_model.calculate_average_aoi(task) if aoi_model else 0
            aois.append(aoi)

            # QoE
            qoe = qoe_model.calculate_qoe(task, delay, energy, aoi)
            qoes.append(qoe)

            # 约束检查
            if delay > task.max_delay:
                constraint_violations += 1
            if hasattr(task, 'max_aoi') and task.max_aoi is not None and aoi > task.max_aoi:
                constraint_violations += 1

        # 公平性
        fairness = fairness_model.calculate_jain_fairness_index(qoes)

        # 约束满足率
        total_constraints = len(system.tasks) * 2  # 每个任务有时延和AoI两个约束
        csr = 1 - constraint_violations / total_constraints

        return {
            'total_energy': total_energy,
            'max_delay': max(delays) if delays else 0,
            'avg_delay': np.mean(delays) if delays else 0,
            'avg_aoi': np.mean(aois) if aois else 0,
            'avg_qoe': np.mean(qoes) if qoes else 0,
            'min_qoe': min(qoes) if qoes else 0,
            'fairness': fairness,
            'csr': csr
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'total_energy': 0,
            'max_delay': 0,
            'avg_delay': 0,
            'avg_aoi': 0,
            'avg_qoe': 0,
            'min_qoe': 0,
            'fairness': 0,
            'csr': 0
        }


def apply_solution_to_system(solution, system):
    """将解应用到系统"""
    try:
        for i, task in enumerate(system.tasks):
            if i >= len(solution):
                break

            loc_idx = int(solution[i][0])
            freq = solution[i][1]

            # 确定执行位置
            num_devices = len(system.devices)
            num_edge = len(system.edge_servers)

            if loc_idx < num_devices:
                task.execution_location = 'device'
                task.execution_node_id = loc_idx
            elif loc_idx < num_devices + num_edge:
                task.execution_location = 'edge'
                task.execution_node_id = loc_idx - num_devices
            else:
                task.execution_location = 'cloud'
                task.execution_node_id = loc_idx - num_devices - num_edge

            task.allocated_resource = freq
    except Exception as e:
        print(f"Error applying solution: {e}")


def save_experiment_results(results, system, output_dir='results/chapter4'):
    """保存实验结果"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存JSON结果
    results_file = os.path.join(output_dir, f'fpa_ts_results_{timestamp}.json')

    # 转换numpy数组为列表
    serializable_results = {}
    for alg_name, alg_results in results.items():
        serializable_results[alg_name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in alg_results.items()
            if k != 'raw_results'
        }

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n结果已保存到: {results_file}")


# ==================== 可视化函数 ====================

def plot_convergence_comparison(results, output_dir='results/chapter4'):
    """
    绘制收敛曲线对比图（与第三章样式一致）
    包含局部放大图显示fitness<1的收敛区域
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    setup_plot_style()  # 应用Times New Roman字体

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, (name, data) in enumerate(results.items()):
        convergence = np.array(data['convergence'])
        x = np.arange(len(convergence))
        color = COLORS[idx % len(COLORS)]

        # 使用标记点增强黑白打印识别度，但不是每个点都标记
        mark_every = max(1, len(x) // 15)  # 每15个点标记一次

        ax.plot(x, convergence,
                label=name,
                color=color,
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                marker=MARKERS[idx % len(MARKERS)],
                markevery=mark_every,
                linewidth=2.5,
                markersize=8)

    ax.set_xlabel('Iterations', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Fitness Value', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Algorithm Convergence Comparison', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)

    # 调整y轴范围，使曲线更清晰
    all_convergence = [data['convergence'] for data in results.values()]
    y_min = min([min(c) for c in all_convergence])
    y_max = max([max(c[:20]) for c in all_convergence])  # 只看前20次迭代的最大值
    ax.set_ylim(y_min * 0.9, y_max * 1.1)

    # 添加局部放大图（放大fitness<1的收敛区域，位置调整到中间偏右下，避免遮挡图例）
    axins = inset_axes(ax, width="22%", height="28%", loc='center',
                      bbox_to_anchor=(0.15, -0.08, 1, 1), bbox_transform=ax.transAxes)

    # 找到所有算法收敛到fitness<1的区域，放大最后的收敛部分
    max_len = max(len(data['convergence']) for data in results.values())

    # 找到fitness值首次降到1以下的位置，从那里开始显示
    start_idx = max_len - 50  # 默认显示最后50次
    for data in results.values():
        conv = data['convergence']
        for i, v in enumerate(conv):
            if v < 1.0:
                start_idx = min(start_idx, max(0, i - 5))  # 从首次<1的位置前5步开始
                break

    # 确保至少显示最后30次迭代
    start_idx = max(start_idx, max_len - 50)

    for idx, (name, data) in enumerate(results.items()):
        convergence = np.array(data['convergence'])
        x = np.arange(len(convergence))
        color = COLORS[idx % len(COLORS)]

        # 在放大图中也使用标记和线型
        axins.plot(x[start_idx:], convergence[start_idx:],
                  color=color,
                  linestyle=LINESTYLES[idx % len(LINESTYLES)],
                  marker=MARKERS[idx % len(MARKERS)],
                  markevery=max(1, len(x[start_idx:]) // 5),
                  linewidth=1.5,
                  markersize=5)

    axins.grid(True, alpha=0.3)
    axins.set_xlim(start_idx, max_len - 1)
    axins.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK-6)

    # 设置放大图的y轴范围为fitness<1的区域，更好地显示RDHO和CWTSSA的差异
    # 只考虑最终收敛值在1以下的算法
    final_values = [data['convergence'][-1] for data in results.values() if data['convergence'][-1] < 0.3]
    if final_values:
        inset_y_min = min(final_values) * 0.95
        inset_y_max = max(final_values) * 1.15
        # 确保y轴范围在合理区间内
        inset_y_max = min(inset_y_max, 0.5)  # 最大不超过0.5，以便更清晰地看到差异
    else:
        inset_y_min = y_min * 0.95
        inset_y_max = 0.3
    axins.set_ylim(inset_y_min, inset_y_max)

    # 添加放大图标题
    axins.set_title('Zoomed', fontsize=FONT_SIZE_TICK-4, pad=2)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'convergence_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"收敛曲线已保存到: {output_dir}/convergence_comparison.png")


def plot_five_objective_radar(results, output_dir='results/chapter4'):
    """绘制五目标雷达图（与第三章样式一致）"""
    from math import pi

    setup_plot_style()  # 应用Times New Roman字体

    categories = ['Energy', 'Delay', 'AoI', 'QoE', 'Fairness']
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 找到最大值用于归一化
    max_energy = max(r['mean_energy'] for r in results.values())
    max_delay = max(r['mean_delay'] for r in results.values())
    max_aoi = max(r['mean_aoi'] for r in results.values())

    for idx, (name, data) in enumerate(results.items()):
        # 归一化指标（能耗、时延、AoI越小越好，QoE和公平性越大越好）
        values = [
            1 - data['mean_energy'] / max_energy if max_energy > 0 else 0,
            1 - data['mean_delay'] / max_delay if max_delay > 0 else 0,
            1 - data['mean_aoi'] / max_aoi if max_aoi > 0 else 0,
            data['mean_qoe'],
            data['mean_fairness']
        ]
        values += values[:1]

        color = COLORS[idx % len(COLORS)]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=name, color=color,
               marker=MARKERS[idx % len(MARKERS)], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK-2)

    plt.title('Five-Objective Performance Comparison', fontsize=FONT_SIZE_TITLE, fontweight='bold', y=1.08)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'five_objective_radar.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'five_objective_radar.pdf'), bbox_inches='tight')
    plt.close()
    print(f"雷达图已保存到: {output_dir}/five_objective_radar.png")


def plot_qoe_fairness_comparison(results, output_dir='results/chapter4'):
    """绘制QoE和公平性对比图（与第三章样式一致）"""
    setup_plot_style()  # 应用Times New Roman字体

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    algorithms = list(results.keys())
    qoe_values = [results[alg]['mean_qoe'] for alg in algorithms]
    fairness_values = [results[alg]['mean_fairness'] for alg in algorithms]

    x = np.arange(len(algorithms))
    width = 0.6

    # QoE柱状图
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(algorithms))]
    bars1 = axes[0].bar(x, qoe_values, width, color=bar_colors,
                        edgecolor='black', linewidth=1.5)
    # 添加纹理
    for idx, bar in enumerate(bars1):
        bar.set_hatch(HATCHES[idx % len(HATCHES)])

    axes[0].set_ylabel('Average QoE', fontsize=FONT_SIZE_LABEL)
    axes[0].set_title('QoE Comparison', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algorithms, fontsize=FONT_SIZE_TICK)
    axes[0].axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0].legend(fontsize=FONT_SIZE_LEGEND-2)
    axes[0].tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, val in zip(bars1, qoe_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE_TEXT-4)

    # Fairness柱状图
    bars2 = axes[1].bar(x, fairness_values, width, color=bar_colors,
                        edgecolor='black', linewidth=1.5)
    for idx, bar in enumerate(bars2):
        bar.set_hatch(HATCHES[idx % len(HATCHES)])

    axes[1].set_ylabel('Jain Fairness Index', fontsize=FONT_SIZE_LABEL)
    axes[1].set_title('Fairness Comparison', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algorithms, fontsize=FONT_SIZE_TICK)
    axes[1].axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Acceptable Threshold')
    axes[1].legend(fontsize=FONT_SIZE_LEGEND-2)
    axes[1].tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, fairness_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE_TEXT-4)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'qoe_fairness_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'qoe_fairness_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"QoE/Fairness对比图已保存到: {output_dir}/qoe_fairness_comparison.png")


def plot_performance_table(results, output_dir='results/chapter4'):
    """生成性能对比表格图（与第三章样式一致）"""
    setup_plot_style()  # 应用Times New Roman字体

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    columns = ['Algorithm', 'Fitness', 'Energy(J)', 'Delay(s)', 'AoI(s)',
               'QoE', 'Fairness', 'CSR(%)', 'Time(s)']

    cell_data = []
    for name, data in results.items():
        row = [
            name,
            f"{data['mean_fitness']:.4f}",
            f"{data['mean_energy']:.2f}",
            f"{data['mean_delay']:.3f}",
            f"{data['mean_aoi']:.3f}",
            f"{data['mean_qoe']:.3f}",
            f"{data['mean_fairness']:.3f}",
            f"{data['mean_csr']*100:.1f}",
            f"{data['mean_runtime']:.2f}"
        ]
        cell_data.append(row)

    # 使用与第三章一致的深蓝色表头
    header_color = COLORS[0]  # 深蓝色
    table = ax.table(cellText=cell_data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=[header_color]*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.3, 2.0)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=14)

    # 高亮RDHO行（最佳算法）
    for i, name in enumerate(results.keys()):
        if name == 'RDHO':
            for j in range(len(columns)):
                table[(i+1, j)].set_facecolor('#E2EFDA')

    plt.title('Performance Comparison of Algorithms', fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'performance_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'performance_table.pdf'), bbox_inches='tight')
    plt.close()
    print(f"性能对比表格已保存到: {output_dir}/performance_table.png")


def generate_all_plots(results, output_dir='results/chapter4'):
    """生成所有实验图表"""
    print("\n生成实验图表...")

    plot_convergence_comparison(results, output_dir)
    print("  ✓ 收敛曲线对比图")

    plot_five_objective_radar(results, output_dir)
    print("  ✓ 五目标雷达图")

    plot_qoe_fairness_comparison(results, output_dir)
    print("  ✓ QoE和公平性对比图")

    plot_performance_table(results, output_dir)
    print("  ✓ 性能对比表格")

    print(f"\n所有图表已保存到: {output_dir}")


# ==================== 主函数 ====================

def main():
    """主函数 - 运行完整的第四章实验"""
    print("="*80)
    print("第四章 RDHO五目标优化实验")
    print("="*80)

    # 实验参数
    config = {
        'num_devices': 20,
        'num_edge_servers': 4,
        'num_cloud_servers': 2,
        'num_tasks': 40,
        'max_iter': 150,
        'population_size': 50,
        'n_runs': 5  # 独立运行次数
    }

    print(f"\n实验配置: {config}")

    # 创建系统
    print("\n1. 创建五目标优化系统...")
    system = create_five_objective_system(
        num_devices=config['num_devices'],
        num_edge_servers=config['num_edge_servers'],
        num_cloud_servers=config['num_cloud_servers'],
        num_tasks=config['num_tasks']
    )
    print(f"   系统创建完成: {len(system.devices)}个设备, "
          f"{len(system.edge_servers)}个边缘服务器, "
          f"{len(system.cloud_servers)}个云服务器, "
          f"{len(system.tasks)}个任务")

    # 运行实验
    print("\n2. 运行算法对比实验...")
    results = run_five_objective_experiment(
        system=system,
        max_iter=config['max_iter'],
        population_size=config['population_size'],
        n_runs=config['n_runs']
    )

    # 生成图表
    print("\n3. 生成实验结果图表...")
    generate_all_plots(results)

    # 打印结果摘要
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)

    print(f"\n{'Algorithm':<12} {'Fitness':<10} {'QoE':<8} {'Fairness':<10} {'CSR(%)':<8}")
    print("-"*50)
    for name, data in results.items():
        print(f"{name:<12} {data['mean_fitness']:<10.4f} {data['mean_qoe']:<8.3f} "
              f"{data['mean_fairness']:<10.3f} {data['mean_csr']*100:<8.1f}")

    # 计算RDHO相对TLBO-HHO的改进
    if 'RDHO' in results and 'TLBO-HHO' in results:
        rdho = results['RDHO']
        tlbo_hho = results['TLBO-HHO']

        print("\n" + "="*80)
        print("RDHO相对TLBO-HHO的改进")
        print("="*80)

        fitness_imp = (tlbo_hho['mean_fitness'] - rdho['mean_fitness']) / tlbo_hho['mean_fitness'] * 100
        qoe_imp = (rdho['mean_qoe'] - tlbo_hho['mean_qoe']) / tlbo_hho['mean_qoe'] * 100
        fairness_imp = (rdho['mean_fairness'] - tlbo_hho['mean_fairness']) / tlbo_hho['mean_fairness'] * 100

        print(f"综合适应度改进: {fitness_imp:.1f}%")
        print(f"QoE改进: {qoe_imp:.1f}%")
        print(f"公平性改进: {fairness_imp:.1f}%")

    # 计算RDHO相对RIME的改进
    if 'RDHO' in results and 'RIME' in results:
        rdho = results['RDHO']
        rime = results['RIME']

        print("\n" + "="*80)
        print("RDHO相对基础RIME的改进")
        print("="*80)

        fitness_imp = (rime['mean_fitness'] - rdho['mean_fitness']) / rime['mean_fitness'] * 100
        qoe_imp = (rdho['mean_qoe'] - rime['mean_qoe']) / rime['mean_qoe'] * 100
        fairness_imp = (rdho['mean_fairness'] - rime['mean_fairness']) / rime['mean_fairness'] * 100

        print(f"综合适应度改进: {fitness_imp:.1f}%")
        print(f"QoE改进: {qoe_imp:.1f}%")
        print(f"公平性改进: {fairness_imp:.1f}%")

    # 计算RDHO相对DBO的改进
    if 'RDHO' in results and 'DBO' in results:
        rdho = results['RDHO']
        dbo = results['DBO']

        print("\n" + "="*80)
        print("RDHO相对基础DBO的改进")
        print("="*80)

        fitness_imp = (dbo['mean_fitness'] - rdho['mean_fitness']) / dbo['mean_fitness'] * 100
        qoe_imp = (rdho['mean_qoe'] - dbo['mean_qoe']) / dbo['mean_qoe'] * 100
        fairness_imp = (rdho['mean_fairness'] - dbo['mean_fairness']) / dbo['mean_fairness'] * 100

        print(f"综合适应度改进: {fitness_imp:.1f}%")
        print(f"QoE改进: {qoe_imp:.1f}%")
        print(f"公平性改进: {fairness_imp:.1f}%")

    print("\n实验完成！")
    return results


if __name__ == '__main__':
    results = main()
