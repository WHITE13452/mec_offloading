#!/usr/bin/env python3
"""
第四章实验结果可视化脚本
生成RDHO算法与对比算法的性能对比图表

使用方法:
    python -m src.experiments.generate_chapter4_figures

或指定数据文件:
    python -m src.experiments.generate_chapter4_figures --data results/chapter4/fpa_ts_results_20260125_150004.json
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import argparse
from pathlib import Path

# ============== 全局样式配置 ==============
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# 算法配置
ALGO_ORDER = ['RDHO', 'RIME', 'DBO', 'TLBO-HHO', 'CWTSSA']

COLORS = {
    'RDHO': '#1f4e79',      # 深蓝色
    'RIME': '#ed7d31',      # 橙色
    'DBO': '#5b9bd5',       # 浅蓝色
    'TLBO-HHO': '#c55a11',  # 棕红色
    'CWTSSA': '#2e75b6'     # 蓝色
}

HATCHES = {
    'RDHO': '',        # 实心
    'RIME': '//',      # 斜线
    'DBO': 'xx',       # 交叉
    'TLBO-HHO': '++',  # 网格
    'CWTSSA': 'oo'     # 圆点
}

MARKERS = {
    'RDHO': 'o',       # 圆点
    'RIME': 's',       # 方形
    'DBO': '^',        # 三角形
    'TLBO-HHO': 'D',   # 菱形
    'CWTSSA': 'v'      # 倒三角
}

LINESTYLES = {
    'RDHO': '-',       # 实线
    'RIME': '--',      # 虚线
    'DBO': '-.',       # 点划线
    'TLBO-HHO': ':',   # 点线
    'CWTSSA': '-'      # 实线
}


def load_data(filepath: str) -> dict:
    """加载实验数据"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_convergence(data: dict, save_path: str):
    """
    图1: 算法收敛曲线对比图（带放大子图）
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # 获取存在的算法
    algos = [algo for algo in ALGO_ORDER if algo in data]

    # 绘制主图
    for algo in algos:
        conv = data[algo]['convergence']
        iterations = range(len(conv))
        ax.plot(iterations, conv,
               label=algo,
               color=COLORS[algo],
               linestyle=LINESTYLES[algo],
               marker=MARKERS[algo],
               markevery=10,
               markersize=8,
               linewidth=2)

    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Fitness Value', fontsize=14)
    ax.set_title('Algorithm Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加放大子图 (显示最终收敛区域)
    axins = ax.inset_axes([0.50, 0.35, 0.45, 0.35])

    for algo in algos:
        conv = data[algo]['convergence']
        # 只显示最后50次迭代
        start_idx = max(0, len(conv) - 50)
        iterations = range(start_idx, len(conv))
        axins.plot(iterations, conv[start_idx:],
                  color=COLORS[algo],
                  linestyle=LINESTYLES[algo],
                  marker=MARKERS[algo],
                  markevery=5,
                  markersize=6,
                  linewidth=1.5)

    # 设置放大子图的范围（只显示fitness < 1的算法的最终收敛值）
    final_values = [data[algo]['convergence'][-1] for algo in algos
                   if data[algo]['convergence'][-1] < 1.0]
    if final_values:
        y_min = min(final_values) * 0.9
        y_max = max(final_values) * 1.2
        y_max = min(y_max, 0.6)  # 最大不超过0.6
    else:
        y_min, y_max = 0.1, 0.6

    axins.set_ylim(y_min, y_max)
    axins.set_xlim(100, 160)
    axins.set_title('Zoomed (Iterations 100-150)', fontsize=10)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=9)

    # 添加放大框指示
    ax.indicate_inset_zoom(axins, edgecolor='black', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_bar_comparison(data: dict, metric: str, ylabel: str, title: str,
                        save_path: str, lower_is_better: bool = True,
                        threshold: float = None, threshold_label: str = None,
                        value_format: str = 'auto'):
    """
    通用柱状图绘制函数

    Args:
        data: 实验数据
        metric: 指标名称
        ylabel: Y轴标签
        title: 图标题
        save_path: 保存路径
        lower_is_better: 是否越小越好
        threshold: 阈值线位置
        threshold_label: 阈值标签
        value_format: 数值格式 ('auto', 'percent', 'int', 'float2', 'float3', 'float4')
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    algos = [algo for algo in ALGO_ORDER if algo in data]
    values = [data[algo][metric] for algo in algos]
    x = np.arange(len(algos))
    width = 0.6

    bars = []
    for i, (algo, val) in enumerate(zip(algos, values)):
        bar = ax.bar(x[i], val, width,
                    color=COLORS[algo],
                    hatch=HATCHES[algo],
                    edgecolor='black',
                    linewidth=1.2)
        bars.append(bar)

        # 格式化数值标注
        if value_format == 'auto':
            if val >= 1000:
                label = f'{val:.0f}'
            elif val >= 10:
                label = f'{val:.2f}'
            elif val >= 1:
                label = f'{val:.3f}'
            else:
                label = f'{val:.4f}'
        elif value_format == 'percent':
            label = f'{val:.1f}%'
        elif value_format == 'int':
            label = f'{val:.0f}'
        elif value_format == 'float2':
            label = f'{val:.2f}'
        elif value_format == 'float3':
            label = f'{val:.3f}'
        elif value_format == 'float4':
            label = f'{val:.4f}'
        else:
            label = f'{val}'

        ax.annotate(label,
                   xy=(x[i], val),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    # 添加阈值线
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
        if threshold_label:
            ax.text(len(algos)-0.5, threshold*1.02, threshold_label,
                   color='red', fontsize=11, ha='right')

    # 标记最优值
    if lower_is_better:
        best_idx = values.index(min(values))
    else:
        best_idx = values.index(max(values))

    # 高亮最优柱子
    bars[best_idx][0].set_edgecolor('gold')
    bars[best_idx][0].set_linewidth(3)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 自动调整y轴范围
    y_max = max(values) * 1.15
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_energy_comparison(data: dict, save_path: str):
    """图2: 能耗性能对比柱状图"""
    plot_bar_comparison(data, 'mean_energy',
                       'Total Energy Consumption (J)',
                       'Energy Consumption Comparison',
                       save_path, lower_is_better=True)


def plot_delay_comparison(data: dict, save_path: str):
    """图3: 时延性能对比柱状图"""
    plot_bar_comparison(data, 'mean_delay',
                       'Total Delay (s)',
                       'Response Time Comparison',
                       save_path, lower_is_better=True,
                       value_format='float3')


def plot_aoi_comparison(data: dict, save_path: str):
    """图4: 信息年龄(AoI)对比柱状图"""
    plot_bar_comparison(data, 'mean_aoi',
                       'Average Age of Information (s)',
                       'Age of Information Comparison',
                       save_path, lower_is_better=True,
                       value_format='float3')


def plot_qoe_fairness(data: dict, save_path: str):
    """图5: QoE和Fairness双子图对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    algos = [algo for algo in ALGO_ORDER if algo in data]
    x = np.arange(len(algos))
    width = 0.6

    # ===== QoE子图 =====
    qoe_values = [data[algo]['mean_qoe'] for algo in algos]
    for i, (algo, val) in enumerate(zip(algos, qoe_values)):
        bar = ax1.bar(x[i], val, width,
               color=COLORS[algo],
               hatch=HATCHES[algo],
               edgecolor='black',
               linewidth=1.2)
        ax1.annotate(f'{val:.3f}',
                    xy=(x[i], val),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold')
    ax1.set_ylabel('Average QoE', fontsize=14)
    ax1.set_title('QoE Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, fontsize=11, rotation=15)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== Fairness子图 =====
    fairness_values = [data[algo]['mean_fairness'] for algo in algos]
    for i, (algo, val) in enumerate(zip(algos, fairness_values)):
        bar = ax2.bar(x[i], val, width,
               color=COLORS[algo],
               hatch=HATCHES[algo],
               edgecolor='black',
               linewidth=1.2)
        ax2.annotate(f'{val:.3f}',
                    xy=(x[i], val),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Acceptable Threshold')
    ax2.set_ylabel('Jain Fairness Index', fontsize=14)
    ax2.set_title('Fairness Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, fontsize=11, rotation=15)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_radar(data: dict, save_path: str):
    """图6: 五目标雷达图"""
    # 定义五个维度
    categories = ['Delay', 'Energy', 'AoI', 'QoE', 'Fairness']
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 提取各指标的范围用于归一化
    algos = [algo for algo in ALGO_ORDER if algo in data]

    metrics = {
        'Delay': ('mean_delay', True),      # (key, lower_is_better)
        'Energy': ('mean_energy', True),
        'AoI': ('mean_aoi', True),
        'QoE': ('mean_qoe', False),
        'Fairness': ('mean_fairness', False)
    }

    # 归一化数据
    normalized_data = {}
    for cat in categories:
        key, lower_is_better = metrics[cat]
        values = [data[algo][key] for algo in algos]
        min_val, max_val = min(values), max(values)

        normalized_data[cat] = {}
        for algo in algos:
            val = data[algo][key]
            if max_val == min_val:
                norm_val = 1.0
            elif lower_is_better:
                # 越小越好 -> 归一化后越大越好（外圈更好）
                norm_val = 1 - (val - min_val) / (max_val - min_val)
            else:
                # 越大越好 -> 直接归一化
                norm_val = (val - min_val) / (max_val - min_val)
            normalized_data[cat][algo] = norm_val

    # 绘制每个算法
    for algo in algos:
        values = [normalized_data[cat][algo] for cat in categories]
        values += values[:1]  # 闭合

        ax.plot(angles, values,
               color=COLORS[algo],
               linestyle=LINESTYLES[algo],
               linewidth=2,
               marker=MARKERS[algo],
               markersize=6,
               label=algo)
        ax.fill(angles, values, color=COLORS[algo], alpha=0.1)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.set_title('Five-Objective Performance Comparison', fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_csr_comparison(data: dict, save_path: str):
    """图7: 约束满足率(CSR)对比柱状图"""
    fig, ax = plt.subplots(figsize=(9, 6))

    algos = [algo for algo in ALGO_ORDER if algo in data]
    values = [data[algo]['mean_csr'] * 100 for algo in algos]  # 转换为百分比
    x = np.arange(len(algos))
    width = 0.6

    bars = []
    for i, (algo, val) in enumerate(zip(algos, values)):
        bar = ax.bar(x[i], val, width,
              color=COLORS[algo],
              hatch=HATCHES[algo],
              edgecolor='black',
              linewidth=1.2)
        bars.append(bar)
        ax.annotate(f'{val:.1f}%',
                   xy=(x[i], val),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    # 标记最优值
    best_idx = values.index(max(values))
    bars[best_idx][0].set_edgecolor('gold')
    bars[best_idx][0].set_linewidth(3)

    ax.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(len(algos)-0.5, 96, 'Target (95%)', color='red', fontsize=11, ha='right')

    ax.set_ylabel('Constraint Satisfaction Rate (%)', fontsize=14)
    ax.set_title('Constraint Satisfaction Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_performance_table(data: dict, save_path: str):
    """图8: 综合性能对比表格"""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    algos = [algo for algo in ALGO_ORDER if algo in data]

    # 表格数据
    columns = ['Algorithm', 'Fitness', 'Energy (J)', 'Delay (s)', 'AoI (s)',
               'QoE', 'Fairness', 'CSR (%)', 'Runtime (s)']

    cell_data = []
    for algo in algos:
        d = data[algo]
        row = [
            algo,
            f"{d['mean_fitness']:.4f}",
            f"{d['mean_energy']:.1f}",
            f"{d['mean_delay']:.3f}",
            f"{d['mean_aoi']:.3f}",
            f"{d['mean_qoe']:.3f}",
            f"{d['mean_fairness']:.4f}",
            f"{d['mean_csr']*100:.1f}",
            f"{d['mean_runtime']:.2f}"
        ]
        cell_data.append(row)

    # 创建表格
    table = ax.table(cellText=cell_data,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#4472C4']*len(columns))

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 高亮RDHO行（第一行数据）
    for j in range(len(columns)):
        table[(1, j)].set_facecolor('#D6E9F8')

    ax.set_title('Table: Algorithm Performance Comparison', fontsize=14, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_fitness_comparison(data: dict, save_path: str):
    """额外: 综合适应度对比柱状图"""
    plot_bar_comparison(data, 'mean_fitness',
                       'Fitness Value',
                       'Fitness Comparison',
                       save_path, lower_is_better=True,
                       value_format='float4')


def print_improvement_analysis(data: dict):
    """打印RDHO相对于其他算法的改进分析"""
    if 'RDHO' not in data:
        return

    rdho = data['RDHO']

    print("\n" + "="*60)
    print("RDHO Performance Improvement Analysis")
    print("="*60)

    for algo in ['RIME', 'DBO', 'TLBO-HHO', 'CWTSSA']:
        if algo not in data:
            continue

        other = data[algo]
        print(f"\n--- RDHO vs {algo} ---")

        # Fitness (越小越好)
        fitness_imp = (other['mean_fitness'] - rdho['mean_fitness']) / other['mean_fitness'] * 100
        print(f"  Fitness improvement: {fitness_imp:+.1f}%")

        # Energy (越小越好)
        energy_imp = (other['mean_energy'] - rdho['mean_energy']) / other['mean_energy'] * 100
        print(f"  Energy improvement: {energy_imp:+.1f}%")

        # Delay (越小越好)
        if other['mean_delay'] > 0:
            delay_imp = (other['mean_delay'] - rdho['mean_delay']) / other['mean_delay'] * 100
            print(f"  Delay change: {delay_imp:+.1f}%")

        # AoI (越小越好)
        aoi_imp = (other['mean_aoi'] - rdho['mean_aoi']) / other['mean_aoi'] * 100
        print(f"  AoI improvement: {aoi_imp:+.1f}%")

        # QoE (越大越好)
        qoe_imp = (rdho['mean_qoe'] - other['mean_qoe']) / other['mean_qoe'] * 100
        print(f"  QoE improvement: {qoe_imp:+.1f}%")

        # Fairness (越大越好)
        fairness_imp = (rdho['mean_fairness'] - other['mean_fairness']) / other['mean_fairness'] * 100
        print(f"  Fairness improvement: {fairness_imp:+.1f}%")


def find_latest_results_file(results_dir: str = 'results/chapter4') -> str:
    """找到最新的实验结果文件"""
    results_path = Path(results_dir)
    json_files = list(results_path.glob('fpa_ts_results_*.json'))

    if not json_files:
        return None

    # 按修改时间排序，返回最新的
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate Chapter 4 experiment figures')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to the JSON results file')
    parser.add_argument('--output', type=str, default='results/chapter4/scriptsbuilt',
                       help='Output directory for figures')
    args = parser.parse_args()

    # 确定数据文件路径
    if args.data:
        data_path = args.data
    else:
        data_path = find_latest_results_file()
        if not data_path:
            print("Error: No results file found in results/chapter4/")
            print("Please run the experiment first or specify a data file with --data")
            return

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found!")
        return

    print(f"Loading data from: {data_path}")

    # 加载数据
    data = load_data(data_path)
    print(f"Loaded data for algorithms: {list(data.keys())}")

    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 生成所有图表
    print("\nGenerating figures...")

    plot_convergence(data, f'{output_dir}/fig4_convergence.png')
    plot_fitness_comparison(data, f'{output_dir}/fig4_fitness.png')
    plot_energy_comparison(data, f'{output_dir}/fig4_energy.png')
    plot_delay_comparison(data, f'{output_dir}/fig4_delay.png')
    plot_aoi_comparison(data, f'{output_dir}/fig4_aoi.png')
    plot_qoe_fairness(data, f'{output_dir}/fig4_qoe_fairness.png')
    plot_radar(data, f'{output_dir}/fig4_radar.png')
    plot_csr_comparison(data, f'{output_dir}/fig4_csr.png')
    plot_performance_table(data, f'{output_dir}/fig4_table.png')

    # 打印改进分析
    print_improvement_analysis(data)

    print(f"\n✅ All figures saved to '{output_dir}/' directory")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('fig4_') and (f.endswith('.png') or f.endswith('.pdf')):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
