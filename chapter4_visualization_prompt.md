# 第四章实验结果可视化 - Claude Code 完整实现指南

## 任务目标

基于实验结果JSON文件，为第四章论文生成完整的实验分析图表。参考第三章的图表风格（能耗箱线图、时延箱线图、AoI箱线图等），为五目标优化问题的每个指标生成独立的对比图表。

## 数据文件

实验结果JSON文件路径: `fpa_ts_results_20260125_150004.json`

数据结构说明：
- 算法: RDHO (提出算法), RIME, DBO, TLBO-HHO, CWTSSA
- 指标: mean_fitness, std_fitness, mean_energy, mean_delay, mean_aoi, mean_qoe, mean_fairness, mean_csr, mean_runtime, convergence

## 需要生成的图表清单

### 图1: 算法收敛曲线对比图 (Fig. 4-X Algorithm Convergence Comparison)

**要求:**
- 主图显示所有算法的收敛曲线
- 右下角添加放大子图(inset)，显示最终收敛区域(iteration 100-150, fitness 0.1-0.2)
- X轴: Iterations, Y轴: Fitness Value
- 使用不同线型和标记区分算法:
  - RDHO: 实线 + 圆点标记, 深蓝色 '#1f4e79'
  - RIME: 虚线 + 方形标记, 橙色 '#ed7d31'
  - DBO: 点划线 + 三角形标记, 浅蓝色 '#5b9bd5'
  - TLBO-HHO: 点线 + 菱形标记, 棕红色 '#c55a11'
  - CWTSSA: 实线 + 倒三角标记, 蓝色 '#2e75b6'
- 标记间隔: 每10个点显示一个标记
- 图例放在右上角
- 图片尺寸: 10x7 inches, DPI: 300

### 图2: 能耗性能对比柱状图 (Fig. 4-X Energy Consumption Comparison)

**要求:**
- 柱状图对比各算法的mean_energy
- 使用不同填充图案(hatching)区分算法:
  - RDHO: 实心填充, 深蓝色
  - RIME: 斜线填充 '//', 橙色
  - DBO: 交叉填充 'xx', 浅蓝色
  - TLBO-HHO: 网格填充 '++', 棕红色
  - CWTSSA: 点填充 '..', 蓝色
- 在柱子上方标注具体数值
- Y轴: Total Energy Consumption (J)
- 添加水平虚线标注RDHO的能耗值作为基准线
- 图片尺寸: 8x6 inches

### 图3: 时延性能对比柱状图 (Fig. 4-X Response Time Comparison)

**要求:**
- 柱状图对比各算法的mean_delay
- 样式与能耗图保持一致
- Y轴: Total Delay (s)
- 在柱子上方标注具体数值
- 图片尺寸: 8x6 inches

### 图4: 信息年龄(AoI)对比柱状图 (Fig. 4-X Age of Information Comparison)

**要求:**
- 柱状图对比各算法的mean_aoi
- 样式与能耗图保持一致
- Y轴: Average AoI (s)
- 在柱子上方标注具体数值
- 添加说明: AoI越低表示信息新鲜度越高
- 图片尺寸: 8x6 inches

### 图5: QoE和Fairness双子图对比 (Fig. 4-X QoE and Fairness Comparison)

**要求:**
- 1行2列的子图布局
- 左图: QoE对比柱状图
  - Y轴: Average QoE, 范围[0, 1]
  - 添加红色虚线在y=0.8处标注"Threshold"
- 右图: Fairness对比柱状图  
  - Y轴: Jain Fairness Index, 范围[0, 1]
  - 添加红色虚线在y=0.8处标注"Acceptable Threshold"
- 柱子上方标注具体数值(保留3位小数)
- 使用与其他图一致的颜色和填充图案
- 图片尺寸: 12x5 inches

### 图6: 五目标雷达图 (Fig. 4-X Five-Objective Performance Comparison)

**要求:**
- 5个维度: Delay, Energy, AoI, QoE, Fairness
- 所有指标归一化到[0,1]区间:
  - Delay, Energy, AoI: 使用 1 - (value - min) / (max - min)，值越大表示越好
  - QoE, Fairness: 直接使用原始值(已经是[0,1]区间)
- 每个算法一条折线，使用半透明填充
- 图例放在右上角外侧
- 图片尺寸: 8x8 inches

### 图7: 约束满足率(CSR)对比柱状图 (Fig. 4-X Constraint Satisfaction Rate Comparison)

**要求:**
- 柱状图对比各算法的mean_csr
- Y轴: Constraint Satisfaction Rate (%), 范围[0, 100]
- 将CSR值转换为百分比显示
- 添加红色虚线在y=95%处标注"Target"
- 在柱子上方标注具体数值
- 图片尺寸: 8x6 inches

### 图8: 综合性能对比表格图 (Table 4-X Algorithm Performance Comparison)

**要求:**
- 使用matplotlib生成表格图像
- 列: 算法, Fitness, Energy(J), Delay(s), AoI(s), QoE, Fairness, CSR(%), Runtime(s)
- 对于每个指标，最优值加粗显示
- Fitness, Energy, Delay, AoI: 值越小越好
- QoE, Fairness, CSR: 值越大越好
- 添加RDHO相对于其他算法的提升百分比行
- 图片尺寸: 14x4 inches

## 代码实现要求

### 1. 全局样式设置

```python
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json

# 设置全局字体和样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# 算法颜色和样式定义
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
    'CWTSSA': '..'     # 点
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

ALGO_ORDER = ['RDHO', 'RIME', 'DBO', 'TLBO-HHO', 'CWTSSA']
```

### 2. 数据加载函数

```python
def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
```

### 3. 各图表绘制函数模板

为每个图表创建独立的函数，例如:
- `plot_convergence(data, save_path)`
- `plot_energy_comparison(data, save_path)`
- `plot_delay_comparison(data, save_path)`
- `plot_aoi_comparison(data, save_path)`
- `plot_qoe_fairness(data, save_path)`
- `plot_radar(data, save_path)`
- `plot_csr_comparison(data, save_path)`
- `plot_performance_table(data, save_path)`

### 4. 主函数

```python
def main():
    data = load_data('fpa_ts_results_20260125_150004.json')
    
    # 创建输出目录
    import os
    os.makedirs('figures', exist_ok=True)
    
    # 生成所有图表
    plot_convergence(data, 'figures/fig4_convergence.png')
    plot_energy_comparison(data, 'figures/fig4_energy.png')
    plot_delay_comparison(data, 'figures/fig4_delay.png')
    plot_aoi_comparison(data, 'figures/fig4_aoi.png')
    plot_qoe_fairness(data, 'figures/fig4_qoe_fairness.png')
    plot_radar(data, 'figures/fig4_radar.png')
    plot_csr_comparison(data, 'figures/fig4_csr.png')
    plot_performance_table(data, 'figures/fig4_table.png')
    
    print("All figures saved to 'figures/' directory")

if __name__ == '__main__':
    main()
```

## 输出要求

1. 所有图表保存为PNG格式，DPI=300
2. 文件命名规范: `fig4_[指标名].png`
3. 图表标题使用英文，适合直接放入论文
4. 确保所有图表风格一致，颜色、字体、线宽统一
5. 图例清晰，不遮挡数据

## 数据参考 (来自JSON - 实际值)

| 算法 | Fitness | Energy | Delay | AoI | QoE | Fairness | CSR |
|------|---------|--------|-------|-----|-----|----------|-----|
| RDHO | 0.1146 | 545.5 | 2.395 | 0.555 | 0.568 | 0.9944 | 100% |
| RIME | 32.62 | 3262.2 | 2.018 | 2.629 | 0.429 | 0.8687 | 76% |
| DBO | 9.75 | 1307.6 | 3.037 | 1.654 | 0.496 | 0.9362 | 89% |
| TLBO-HHO | 0.511 | 792.8 | 1.766 | 1.147 | 0.568 | 0.9924 | 99.75% |
| CWTSSA | 0.135 | 3212.3 | 2.617 | 0.490 | 0.570 | 0.9933 | 100% |

**关键发现:**
- RDHO的Fitness值(0.1146)在所有算法中最优
- RDHO的能耗(545.5J)显著低于其他算法
- RDHO实现了100%的约束满足率
- RDHO在QoE(0.568)和Fairness(0.9944)上表现优异

## 注意事项

1. RDHO是提出的算法，在图表中应该放在第一位
2. 对于"越小越好"的指标(Fitness, Energy, Delay, AoI)，RDHO表现最优时应突出显示
3. 对于"越大越好"的指标(QoE, Fairness, CSR)，同样需要突出RDHO的优势
4. 雷达图需要统一方向：所有维度都是"越大越好"（外圈更好）
5. 确保图表可以在黑白打印时也能区分（使用不同填充图案）

---

## 完整代码实现

请将以下代码保存为 `generate_chapter4_figures.py` 并运行：

```python
#!/usr/bin/env python3
"""
第四章实验结果可视化脚本
生成RDHO算法与对比算法的性能对比图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from matplotlib.patches import Patch

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


def load_data(filepath):
    """加载实验数据"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_convergence(data, save_path):
    """图1: 算法收敛曲线对比图（带放大子图）"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制主图
    for algo in ALGO_ORDER:
        if algo in data:
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
    
    # 添加放大子图
    axins = ax.inset_axes([0.55, 0.35, 0.4, 0.35])
    for algo in ALGO_ORDER:
        if algo in data:
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
    
    # 设置放大子图的范围（只显示fitness < 1的算法）
    axins.set_ylim(0.1, 0.6)
    axins.set_xlim(100, 160)
    axins.set_title('Zoomed', fontsize=10)
    axins.grid(True, alpha=0.3)
    
    # 添加放大框指示
    ax.indicate_inset_zoom(axins, edgecolor='black', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_bar_comparison(data, metric, ylabel, title, save_path, 
                        lower_is_better=True, threshold=None, threshold_label=None):
    """通用柱状图绘制函数"""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    values = [data[algo][metric] for algo in ALGO_ORDER if algo in data]
    algos = [algo for algo in ALGO_ORDER if algo in data]
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
        
        # 在柱子上方标注数值
        if val >= 1000:
            label = f'{val:.0f}'
        elif val >= 10:
            label = f'{val:.2f}'
        elif val >= 1:
            label = f'{val:.3f}'
        else:
            label = f'{val:.4f}'
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
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_energy_comparison(data, save_path):
    """图2: 能耗性能对比柱状图"""
    plot_bar_comparison(data, 'mean_energy', 
                       'Total Energy Consumption (J)',
                       'Energy Consumption Comparison',
                       save_path, lower_is_better=True)


def plot_delay_comparison(data, save_path):
    """图3: 时延性能对比柱状图"""
    plot_bar_comparison(data, 'mean_delay',
                       'Total Delay (s)',
                       'Response Time Comparison',
                       save_path, lower_is_better=True)


def plot_aoi_comparison(data, save_path):
    """图4: 信息年龄(AoI)对比柱状图"""
    plot_bar_comparison(data, 'mean_aoi',
                       'Average Age of Information (s)',
                       'Age of Information Comparison',
                       save_path, lower_is_better=True)


def plot_qoe_fairness(data, save_path):
    """图5: QoE和Fairness双子图对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    algos = [algo for algo in ALGO_ORDER if algo in data]
    x = np.arange(len(algos))
    width = 0.6
    
    # QoE子图
    qoe_values = [data[algo]['mean_qoe'] for algo in algos]
    for i, (algo, val) in enumerate(zip(algos, qoe_values)):
        ax1.bar(x[i], val, width,
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
    ax1.set_xticklabels(algos, fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Fairness子图
    fairness_values = [data[algo]['mean_fairness'] for algo in algos]
    for i, (algo, val) in enumerate(zip(algos, fairness_values)):
        ax2.bar(x[i], val, width,
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
    ax2.set_xticklabels(algos, fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_radar(data, save_path):
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
                # 越小越好 -> 归一化后越大越好
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
    plt.close()
    print(f"Saved: {save_path}")


def plot_csr_comparison(data, save_path):
    """图7: 约束满足率(CSR)对比柱状图"""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    algos = [algo for algo in ALGO_ORDER if algo in data]
    values = [data[algo]['mean_csr'] * 100 for algo in algos]  # 转换为百分比
    x = np.arange(len(algos))
    width = 0.6
    
    for i, (algo, val) in enumerate(zip(algos, values)):
        ax.bar(x[i], val, width,
              color=COLORS[algo],
              hatch=HATCHES[algo],
              edgecolor='black',
              linewidth=1.2)
        ax.annotate(f'{val:.1f}%',
                   xy=(x[i], val),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
    
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
    plt.close()
    print(f"Saved: {save_path}")


def plot_performance_table(data, save_path):
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
    
    # 高亮RDHO行
    for j in range(len(columns)):
        table[(1, j)].set_facecolor('#D6E9F8')
    
    ax.set_title('Table: Algorithm Performance Comparison', fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """主函数"""
    # 数据文件路径 - 请根据实际情况修改
    data_path = 'fpa_ts_results_20260125_150004.json'
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found!")
        print("Please ensure the JSON file is in the current directory.")
        return
    
    # 加载数据
    data = load_data(data_path)
    print(f"Loaded data for algorithms: {list(data.keys())}")
    
    # 创建输出目录
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成所有图表
    print("\nGenerating figures...")
    plot_convergence(data, f'{output_dir}/fig4_convergence.png')
    plot_energy_comparison(data, f'{output_dir}/fig4_energy.png')
    plot_delay_comparison(data, f'{output_dir}/fig4_delay.png')
    plot_aoi_comparison(data, f'{output_dir}/fig4_aoi.png')
    plot_qoe_fairness(data, f'{output_dir}/fig4_qoe_fairness.png')
    plot_radar(data, f'{output_dir}/fig4_radar.png')
    plot_csr_comparison(data, f'{output_dir}/fig4_csr.png')
    plot_performance_table(data, f'{output_dir}/fig4_table.png')
    
    print(f"\n✅ All figures saved to '{output_dir}/' directory")
    print("\nGenerated files:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
```

## 运行说明

1. 将上述代码保存为 `generate_chapter4_figures.py`
2. 确保 `fpa_ts_results_20260125_150004.json` 在同一目录下
3. 运行: `python generate_chapter4_figures.py`
4. 图表将保存在 `figures/` 目录下

## 生成的图表文件列表

- `fig4_convergence.png` - 收敛曲线对比图
- `fig4_energy.png` - 能耗对比柱状图
- `fig4_delay.png` - 时延对比柱状图
- `fig4_aoi.png` - AoI对比柱状图
- `fig4_qoe_fairness.png` - QoE和Fairness双子图
- `fig4_radar.png` - 五目标雷达图
- `fig4_csr.png` - 约束满足率对比图
- `fig4_table.png` - 综合性能对比表格
