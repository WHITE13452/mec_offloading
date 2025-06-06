# src/experiments/complete_aoi_experiment.py
import os
import numpy as np
import matplotlib.pyplot as plt
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
    """运行考虑AoI的完整实验"""
    
    # 1. 创建或加载系统
    if system_config is None:
        print("="*100)
        print("1. 创建考虑AoI的边缘计算系统")
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
            if task.update_interval is None:
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
    print("2. 系统AoI特性分析")
    print("="*100)
    analyze_aoi_characteristics(system)
    
    # 3. 运行不同权重组合的实验
    print("\n" + "="*100)
    print("3. 运行不同权重组合的算法对比实验")
    print("="*100)
    
    # 定义不同的权重组合
    weight_combinations = [
        {'name': 'Energy-Delay', 'w_energy': 0.5, 'w_delay': 0.5, 'w_aoi': 0.0},
        {'name': 'Energy-AoI', 'w_energy': 0.5, 'w_delay': 0.0, 'w_aoi': 0.5},
        {'name': 'Delay-AoI', 'w_energy': 0.0, 'w_delay': 0.5, 'w_aoi': 0.5},
        {'name': 'Balanced', 'w_energy': 0.33, 'w_delay': 0.33, 'w_aoi': 0.34},
        {'name': 'AoI-focused', 'w_energy': 0.2, 'w_delay': 0.2, 'w_aoi': 0.6}
    ]
    
    all_results = {}
    for weights in weight_combinations:
        print(f"\n运行权重组合: {weights['name']}")
        results = run_algorithm_comparison_with_weights(
            system, 
            weights['w_energy'], 
            weights['w_delay'], 
            weights['w_aoi'],
            max_iter=150, 
            population_size=50, 
            n_runs=5
        )
        all_results[weights['name']] = results
    
    # 4. 分析和可视化结果
    print("\n" + "="*100)
    print("4. 分析实验结果")
    print("="*100)
    
    analyze_aoi_results(all_results, system)
    
    # 5. 生成实验图表
    print("\n" + "="*100)
    print("5. 生成实验结果图表")
    print("="*100)
    
    plot_aoi_convergence_curves(all_results)
    plot_aoi_performance_comparison(all_results, system)
    plot_weight_combination_effects(all_results, system)
    plot_aoi_distribution(all_results, system)
    
    print("\nAoI实验完成！所有结果已保存到 results/aoi/ 目录")
    
    return system, all_results


def run_algorithm_comparison_with_weights(system, w_energy, w_delay, w_aoi, 
                                         max_iter=150, population_size=50, n_runs=5):
    """运行指定权重的算法对比实验"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 初始化算法，使用指定的权重
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
            hho_prob=0.3, verbose=False
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
        print(f"  运行 {name} 算法...")
        
        best_fitness_history = np.zeros((n_runs, max_iter + 1))
        best_solutions = []
        best_fitness_values = []
        detailed_metrics = []
        
        for run in range(n_runs):
            print(f"    运行 {run + 1}/{n_runs}", end=' ... ')
            
            # 重置系统状态
            for task in system.tasks:
                task.execution_location = None
                task.execution_node_id = None
                task.allocated_resource = None
                task.delay = None
                task.energy = None
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
            
            print(f"完成，适应度: {best_fitness:.6f}")
        
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
        
        print(f"    {name} 完成：平均适应度 {results[name]['mean_fitness']:.6f} ± {results[name]['std_fitness']:.6f}")
    
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
        if task.update_interval is not None:
            try:
                task.aoi = aoi_model.calculate_average_aoi(task)
                metrics['total_aoi'] += task.aoi
                valid_aoi_tasks += 1
                
                # 检查AoI违规
                if task.max_aoi is not None and task.aoi > task.max_aoi:
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
    print("\n任务AoI特性分析:")
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
        if task.update_interval is not None:
            task_types[task_type]['update_intervals'].append(task.update_interval)
        if task.max_aoi is not None:
            task_types[task_type]['max_aois'].append(task.max_aoi)
    
    print(f"{'任务类型':<20} {'数量':<8} {'平均更新间隔(s)':<15} {'平均最大AoI(s)':<15}")
    print("-" * 60)
    
    for task_type, data in task_types.items():
        avg_interval = np.mean(data['update_intervals']) if data['update_intervals'] else 0
        avg_max_aoi = np.mean(data['max_aois']) if data['max_aois'] else 0
        print(f"{task_type:<20} {data['count']:<8} {avg_interval:<15.3f} {avg_max_aoi:<15.3f}")


def analyze_aoi_results(all_results, system):
    """分析AoI实验结果"""
    print("\n不同权重组合下的算法性能对比:")
    print("=" * 100)
    
    for weight_name, results in all_results.items():
        print(f"\n权重组合: {weight_name}")
        print("-" * 80)
        print(f"{'算法':<10} {'平均适应度':<12} {'平均能耗(J)':<12} {'平均延迟(s)':<12} {'平均AoI(s)':<12} {'AoI违规率(%)':<15}")
        print("-" * 80)
        
        for alg_name, data in results.items():
            # 计算平均指标
            metrics_list = data['detailed_metrics']
            avg_energy = np.mean([m['avg_energy'] for m in metrics_list])
            avg_delay = np.mean([m['avg_delay'] for m in metrics_list])
            avg_aoi = np.mean([m['avg_aoi'] for m in metrics_list])
            
            # 计算AoI违规率
            total_tasks = len(system.tasks)
            aoi_tasks = sum(1 for t in system.tasks if t.update_interval is not None)
            avg_aoi_violations = np.mean([m['aoi_violations'] for m in metrics_list])
            aoi_violation_rate = (avg_aoi_violations / aoi_tasks * 100) if aoi_tasks > 0 else 0
            
            print(f"{alg_name:<10} {data['mean_fitness']:<12.6f} {avg_energy:<12.6f} "
                  f"{avg_delay:<12.6f} {avg_aoi:<12.6f} {aoi_violation_rate:<15.2f}")


def plot_aoi_convergence_curves(all_results, save_dir='results/aoi/'):
    """绘制不同权重组合下的收敛曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个权重组合创建一个子图
    n_weights = len(all_results)
    fig, axes = plt.subplots(1, n_weights, figsize=(5*n_weights, 5))
    
    if n_weights == 1:
        axes = [axes]
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for idx, (weight_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        for i, (alg_name, data) in enumerate(results.items()):
            avg_history = data['avg_fitness_history']
            std_history = data['std_fitness_history']
            
            x = np.arange(len(avg_history))
            color = colors[i % len(colors)]
            
            ax.plot(x, avg_history, label=alg_name, color=color, linewidth=2)
            ax.fill_between(x, avg_history - std_history, avg_history + std_history, 
                           alpha=0.2, color=color)
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('适应度值')
        ax.set_title(f'{weight_name}权重组合')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'aoi_convergence_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AoI收敛曲线已保存到: {os.path.join(save_dir, 'aoi_convergence_curves.png')}")


def plot_aoi_performance_comparison(all_results, system, save_dir='results/aoi/'):
    """绘制AoI性能对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    weight_names = list(all_results.keys())
    algorithms = list(next(iter(all_results.values())).keys())
    
    # 创建性能指标矩阵
    metrics_names = ['avg_energy', 'avg_delay', 'avg_aoi', 'aoi_violation_rate']
    metrics_labels = ['平均能耗 (J)', '平均延迟 (s)', '平均AoI (s)', 'AoI违规率 (%)']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for metric_idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
        ax = axes[metric_idx]
        
        # 准备数据
        data_matrix = []
        for weight_name in weight_names:
            row = []
            for alg_name in algorithms:
                metrics_list = all_results[weight_name][alg_name]['detailed_metrics']
                
                if metric_name == 'aoi_violation_rate':
                    aoi_tasks = sum(1 for t in system.tasks if t.update_interval is not None)
                    avg_violations = np.mean([m['aoi_violations'] for m in metrics_list])
                    value = (avg_violations / aoi_tasks * 100) if aoi_tasks > 0 else 0
                else:
                    value = np.mean([m[metric_name] for m in metrics_list])
                
                row.append(value)
            data_matrix.append(row)
        
        # 绘制热力图
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置标签
        ax.set_xticks(np.arange(len(algorithms)))
        ax.set_yticks(np.arange(len(weight_names)))
        ax.set_xticklabels(algorithms)
        ax.set_yticklabels(weight_names)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标签
        for i in range(len(weight_names)):
            for j in range(len(algorithms)):
                text = ax.text(j, i, f'{data_matrix[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(metric_label)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric_label, rotation=90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'aoi_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AoI性能热力图已保存到: {os.path.join(save_dir, 'aoi_performance_heatmap.png')}")


def plot_weight_combination_effects(all_results, system, save_dir='results/aoi/'):
    """绘制权重组合对性能的影响"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择最好的算法（基于平均适应度）
    best_algorithms = {}
    for weight_name, results in all_results.items():
        best_alg = min(results.keys(), key=lambda x: results[x]['mean_fitness'])
        best_algorithms[weight_name] = best_alg
    
    # 准备数据
    weight_names = list(all_results.keys())
    metrics = {
        'energy': [],
        'delay': [],
        'aoi': [],
        'fitness': []
    }
    
    for weight_name in weight_names:
        best_alg = best_algorithms[weight_name]
        data = all_results[weight_name][best_alg]
        metrics_list = data['detailed_metrics']
        
        metrics['energy'].append(np.mean([m['avg_energy'] for m in metrics_list]))
        metrics['delay'].append(np.mean([m['avg_delay'] for m in metrics_list]))
        metrics['aoi'].append(np.mean([m['avg_aoi'] for m in metrics_list]))
        metrics['fitness'].append(data['mean_fitness'])
    
    # 归一化数据
    for key in ['energy', 'delay', 'aoi']:
        max_val = max(metrics[key])
        metrics[key] = [v/max_val for v in metrics[key]]
    
    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics) - 1, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    for i, weight_name in enumerate(weight_names):
        values = [metrics['energy'][i], metrics['delay'][i], metrics['aoi'][i]]
        values += [values[0]]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'{weight_name} ({best_algorithms[weight_name]})')
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['能耗', '延迟', 'AoI'])
    ax.set_ylim(0, 1.2)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.title('不同权重组合下的性能权衡')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_combination_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"权重组合雷达图已保存到: {os.path.join(save_dir, 'weight_combination_radar.png')}")


def plot_aoi_distribution(all_results, system, save_dir='results/aoi/'):
    """绘制AoI分布图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择平衡权重的结果进行详细分析
    balanced_results = all_results.get('Balanced', next(iter(all_results.values())))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (alg_name, data) in enumerate(balanced_results.items()):
        ax = axes[idx // 2, idx % 2]
        
        # 使用最优解
        best_idx = np.argmin(data['best_fitness_values'])
        best_solution = data['best_solutions'][best_idx]
        
        # 应用解决方案
        system.apply_solution(best_solution)
        
        # 收集不同类型任务的AoI
        task_type_aois = {}
        aoi_model = AoIModel(system, DelayModel(system))
        
        for task in system.tasks:
            if task.update_interval is not None:
                task_type = get_task_type(task)
                if task_type not in task_type_aois:
                    task_type_aois[task_type] = []
                
                try:
                    aoi = aoi_model.calculate_average_aoi(task)
                    task_type_aois[task_type].append(aoi)
                except:
                    pass
        
        # 绘制箱线图
        if task_type_aois:
            labels = list(task_type_aois.keys())
            data_to_plot = [task_type_aois[label] for label in labels]
            
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(box_plot['boxes'], colors[:len(labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('任务类型')
            ax.set_ylabel('AoI (s)')
            ax.set_title(f'{alg_name} - AoI分布')
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'aoi_distribution_by_task_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AoI分布图已保存到: {os.path.join(save_dir, 'aoi_distribution_by_task_type.png')}")


def run_aoi_optimization_comparison():
    """运行AoI优化对比实验"""
    print("="*100)
    print("AoI优化对比实验")
    print("="*100)
    
    # 创建系统
    system = create_aoi_aware_system(
        num_devices=20, 
        num_edge_servers=4, 
        num_cloud_servers=2, 
        num_tasks=40
    )
    
    # 运行两种场景的对比
    scenarios = {
        'Without_AoI': {'w_energy': 0.5, 'w_delay': 0.5, 'w_aoi': 0.0},
        'With_AoI': {'w_energy': 0.3, 'w_delay': 0.3, 'w_aoi': 0.4}
    }
    
    comparison_results = {}
    
    for scenario_name, weights in scenarios.items():
        print(f"\n运行场景: {scenario_name}")
        results = run_algorithm_comparison_with_weights(
            system,
            weights['w_energy'],
            weights['w_delay'],
            weights['w_aoi'],
            max_iter=150,
            population_size=50,
            n_runs=5
        )
        comparison_results[scenario_name] = results
    
    # 分析对比结果
    analyze_aoi_optimization_impact(comparison_results, system)
    
    return comparison_results


def analyze_aoi_optimization_impact(comparison_results, system):
   """分析AoI优化的影响"""
   print("\n" + "="*80)
   print("AoI优化影响分析")
   print("="*80)
   
   # 计算改进率
   for alg_name in comparison_results['Without_AoI'].keys():
       without_aoi = comparison_results['Without_AoI'][alg_name]
       with_aoi = comparison_results['With_AoI'][alg_name]
       
       # 获取平均指标
       without_metrics = without_aoi['detailed_metrics']
       with_metrics = with_aoi['detailed_metrics']
       
       avg_aoi_without = np.mean([m['avg_aoi'] for m in without_metrics])
       avg_aoi_with = np.mean([m['avg_aoi'] for m in with_metrics])
       
       avg_energy_without = np.mean([m['avg_energy'] for m in without_metrics])
       avg_energy_with = np.mean([m['avg_energy'] for m in with_metrics])
       
       avg_delay_without = np.mean([m['avg_delay'] for m in without_metrics])
       avg_delay_with = np.mean([m['avg_delay'] for m in with_metrics])
       
       # 计算改进率
       aoi_improvement = (avg_aoi_without - avg_aoi_with) / avg_aoi_without * 100
       energy_change = (avg_energy_with - avg_energy_without) / avg_energy_without * 100
       delay_change = (avg_delay_with - avg_delay_without) / avg_delay_without * 100
       
       print(f"\n{alg_name}算法:")
       print(f"  AoI改进: {aoi_improvement:.2f}% (从 {avg_aoi_without:.3f}s 到 {avg_aoi_with:.3f}s)")
       print(f"  能耗变化: {energy_change:+.2f}% (从 {avg_energy_without:.3f}J 到 {avg_energy_with:.3f}J)")
       print(f"  延迟变化: {delay_change:+.2f}% (从 {avg_delay_without:.3f}s 到 {avg_delay_with:.3f}s)")


def run_dynamic_aoi_experiment(system, time_slots=24):
   """运行动态AoI实验，模拟一天中不同时段的负载变化"""
   print("\n" + "="*100)
   print("动态AoI实验 - 24小时负载变化")
   print("="*100)
   
   # 定义一天中的负载模式（24小时）
   load_multipliers = [
       0.3, 0.3, 0.3, 0.3, 0.4, 0.6,  # 0-5点：低负载
       0.8, 1.2, 1.5, 1.3, 1.0, 0.9,  # 6-11点：早高峰
       1.1, 1.2, 1.0, 0.9, 0.8, 0.9,  # 12-17点：午后
       1.4, 1.6, 1.5, 1.2, 0.8, 0.5   # 18-23点：晚高峰后逐渐降低
   ]
   
   results_over_time = []
   
   delay_model = DelayModel(system)
   energy_model = EnergyModel(system)
   aoi_model = AoIModel(system, delay_model)
   
   # 使用TLBOHHO算法（基于之前实验表现最好）
   algorithm = TLBOHHO(
       system, delay_model, energy_model, aoi_model,
       max_iter=100, population_size=40,
       w_energy=0.3, w_delay=0.3, w_aoi=0.4,
       hho_prob=0.3, verbose=False
   )
   
   for hour in range(time_slots):
       print(f"\n时段 {hour}:00 - {hour+1}:00 (负载系数: {load_multipliers[hour]:.1f})")
       
       # 调整任务到达率
       for task in system.tasks:
           task.arrival_rate = task.arrival_rate * load_multipliers[hour]
           # 根据负载调整更新间隔
           if load_multipliers[hour] > 1.0:
               # 高负载时，更新间隔可能需要增加
               task.update_interval = task.update_interval * (1 + (load_multipliers[hour] - 1) * 0.2)
       
       # 运行优化
       best_solution, best_fitness, _ = algorithm.optimize()
       
       # 应用解决方案并计算指标
       system.apply_solution(best_solution)
       metrics = calculate_detailed_metrics(system, delay_model, energy_model, aoi_model)
       metrics['hour'] = hour
       metrics['load_multiplier'] = load_multipliers[hour]
       
       results_over_time.append(metrics)
       
       print(f"  平均AoI: {metrics['avg_aoi']:.3f}s")
       print(f"  AoI违规率: {metrics['aoi_violations']/len(system.tasks)*100:.1f}%")
   
   # 绘制动态结果
   plot_dynamic_aoi_results(results_over_time)
   
   return results_over_time


def plot_dynamic_aoi_results(results_over_time, save_dir='results/aoi/'):
   """绘制动态AoI实验结果"""
   os.makedirs(save_dir, exist_ok=True)
   
   hours = [r['hour'] for r in results_over_time]
   avg_aois = [r['avg_aoi'] for r in results_over_time]
   avg_energies = [r['avg_energy'] for r in results_over_time]
   avg_delays = [r['avg_delay'] for r in results_over_time]
   load_multipliers = [r['load_multiplier'] for r in results_over_time]
   
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   # 1. AoI随时间变化
   ax1 = axes[0, 0]
   ax1.plot(hours, avg_aois, 'b-', linewidth=2, marker='o')
   ax1.set_xlabel('时间 (小时)')
   ax1.set_ylabel('平均AoI (s)')
   ax1.set_title('24小时平均AoI变化')
   ax1.grid(True, alpha=0.3)
   ax1.set_xticks(range(0, 24, 3))
   
   # 2. 负载与AoI关系
   ax2 = axes[0, 1]
   ax2_twin = ax2.twinx()
   ax2.bar(hours, load_multipliers, alpha=0.3, color='gray', label='负载系数')
   ax2_twin.plot(hours, avg_aois, 'r-', linewidth=2, marker='s', label='平均AoI')
   ax2.set_xlabel('时间 (小时)')
   ax2.set_ylabel('负载系数', color='gray')
   ax2_twin.set_ylabel('平均AoI (s)', color='red')
   ax2.set_title('负载与AoI的关系')
   ax2.set_xticks(range(0, 24, 3))
   
   # 3. 能耗随时间变化
   ax3 = axes[1, 0]
   ax3.plot(hours, avg_energies, 'g-', linewidth=2, marker='^')
   ax3.set_xlabel('时间 (小时)')
   ax3.set_ylabel('平均能耗 (J)')
   ax3.set_title('24小时平均能耗变化')
   ax3.grid(True, alpha=0.3)
   ax3.set_xticks(range(0, 24, 3))
   
   # 4. 三个指标的综合对比
   ax4 = axes[1, 1]
   # 归一化数据以便在同一图中显示
   norm_aois = np.array(avg_aois) / np.max(avg_aois)
   norm_energies = np.array(avg_energies) / np.max(avg_energies)
   norm_delays = np.array(avg_delays) / np.max(avg_delays)
   
   ax4.plot(hours, norm_aois, 'b-', linewidth=2, label='AoI (归一化)')
   ax4.plot(hours, norm_energies, 'g-', linewidth=2, label='能耗 (归一化)')
   ax4.plot(hours, norm_delays, 'r-', linewidth=2, label='延迟 (归一化)')
   ax4.set_xlabel('时间 (小时)')
   ax4.set_ylabel('归一化值')
   ax4.set_title('性能指标综合对比')
   ax4.legend()
   ax4.grid(True, alpha=0.3)
   ax4.set_xticks(range(0, 24, 3))
   
   plt.tight_layout()
   plt.savefig(os.path.join(save_dir, 'dynamic_aoi_results.png'), dpi=300, bbox_inches='tight')
   plt.close()
   print(f"动态AoI结果已保存到: {os.path.join(save_dir, 'dynamic_aoi_results.png')}")


def run_aoi_sensitivity_analysis(system):
   """运行AoI权重敏感性分析"""
   print("\n" + "="*100)
   print("AoI权重敏感性分析")
   print("="*100)
   
   # 定义AoI权重范围
   aoi_weights = np.linspace(0, 0.8, 9)  # 从0到0.8，步长0.1
   
   results = {}
   
   delay_model = DelayModel(system)
   energy_model = EnergyModel(system)
   aoi_model = AoIModel(system, delay_model)
   
   for w_aoi in aoi_weights:
       # 调整其他权重
       remaining_weight = 1.0 - w_aoi
       w_energy = remaining_weight * 0.5
       w_delay = remaining_weight * 0.5
       
       print(f"\n测试权重: w_aoi={w_aoi:.1f}, w_energy={w_energy:.2f}, w_delay={w_delay:.2f}")
       
       # 使用TLBOHHO算法
       algorithm = TLBOHHO(
           system, delay_model, energy_model, aoi_model,
           max_iter=100, population_size=40,
           w_energy=w_energy, w_delay=w_delay, w_aoi=w_aoi,
           hho_prob=0.3, verbose=False
       )
       
       # 运行多次取平均
       metrics_list = []
       for run in range(3):
           best_solution, best_fitness, _ = algorithm.optimize()
           system.apply_solution(best_solution)
           metrics = calculate_detailed_metrics(system, delay_model, energy_model, aoi_model)
           metrics_list.append(metrics)
       
       # 计算平均值
       avg_metrics = {
           'avg_aoi': np.mean([m['avg_aoi'] for m in metrics_list]),
           'avg_energy': np.mean([m['avg_energy'] for m in metrics_list]),
           'avg_delay': np.mean([m['avg_delay'] for m in metrics_list]),
           'aoi_violations': np.mean([m['aoi_violations'] for m in metrics_list])
       }
       
       results[w_aoi] = avg_metrics
   
   # 绘制敏感性分析结果
   plot_aoi_sensitivity_analysis(results)
   
   return results


def plot_aoi_sensitivity_analysis(results, save_dir='results/aoi/'):
   """绘制AoI权重敏感性分析结果"""
   os.makedirs(save_dir, exist_ok=True)
   
   weights = list(results.keys())
   avg_aois = [results[w]['avg_aoi'] for w in weights]
   avg_energies = [results[w]['avg_energy'] for w in weights]
   avg_delays = [results[w]['avg_delay'] for w in weights]
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
   
   # 1. 性能指标随AoI权重变化
   ax1.plot(weights, avg_aois, 'b-', linewidth=2, marker='o', label='平均AoI')
   ax1_twin = ax1.twinx()
   ax1_twin.plot(weights, avg_energies, 'g-', linewidth=2, marker='s', label='平均能耗')
   ax1_twin.plot(weights, avg_delays, 'r-', linewidth=2, marker='^', label='平均延迟')
   
   ax1.set_xlabel('AoI权重')
   ax1.set_ylabel('平均AoI (s)', color='blue')
   ax1_twin.set_ylabel('能耗(J) / 延迟(s)')
   ax1.set_title('性能指标随AoI权重的变化')
   ax1.grid(True, alpha=0.3)
   
   # 添加图例
   lines1, labels1 = ax1.get_legend_handles_labels()
   lines2, labels2 = ax1_twin.get_legend_handles_labels()
   ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
   
   # 2. 归一化性能对比
   ax2.plot(weights, np.array(avg_aois)/avg_aois[0], 'b-', linewidth=2, marker='o', label='AoI (相对变化)')
   ax2.plot(weights, np.array(avg_energies)/avg_energies[0], 'g-', linewidth=2, marker='s', label='能耗 (相对变化)')
   ax2.plot(weights, np.array(avg_delays)/avg_delays[0], 'r-', linewidth=2, marker='^', label='延迟 (相对变化)')
   
   ax2.set_xlabel('AoI权重')
   ax2.set_ylabel('相对变化比例')
   ax2.set_title('归一化性能指标变化')
   ax2.legend()
   ax2.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig(os.path.join(save_dir, 'aoi_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
   plt.close()
   print(f"AoI敏感性分析结果已保存到: {os.path.join(save_dir, 'aoi_sensitivity_analysis.png')}")


# 主函数
def main():
   """主函数 - 运行完整的AoI实验"""
   print("选择AoI实验模式:")
   print("1. 完整AoI实验 (多权重组合对比)")
   print("2. AoI优化影响分析")
   print("3. 动态AoI实验 (24小时)")
   print("4. AoI权重敏感性分析")
   print("5. 运行所有实验")
   
   choice = input("请输入选择 (1-5): ").strip()
   
   if choice == '1':
       system, results = run_aoi_experiment()
   elif choice == '2':
       results = run_aoi_optimization_comparison()
   elif choice == '3':
       system = create_aoi_aware_system()
       results = run_dynamic_aoi_experiment(system)
   elif choice == '4':
       system = create_aoi_aware_system()
       results = run_aoi_sensitivity_analysis(system)
   elif choice == '5':
       # 运行所有实验
       print("\n运行完整AoI实验套件...")
       
       # 1. 完整AoI实验
       system, aoi_results = run_aoi_experiment()
       
       # 2. AoI优化影响分析
       comparison_results = run_aoi_optimization_comparison()
       
       # 3. 动态AoI实验
       dynamic_results = run_dynamic_aoi_experiment(system)
       
       # 4. 敏感性分析
       sensitivity_results = run_aoi_sensitivity_analysis(system)
       
       print("\n所有AoI实验完成！")
   else:
       print("无效选择，运行默认完整AoI实验")
       system, results = run_aoi_experiment()


if __name__ == "__main__":
   main()