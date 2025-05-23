# src/experiments/complete_realistic_experiment.py
import os
import numpy as np
from ..models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..algorithms.tlbo import TLBO
from ..algorithms.tlbo_hho import TLBOHHO
from ..algorithms.ga import GA
from ..algorithms.gwo import GWO
import matplotlib.pyplot as plt
from src.utils.plotting_utils import init_plotting_style
init_plotting_style()

# 导入之前的系统创建函数
from .realistic_system_setup import (
    create_realistic_edge_computing_system,
    create_battery_constrained_scenario,
    create_computation_heavy_scenario,
    create_network_optimized_scenario,
    analyze_task_execution_feasibility,
    print_feasibility_analysis,
    calculate_offload_necessity,
    plot_task_characteristics,
    plot_device_capabilities
)


def run_complete_experiment(system_config=None):
    """运行完整的实验，包括系统分析和算法对比"""
    
    # 1. 创建或加载系统
    if system_config is None:
        print("="*100)
        print("1. 创建现实边缘计算系统")
        print("="*100)
        system = create_realistic_edge_computing_system(
            num_devices=20, 
            num_edge_servers=4, 
            num_cloud_servers=2, 
            num_tasks=40
        )
    else:
        system = SystemModel.from_config(system_config)
    
    # 2. 系统可行性分析
    print("\n" + "="*100)
    print("2. 系统可行性分析")
    print("="*100)
    
    analysis_results = analyze_task_execution_feasibility(system)
    print_feasibility_analysis(analysis_results)
    
    # 计算卸载必要性
    offload_stats = calculate_offload_necessity(analysis_results)
    print(f"\n卸载必要性分析:")
    print(f"  必须卸载的任务: {offload_stats['must_offload']}/{offload_stats['total']} ({offload_stats['must_offload_rate']:.1f}%)")
    print(f"  本地不可行的任务: {offload_stats['local_infeasible']}/{offload_stats['total']} ({offload_stats['local_infeasible_rate']:.1f}%)")
    print(f"  卸载更优的任务: {offload_stats['offload_better']}/{offload_stats['total']} ({offload_stats['offload_better_rate']:.1f}%)")
    
    # 3. 生成系统特征图
    print("\n" + "="*100)
    print("3. 生成系统特征图")
    print("="*100)
    
    plot_task_characteristics(system)
    plot_device_capabilities(system)
    print("系统特征图已保存到 results/ 目录")
    
    # 4. 运行算法对比实验
    print("\n" + "="*100)
    print("4. 运行算法对比实验")
    print("="*100)
    
    results = run_algorithm_comparison(system, max_iter=200, population_size=50, n_runs=10)
    
    # 5. 分析实验结果
    print("\n" + "="*100)
    print("5. 实验结果分析")
    print("="*100)
    
    analyze_experiment_results(results, system)
    
    # 6. 绘制结果图表
    print("\n" + "="*100)
    print("6. 生成实验结果图表")
    print("="*100)
    
    plot_convergence_curves(results)
    plot_task_allocation_comparison(results, system)
    plot_performance_metrics_comparison(results, system)
    
    print("实验完成！所有结果已保存到 results/ 目录")
    
    return system, results


def run_algorithm_comparison(system, max_iter=200, population_size=50, n_runs=10):
    """运行算法对比实验"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    
    # 初始化算法，使用修改后的适应度函数权重
    algorithms = {
        'TLBO': TLBO(
            system, delay_model, energy_model, 
            max_iter=max_iter, population_size=population_size,
            w_energy=0.3, w_delay=0.7, verbose=False
        ),
        'TLBOHHO': TLBOHHO(
            system, delay_model, energy_model, 
            max_iter=max_iter, population_size=population_size,
            w_energy=0.3, w_delay=0.7, hho_prob=0.3, verbose=False
        ),
        'GA': GA(
            system, delay_model, energy_model,
            max_iter=max_iter, population_size=population_size,
            w_energy=0.3, w_delay=0.7, verbose=False
        ),
        'GWO': GWO(
            system, delay_model, energy_model,
            max_iter=max_iter, population_size=population_size,
            w_energy=0.3, w_delay=0.7, verbose=False
        )
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\n运行 {name} 算法...")
        
        best_fitness_history = np.zeros((n_runs, max_iter + 1))
        best_solutions = []
        best_fitness_values = []
        
        for run in range(n_runs):
            print(f"  运行 {run + 1}/{n_runs}", end=' ... ')
            
            # 重置系统状态
            for task in system.tasks:
                task.execution_location = None
                task.execution_node_id = None
                task.allocated_resource = None
                task.delay = None
                task.energy = None
            
            # 运行算法
            best_solution, best_fitness, history = algorithm.optimize()
            
            # 确保历史长度正确
            if len(history) <= max_iter + 1:
                best_fitness_history[run, :len(history)] = history
                # 如果历史较短，用最后一个值填充
                if len(history) < max_iter + 1:
                    best_fitness_history[run, len(history):] = history[-1]
            else:
                best_fitness_history[run, :] = history[:max_iter + 1]
            
            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            
            print(f"完成，最优适应度: {best_fitness:.6f}")
        
        # 计算平均收敛曲线
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
            'best_fitness': np.min(best_fitness_values)
        }
        
        print(f"{name} 完成：平均适应度 {results[name]['mean_fitness']:.6f} ± {results[name]['std_fitness']:.6f}")
    
    return results


def analyze_experiment_results(results, system):
    """分析实验结果"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    
    print("\n算法性能对比:")
    print("-" * 80)
    print(f"{'算法':<12} {'平均适应度':<15} {'标准差':<10} {'最优适应度':<15} {'任务分配':<25}")
    print("-" * 80)
    
    for name, data in results.items():
        # 使用最优解进行分析
        best_run_idx = np.argmin(data['best_fitness_values'])
        best_solution = data['best_solutions'][best_run_idx]
        
        # 应用最优解
        system.apply_solution(best_solution)
        
        # 分析任务分配
        allocation = analyze_task_allocation(system)
        allocation_str = f"D:{allocation['device']}/E:{allocation['edge']}/C:{allocation['cloud']}"
        
        print(f"{name:<12} {data['mean_fitness']:<15.6f} {data['std_fitness']:<10.6f} {data['best_fitness']:<15.6f} {allocation_str:<25}")
    
    # 详细分析最优算法
    best_algorithm = min(results.keys(), key=lambda x: results[x]['best_fitness'])
    print(f"\n最优算法: {best_algorithm}")
    print("-" * 50)
    
    best_data = results[best_algorithm]
    best_run_idx = np.argmin(best_data['best_fitness_values'])
    best_solution = best_data['best_solutions'][best_run_idx]
    
    system.apply_solution(best_solution)
    
    # 计算详细指标
    total_energy = 0
    total_delay = 0
    delay_violations = 0
    
    for task in system.tasks:
        if task.delay is None:
            task.delay = delay_model.calculate_total_delay(task)
        if task.energy is None:
            task.energy = energy_model.calculate_total_energy(task)
        
        total_energy += task.energy
        total_delay += task.delay
        
        if task.delay > task.max_delay:
            delay_violations += 1
    
    print(f"总能耗: {total_energy:.6f} J")
    print(f"总延迟: {total_delay:.6f} s")
    print(f"平均延迟: {total_delay/len(system.tasks):.6f} s")
    print(f"延迟违规任务: {delay_violations}/{len(system.tasks)} ({delay_violations/len(system.tasks)*100:.1f}%)")
    
    # 分析不同类型任务的分配情况
    analyze_task_type_allocation(system)


def analyze_task_allocation(system):
    """分析任务分配情况"""
    device_count = 0
    edge_count = 0
    cloud_count = 0
    
    for task in system.tasks:
        if task.execution_location == 'device':
            device_count += 1
        elif task.execution_location == 'edge':
            edge_count += 1
        elif task.execution_location == 'cloud':
            cloud_count += 1
    
    return {
        'device': device_count,
        'edge': edge_count,
        'cloud': cloud_count
    }


def analyze_task_type_allocation(system):
    """分析不同类型任务的分配情况"""
    from .realistic_system_setup import get_task_type
    
    type_allocation = {}
    
    for task in system.tasks:
        task_type = get_task_type(task)
        if task_type not in type_allocation:
            type_allocation[task_type] = {'device': 0, 'edge': 0, 'cloud': 0, 'total': 0}
        
        type_allocation[task_type]['total'] += 1
        
        if task.execution_location == 'device':
            type_allocation[task_type]['device'] += 1
        elif task.execution_location == 'edge':
            type_allocation[task_type]['edge'] += 1
        elif task.execution_location == 'cloud':
            type_allocation[task_type]['cloud'] += 1
    
    print(f"\n不同类型任务的分配情况:")
    print("-" * 60)
    print(f"{'任务类型':<20} {'设备':<8} {'边缘':<8} {'云端':<8} {'总数':<8}")
    print("-" * 60)
    
    for task_type, allocation in type_allocation.items():
        print(f"{task_type:<20} {allocation['device']:<8} {allocation['edge']:<8} {allocation['cloud']:<8} {allocation['total']:<8}")


def plot_convergence_curves(results, save_path='results/convergence_curves.png'):
    """绘制收敛曲线"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, (name, data) in enumerate(results.items()):
        avg_history = data['avg_fitness_history']
        std_history = data['std_fitness_history']
        
        x = np.arange(len(avg_history))
        color = colors[i % len(colors)]
        
        plt.plot(x, avg_history, label=name, color=color, linewidth=2)
        plt.fill_between(x, avg_history - std_history, avg_history + std_history, 
                        alpha=0.2, color=color)
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('适应度值', fontsize=12)
    plt.title('不同算法的收敛曲线对比', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"收敛曲线已保存到: {save_path}")


def plot_task_allocation_comparison(results, system, save_path='results/task_allocation_comparison.png'):
    """绘制任务分配对比图"""
    algorithms = list(results.keys())
    allocations = []
    
    for name, data in results.items():
        # 使用最优解
        best_run_idx = np.argmin(data['best_fitness_values'])
        best_solution = data['best_solutions'][best_run_idx]
        
        system.apply_solution(best_solution)
        allocation = analyze_task_allocation(system)
        allocations.append(allocation)
    
    # 创建堆积柱状图
    device_counts = [alloc['device'] for alloc in allocations]
    edge_counts = [alloc['edge'] for alloc in allocations]
    cloud_counts = [alloc['cloud'] for alloc in allocations]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    p1 = ax.bar(x, device_counts, width, label='设备', color='lightblue')
    p2 = ax.bar(x, edge_counts, width, bottom=device_counts, label='边缘', color='lightgreen')
    p3 = ax.bar(x, cloud_counts, width, bottom=np.array(device_counts) + np.array(edge_counts), 
                label='云端', color='lightcoral')
    
    ax.set_xlabel('算法')
    ax.set_ylabel('任务数量')
    ax.set_title('不同算法的任务分配对比')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    
    # 添加数值标签
    for i, (device, edge, cloud) in enumerate(zip(device_counts, edge_counts, cloud_counts)):
        total = device + edge + cloud
        if device > 0:
            ax.text(i, device/2, str(device), ha='center', va='center')
        if edge > 0:
            ax.text(i, device + edge/2, str(edge), ha='center', va='center')
        if cloud > 0:
            ax.text(i, device + edge + cloud/2, str(cloud), ha='center', va='center')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"任务分配对比图已保存到: {save_path}")


def plot_performance_metrics_comparison(results, system, save_path='results/performance_metrics.png'):
    """绘制性能指标对比图"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    
    algorithms = list(results.keys())
    metrics = {'energy': [], 'delay': [], 'violation_rate': []}
    
    for name, data in results.items():
        best_run_idx = np.argmin(data['best_fitness_values'])
        best_solution = data['best_solutions'][best_run_idx]
        
        system.apply_solution(best_solution)
        
        total_energy = 0
        total_delay = 0
        delay_violations = 0
        
        for task in system.tasks:
            if task.delay is None:
                task.delay = delay_model.calculate_total_delay(task)
            if task.energy is None:
                task.energy = energy_model.calculate_total_energy(task)
            
            total_energy += task.energy
            total_delay += task.delay
            
            if task.delay > task.max_delay:
                delay_violations += 1
        
        metrics['energy'].append(total_energy)
        metrics['delay'].append(total_delay)
        metrics['violation_rate'].append(delay_violations / len(system.tasks) * 100)
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    x = np.arange(len(algorithms))
    
    # 1. 总能耗对比
    bars1 = ax1.bar(x, metrics['energy'], color='skyblue', alpha=0.7)
    ax1.set_xlabel('算法')
    ax1.set_ylabel('总能耗 (J)')
    ax1.set_title('总能耗对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    
    # 添加数值标签
    for bar, value in zip(bars1, metrics['energy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics['energy'])*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 总延迟对比
    bars2 = ax2.bar(x, metrics['delay'], color='lightgreen', alpha=0.7)
    ax2.set_xlabel('算法')
    ax2.set_ylabel('总延迟 (s)')
    ax2.set_title('总延迟对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    
    for bar, value in zip(bars2, metrics['delay']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics['delay'])*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 延迟违规率对比
    bars3 = ax3.bar(x, metrics['violation_rate'], color='lightcoral', alpha=0.7)
    ax3.set_xlabel('算法')
    ax3.set_ylabel('延迟违规率 (%)')
    ax3.set_title('延迟违规率对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms)
    
    for bar, value in zip(bars3, metrics['violation_rate']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics['violation_rate'])*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. 适应度值对比（箱线图）
    fitness_data = [data['best_fitness_values'] for data in results.values()]
    box_plot = ax4.boxplot(fitness_data, labels=algorithms, patch_artist=True)
    
    colors = ['lightblue', 'orange', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('算法')
    ax4.set_ylabel('适应度值')
    ax4.set_title('适应度值分布对比')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能指标对比图已保存到: {save_path}")


def run_scenario_comparison():
    """运行多场景对比实验"""
    print("="*100)
    print("多场景算法对比实验")
    print("="*100)
    
    # 创建基础系统
    base_system = create_realistic_edge_computing_system(
        num_devices=20, 
        num_edge_servers=4, 
        num_cloud_servers=2, 
        num_tasks=40
    )
    
    scenarios = {
        'base': base_system,
        'battery_constrained': create_battery_constrained_scenario(base_system),
        'computation_heavy': create_computation_heavy_scenario(base_system),
        'network_optimized': create_network_optimized_scenario(base_system)
    }
    
    all_scenario_results = {}
    
    for scenario_name, system in scenarios.items():
        print(f"\n{'='*50}")
        print(f"场景: {scenario_name.upper()}")
        print(f"{'='*50}")
        
        # 运行算法对比
        results = run_algorithm_comparison(system, max_iter=150, population_size=40, n_runs=5)
        all_scenario_results[scenario_name] = results
        
        # 分析结果
        analyze_experiment_results(results, system)
        
        # 生成场景特定的图表
        plot_convergence_curves(results, f'results/{scenario_name}_convergence.png')
        plot_task_allocation_comparison(results, system, f'results/{scenario_name}_allocation.png')
    
    return all_scenario_results


# 主函数
def main():
    """主函数 - 运行完整实验"""
    print("选择实验模式:")
    print("1. 完整实验 (系统分析 + 算法对比)")
    print("2. 仅算法对比")
    print("3. 多场景对比")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == '1':
        system, results = run_complete_experiment()
    elif choice == '2':
        system = create_realistic_edge_computing_system()
        results = run_algorithm_comparison(system)
        analyze_experiment_results(results, system)
        plot_convergence_curves(results)
        plot_task_allocation_comparison(results, system)
        plot_performance_metrics_comparison(results, system)
    elif choice == '3':
        all_results = run_scenario_comparison()
    else:
        print("无效选择，运行默认完整实验")
        system, results = run_complete_experiment()


if __name__ == "__main__":
    main()