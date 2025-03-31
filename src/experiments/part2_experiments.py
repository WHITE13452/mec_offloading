# src/experiments/part2_experiments.py
import os
import numpy as np
import matplotlib.pyplot as plt
from ..models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..algorithms.tlbo_plus import TLBOPlus
from ..algorithms.mo_tlbo import MOTLBO


def create_test_system_with_aoi(num_devices=5, num_edge_servers=3, num_cloud_servers=1, num_tasks=10):
    """创建考虑AoI的测试系统"""
    system = SystemModel()
    
    # 添加设备
    for i in range(num_devices):
        device = Device(
            device_id=i,
            max_cpu_frequency=np.random.uniform(1.0e9, 2.0e9),  # 1-2 GHz
            energy_coefficient=np.random.uniform(0.8e-27, 1.2e-27),
            transmission_power=np.random.uniform(0.4, 0.6)  # 0.4-0.6 W
        )
        system.add_device(device)
    
    # 添加边缘服务器
    for i in range(num_edge_servers):
        server = EdgeServer(
            server_id=i,
            max_cpu_frequency=np.random.uniform(2.5e9, 3.5e9),  # 2.5-3.5 GHz
            energy_coefficient=np.random.uniform(1.5e-27, 2.5e-27),
            transmission_power=np.random.uniform(0.8, 1.2)  # 0.8-1.2 W
        )
        system.add_edge_server(server)
    
    # 添加云服务器
    for i in range(num_cloud_servers):
        server = CloudServer(
            server_id=i,
            max_cpu_frequency=np.random.uniform(3.5e9, 4.5e9),  # 3.5-4.5 GHz
            energy_coefficient=np.random.uniform(2.5e-27, 3.5e-27)
        )
        system.add_cloud_server(server)
    
    # 添加任务，包括更新间隔和最大AoI
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            data_size=np.random.uniform(0.5e6, 1.5e6),  # 0.5-1.5 MB
            computation_complexity=np.random.uniform(80, 120),  # 80-120 cycles/bit
            arrival_rate=np.random.uniform(0.05, 0.15),  # 0.05-0.15 tasks/s
            priority=np.random.uniform(0.5, 1.5),
            max_delay=np.random.uniform(0.8, 1.2),  # 0.8-1.2 s
            update_interval=np.random.uniform(0.3, 0.7),  # 0.3-0.7 s
            max_aoi=np.random.uniform(1.0, 2.0)  # 1.0-2.0 s
        )
        system.add_task(task)
    
    # 设置网络参数
    for device in system.devices:
        for edge_server in system.edge_servers:
            rate = np.random.uniform(8e6, 12e6)  # 8-12 Mbps
            bandwidth = np.random.uniform(15e6, 25e6)  # 15-25 MHz
            system.set_device_to_edge_rate(device.device_id, edge_server.server_id, rate, bandwidth)
    
    for edge_server in system.edge_servers:
        for cloud_server in system.cloud_servers:
            rate = np.random.uniform(80e6, 120e6)  # 80-120 Mbps
            bandwidth = np.random.uniform(150e6, 250e6)  # 150-250 MHz
            system.set_edge_to_cloud_rate(edge_server.server_id, cloud_server.server_id, rate, bandwidth)
    
    # 设置任务到达率
    for device in system.devices:
        for task in system.tasks:
            device.arrival_rates[task.task_id] = np.random.uniform(0.05, 0.15)
    
    for edge_server in system.edge_servers:
        for task in system.tasks:
            edge_server.arrival_rates[task.task_id] = np.random.uniform(0.01, 0.05)
    
    for cloud_server in system.cloud_servers:
        for task in system.tasks:
            cloud_server.arrival_rates[task.task_id] = np.random.uniform(0.005, 0.02)
    
    return system


def run_weighted_sum_optimization(system, max_iter=100, population_size=50, n_runs=5):
    """运行基于加权和的单目标优化"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 不同权重组合
    weight_combinations = [
        {'name': 'Energy-focused', 'w_energy': 0.7, 'w_delay': 0.15, 'w_aoi': 0.15},
        {'name': 'Delay-focused', 'w_energy': 0.15, 'w_delay': 0.7, 'w_aoi': 0.15},
        {'name': 'AoI-focused', 'w_energy': 0.15, 'w_delay': 0.15, 'w_aoi': 0.7},
        {'name': 'Balanced', 'w_energy': 0.33, 'w_delay': 0.33, 'w_aoi': 0.34}
    ]
    
    results = {}
    
    for weights in weight_combinations:
        print(f"Running TLBO+ with weights: {weights['name']}...")
        
        algorithm = TLBOPlus(
            system_model=system,
            delay_model=delay_model,
            energy_model=energy_model,
            aoi_model=aoi_model,
            max_iter=max_iter,
            population_size=population_size,
            w_energy=weights['w_energy'],
            w_delay=weights['w_delay'],
            w_aoi=weights['w_aoi']
        )
        
        best_solutions = []
        best_fitness_values = []
        all_objectives = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}")
            
            best_solution, best_fitness, _ = algorithm.optimize()
            
            # 应用最优解以获取目标值
            system.apply_solution(best_solution)
            
            # 计算各目标值
            total_energy = sum(energy_model.calculate_total_energy(task) for task in system.tasks)
            total_delay = sum(delay_model.calculate_total_delay(task) for task in system.tasks)
            total_aoi = sum(aoi_model.calculate_average_aoi(task) for task in system.tasks if task.update_interval is not None)
            
            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            all_objectives.append((total_energy, total_delay, total_aoi))
        
        # 记录结果
        results[weights['name']] = {
            'best_solutions': best_solutions,
            'best_fitness_values': best_fitness_values,
            'all_objectives': all_objectives
        }
    
    return results


def run_multi_objective_optimization(system, max_iter=100, population_size=50):
    """运行多目标优化"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    print("Running Multi-Objective TLBO...")
    
    algorithm = MOTLBO(
        system_model=system,
        delay_model=delay_model,
        energy_model=energy_model,
        aoi_model=aoi_model,
        max_iter=max_iter,
        population_size=population_size
    )
    
    # 执行优化
    pareto_front = algorithm.optimize()
    
    # 计算Pareto前沿上每个解的目标值
    objectives = []
    
    for solution in pareto_front:
        system.apply_solution(solution)
        
        total_energy = sum(energy_model.calculate_total_energy(task) for task in system.tasks)
        total_delay = sum(delay_model.calculate_total_delay(task) for task in system.tasks)
        total_aoi = sum(aoi_model.calculate_average_aoi(task) for task in system.tasks if task.update_interval is not None)
        
        objectives.append((total_energy, total_delay, total_aoi))
    
    return pareto_front, objectives


def plot_pareto_front(objectives, save_path='results/pareto_front.png'):
    """绘制Pareto前沿"""
    # 提取目标值
    energy_values = [obj[0] for obj in objectives]
    delay_values = [obj[1] for obj in objectives]
    aoi_values = [obj[2] for obj in objectives]
    
    # 绘制3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(energy_values, delay_values, aoi_values, marker='o', s=50, alpha=0.7)
    
    ax.set_xlabel('Energy')
    ax.set_ylabel('Delay')
    ax.set_zlabel('AoI')
    ax.set_title('Pareto Front')
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # 绘制2D投影图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Energy vs Delay
    axs[0].scatter(energy_values, delay_values, marker='o', alpha=0.7)
    axs[0].set_xlabel('Energy')
    axs[0].set_ylabel('Delay')
    axs[0].set_title('Energy vs Delay')
    axs[0].grid(True)
    
    # Energy vs AoI
    axs[1].scatter(energy_values, aoi_values, marker='o', alpha=0.7)
    axs[1].set_xlabel('Energy')
    axs[1].set_ylabel('AoI')
    axs[1].set_title('Energy vs AoI')
    axs[1].grid(True)
    
    # Delay vs AoI
    axs[2].scatter(delay_values, aoi_values, marker='o', alpha=0.7)
    axs[2].set_xlabel('Delay')
    axs[2].set_ylabel('AoI')
    axs[2].set_title('Delay vs AoI')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_2d.png'), dpi=300)
    plt.close()


def compare_weighted_sum_results(results, save_path='results/weight_comparison.png'):
    """比较不同权重组合的结果"""
    # 提取结果
    names = list(results.keys())
    avg_energy = []
    avg_delay = []
    avg_aoi = []
    
    for name, data in results.items():
        energy_values = [obj[0] for obj in data['all_objectives']]
        delay_values = [obj[1] for obj in data['all_objectives']]
        aoi_values = [obj[2] for obj in data['all_objectives']]
        
        avg_energy.append(np.mean(energy_values))
        avg_delay.append(np.mean(delay_values))
        avg_aoi.append(np.mean(aoi_values))
    
    # 绘制柱状图
    x = np.arange(len(names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 归一化值以便在同一图上显示
    norm_energy = avg_energy / np.max(avg_energy)
    norm_delay = avg_delay / np.max(avg_delay)
    norm_aoi = avg_aoi / np.max(avg_aoi)
    
    rects1 = ax.bar(x - width, norm_energy, width, label='Energy')
    rects2 = ax.bar(x, norm_delay, width, label='Delay')
    rects3 = ax.bar(x + width, norm_aoi, width, label='AoI')
    
    ax.set_xlabel('Weight Combination')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Performance Comparison of Different Weight Combinations')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_update_interval_distribution(pareto_front, system, save_path='results/update_interval_distribution.png'):
    """分析Pareto前沿解中更新间隔的分布"""
    all_intervals = []
    
    for solution in pareto_front:
        system.apply_solution(solution)
        
        # 收集所有任务的更新间隔
        intervals = [task.update_interval for task in system.tasks if task.update_interval is not None]
        all_intervals.extend(intervals)
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_intervals, bins=20, alpha=0.7)
    plt.xlabel('Update Interval (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Update Intervals in Pareto Optimal Solutions')
    plt.grid(True)
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return all_intervals


def analyze_task_allocation_distribution(pareto_front, system, save_path='results/task_allocation_distribution.png'):
    """分析Pareto前沿解中任务分配位置的分布"""
    device_counts = []
    edge_counts = []
    cloud_counts = []
    
    for solution in pareto_front:
        system.apply_solution(solution)
        
        # 统计任务分配位置
        device_count = sum(1 for task in system.tasks if task.execution_location == 'device')
        edge_count = sum(1 for task in system.tasks if task.execution_location == 'edge')
        cloud_count = sum(1 for task in system.tasks if task.execution_location == 'cloud')
        
        device_counts.append(device_count)
        edge_counts.append(edge_count)
        cloud_counts.append(cloud_count)
    
    # 计算平均分配比例
    total_tasks = len(system.tasks)
    avg_device_ratio = np.mean(device_counts) / total_tasks
    avg_edge_ratio = np.mean(edge_counts) / total_tasks
    avg_cloud_ratio = np.mean(cloud_counts) / total_tasks
    
    # 绘制饼图
    plt.figure(figsize=(8, 8))
    plt.pie([avg_device_ratio, avg_edge_ratio, avg_cloud_ratio], 
            labels=['Device', 'Edge', 'Cloud'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#ff9999','#66b3ff','#99ff99'])
    plt.axis('equal')
    plt.title('Average Task Allocation Distribution in Pareto Front')
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return {
        'device_ratio': avg_device_ratio,
        'edge_ratio': avg_edge_ratio,
        'cloud_ratio': avg_cloud_ratio
    }


def analyze_aoi_impact(system, delay_model, energy_model, aoi_model, save_path='results/aoi_impact.png'):
    """分析考虑AoI对任务卸载决策的影响"""
    # 生成不同的更新间隔
    update_intervals = np.linspace(0.1, 2.0, 20)
    energy_values = []
    delay_values = []
    aoi_values = []
    
    # 选择一个任务进行分析
    task = system.tasks[0]
    original_interval = task.update_interval
    
    for interval in update_intervals:
        # 设置更新间隔
        task.update_interval = interval
        
        # 优化更新间隔
        optimal_interval = aoi_model.optimize_update_interval(task)
        
        # 计算性能指标
        energy = energy_model.calculate_total_energy(task)
        delay = delay_model.calculate_total_delay(task)
        aoi = aoi_model.calculate_average_aoi(task)
        
        energy_values.append(energy)
        delay_values.append(delay)
        aoi_values.append(aoi)
    
    # 还原原始更新间隔
    task.update_interval = original_interval
    
    # 绘制影响曲线
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Update Interval (s)')
    ax1.set_ylabel('Energy', color=color)
    ax1.plot(update_intervals, energy_values, color=color, marker='o', label='Energy')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('AoI', color=color)
    ax2.plot(update_intervals, aoi_values, color=color, marker='s', label='AoI')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Impact of Update Interval on Energy and AoI')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return update_intervals, energy_values, aoi_values


def main():
    """主函数"""
    # 创建测试系统
    system = create_test_system_with_aoi()
    
    # 创建模型
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 运行基于加权和的单目标优化
    weighted_results = run_weighted_sum_optimization(system)
    
    # 比较不同权重组合的结果
    compare_weighted_sum_results(weighted_results)
    
    # 运行多目标优化
    pareto_front, objectives = run_multi_objective_optimization(system)
    
    # 绘制Pareto前沿
    plot_pareto_front(objectives)
    
    # 分析更新间隔分布
    intervals = analyze_update_interval_distribution(pareto_front, system)
    print(f"\nUpdate interval statistics:")
    print(f"  Mean: {np.mean(intervals):.4f} s")
    print(f"  Std: {np.std(intervals):.4f} s")
    print(f"  Min: {np.min(intervals):.4f} s")
    print(f"  Max: {np.max(intervals):.4f} s")
    
    # 分析任务分配分布
    allocation = analyze_task_allocation_distribution(pareto_front, system)
    print(f"\nTask allocation distribution:")
    print(f"  Device: {allocation['device_ratio']*100:.1f}%")
    print(f"  Edge: {allocation['edge_ratio']*100:.1f}%")
    print(f"  Cloud: {allocation['cloud_ratio']*100:.1f}%")
    
    # 分析AoI对任务卸载决策的影响
    analyze_aoi_impact(system, delay_model, energy_model, aoi_model)
    
    print(f"\nFound {len(pareto_front)} non-dominated solutions in the Pareto front.")


if __name__ == "__main__":
    main()