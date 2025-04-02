# src/experiments/part1_experiments.py
import os
import numpy as np
import matplotlib.pyplot as plt
from ..models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..algorithms.tlbo import TLBO
# from ..algorithms.tlbo_plus import TLBOPlus
from ..algorithms.ga import GA  # 假设我们有遗传算法实现
from ..algorithms.gwo import GWO  # 假设我们有灰狼优化算法实现


def create_test_system(num_devices=5, num_edge_servers=3, num_cloud_servers=1, num_tasks=10):
    """创建测试系统"""
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
    
    # 添加任务
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            data_size=np.random.uniform(0.5e6, 1.5e6),  # 0.5-1.5 MB
            computation_complexity=np.random.uniform(80, 120),  # 80-120 cycles/bit
            arrival_rate=np.random.uniform(0.05, 0.15),  # 0.05-0.15 tasks/s
            priority=np.random.uniform(0.5, 1.5),
            max_delay=np.random.uniform(0.8, 1.2)  # 0.8-1.2 s
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


def run_algorithm_comparison(system, max_iter=100, population_size=50, n_runs=10):
    """比较不同算法的性能"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    
    # 初始化算法
    algorithms = {
        'TLBO': TLBO(system, delay_model, energy_model, max_iter=max_iter, population_size=population_size),
        # 'TLBO+': TLBOPlus(system, delay_model, energy_model, max_iter=max_iter, population_size=population_size),
        'GA': GA(system, delay_model, energy_model, max_iter=max_iter, population_size=population_size),
        'GWO': GWO(system, delay_model, energy_model, max_iter=max_iter, population_size=population_size)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Running {name}...")
        
        best_fitness_history = np.zeros((n_runs, max_iter + 1))
        best_solutions = []
        best_fitness_values = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}")
            
            best_solution, best_fitness, history = algorithm.optimize()
            
            best_fitness_history[run, :len(history)] = history
            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
        
        # 计算平均收敛曲线
        avg_fitness_history = np.mean(best_fitness_history, axis=0)
        std_fitness_history = np.std(best_fitness_history, axis=0)
        
        # 记录结果
        results[name] = {
            'best_solutions': best_solutions,
            'best_fitness_values': best_fitness_values,
            'avg_fitness_history': avg_fitness_history,
            'std_fitness_history': std_fitness_history
        }
    
    return results


def plot_convergence_curves(results, save_path='results/convergence_curves.png'):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))
    
    for name, data in results.items():
        avg_history = data['avg_fitness_history']
        std_history = data['std_fitness_history']
        
        x = np.arange(len(avg_history))
        plt.plot(x, avg_history, label=name)
        plt.fill_between(x, avg_history - std_history, avg_history + std_history, alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('Convergence Curves of Different Algorithms')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_task_allocation(system, best_solution):
    """分析任务分配情况"""
    # 应用最优解
    system.apply_solution(best_solution)
    
    # 统计任务分配情况
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


def main():
    """主函数"""
    # 创建测试系统
    system = create_test_system()
    
    # 运行算法比较
    results = run_algorithm_comparison(system)
    
    # 绘制收敛曲线
    plot_convergence_curves(results)
    
    # 分析最优解
    for name, data in results.items():
        # 找出所有运行中的最优解
        best_run_idx = np.argmin(data['best_fitness_values'])
        best_solution = data['best_solutions'][best_run_idx]
        best_fitness = data['best_fitness_values'][best_run_idx]
        
        print(f"\nBest solution found by {name}:")
        print(f"Fitness value: {best_fitness:.6f}")
        
        # 分析任务分配
        allocation = analyze_task_allocation(system, best_solution)
        print(f"Task allocation: Device={allocation['device']}, Edge={allocation['edge']}, Cloud={allocation['cloud']}")


if __name__ == "__main__":
    main()