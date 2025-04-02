# test_mec_optimization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# 添加源代码路径
sys.path.append(os.path.abspath('.'))

# 导入模型和算法
from src.models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from src.models.delay_model import DelayModel
from src.models.energy_model import EnergyModel
from src.models.aoi_model import AoIModel
from src.algorithms.tlbo import TLBO
from src.algorithms.tlbo_plus import TLBOPlus
from src.algorithms.ga import GA
from src.algorithms.gwo import GWO


def create_simple_system():
    """创建一个简单的测试系统"""
    system = SystemModel()
    
    # 添加设备
    device = Device(
        device_id=0,
        max_cpu_frequency=1.5e9,  # 1.5 GHz
        energy_coefficient=1e-27,
        transmission_power=0.5  # 0.5 W
    )
    system.add_device(device)
    
    # 添加边缘服务器
    edge_server = EdgeServer(
        server_id=0,
        max_cpu_frequency=3.0e9,  # 3.0 GHz
        energy_coefficient=2e-27,
        transmission_power=1.0  # 1.0 W
    )
    system.add_edge_server(edge_server)
    
    # 添加云服务器
    cloud_server = CloudServer(
        server_id=0,
        max_cpu_frequency=4.0e9,  # 4.0 GHz
        energy_coefficient=3e-27
    )
    system.add_cloud_server(cloud_server)
    
    # 添加任务
    task = Task(
        task_id=0,
        data_size=1e6,  # 1 MB
        computation_complexity=100,  # 100 cycles/bit
        arrival_rate=0.1,  # 0.1 tasks/s
        priority=1.0,
        max_delay=1.0,  # 1.0 s
        update_interval=0.5,  # 0.5 s (仅用于AoI测试)
        max_aoi=1.5  # 1.5 s (仅用于AoI测试)
    )
    system.add_task(task)
    
    # 设置网络参数
    system.set_device_to_edge_rate(
        device_id=0,
        edge_id=0,
        rate=10e6,  # 10 Mbps
        bandwidth=20e6  # 20 MHz
    )
    
    system.set_edge_to_cloud_rate(
        edge_id=0,
        cloud_id=0,
        rate=100e6,  # 100 Mbps
        bandwidth=200e6  # 200 MHz
    )
    
    # 设置任务到达率
    device.arrival_rates[0] = 0.1
    edge_server.arrival_rates[0] = 0.05
    cloud_server.arrival_rates[0] = 0.01
    
    return system


def test_models():
    """测试模型实现"""
    print("\n=== 测试模型实现 ===")
    
    # 创建系统
    system = create_simple_system()
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 测试不同执行位置
    locations = ['device', 'edge', 'cloud']
    results = []
    
    for location in locations:
        # 设置任务执行位置
        task = system.tasks[0]
        
        if location == 'device':
            task.execution_location = 'device'
            task.execution_node_id = 0
            task.allocated_resource = 1.5e9  # 使用设备的最大CPU频率
        elif location == 'edge':
            task.execution_location = 'edge'
            task.execution_node_id = 0
            task.allocated_resource = 3.0e9  # 使用边缘服务器的最大CPU频率
        else:  # cloud
            task.execution_location = 'cloud'
            task.execution_node_id = 0
            task.allocated_resource = 4.0e9  # 使用云服务器的最大CPU频率
        
        # 计算延迟、能耗和AoI
        delay = delay_model.calculate_total_delay(task)
        energy = energy_model.calculate_total_energy(task)
        aoi = aoi_model.calculate_average_aoi(task)
        
        results.append((location, delay, energy, aoi))
    
    # 打印结果
    print("执行位置\t延迟(ms)\t能耗(J)\t\tAoI(s)")
    print("-" * 60)
    for location, delay, energy, aoi in results:
        print(f"{location}\t\t{delay*1000:.2f}\t\t{energy:.6f}\t{aoi:.2f}")


def test_algorithms():
    """测试优化算法"""
    print("\n=== 测试优化算法 ===")
    
    # 创建系统
    system = create_simple_system()
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 算法参数
    max_iter = 50
    population_size = 20
    
    # 记录每个算法的性能
    algorithm_times = {}
    algorithm_results = {}
    
    # 测试TLBO
    print("测试TLBO算法...")
    start_time = time.time()
    tlbo = TLBO(
        system_model=system,
        delay_model=delay_model,
        energy_model=energy_model,
        aoi_model=aoi_model,
        max_iter=max_iter,
        population_size=population_size,
        w_energy=0.5,
        w_delay=0.3,
        w_aoi=0.2,
        verbose=True
    )
    best_solution, best_fitness, history = tlbo.optimize()
    algorithm_times['TLBO'] = time.time() - start_time
    algorithm_results['TLBO'] = (best_solution, best_fitness, history)
    
    # # 测试TLBO+
    # print("\n测试TLBO+算法...")
    # start_time = time.time()
    # tlbo_plus = TLBOPlus(
    #     system_model=system,
    #     delay_model=delay_model,
    #     energy_model=energy_model,
    #     aoi_model=aoi_model,
    #     max_iter=max_iter,
    #     population_size=population_size,
    #     w_energy=0.5,
    #     w_delay=0.3,
    #     w_aoi=0.2,
    #     verbose=True
    # )
    # best_solution_plus, best_fitness_plus, history_plus = tlbo_plus.optimize()
    # algorithm_times['TLBO+'] = time.time() - start_time
    # algorithm_results['TLBO+'] = (best_solution_plus, best_fitness_plus, history_plus)
    
    # 测试GA
    print("\n测试GA算法...")
    start_time = time.time()
    ga = GA(
        system_model=system,
        delay_model=delay_model,
        energy_model=energy_model,
        aoi_model=aoi_model,
        max_iter=max_iter,
        population_size=population_size,
        w_energy=0.5,
        w_delay=0.3,
        w_aoi=0.2,
        verbose=True
    )
    best_solution_ga, best_fitness_ga, history_ga = ga.optimize()
    algorithm_times['GA'] = time.time() - start_time
    algorithm_results['GA'] = (best_solution_ga, best_fitness_ga, history_ga)
    
    # 测试GWO
    print("\n测试GWO算法...")
    start_time = time.time()
    gwo = GWO(
        system_model=system,
        delay_model=delay_model,
        energy_model=energy_model,
        aoi_model=aoi_model,
        max_iter=max_iter,
        population_size=population_size,
        w_energy=0.5,
        w_delay=0.3,
        w_aoi=0.2,
        verbose=True
    )
    best_solution_gwo, best_fitness_gwo, history_gwo = gwo.optimize()
    algorithm_times['GWO'] = time.time() - start_time
    algorithm_results['GWO'] = (best_solution_gwo, best_fitness_gwo, history_gwo)
    
    # 打印结果比较
    print("\n=== 算法性能比较 ===")
    print("算法\t\t最优适应度\t运行时间(s)")
    print("-" * 50)
    for name, (_, fitness, _) in algorithm_results.items():
        print(f"{name}\t\t{fitness:.6f}\t{algorithm_times[name]:.2f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    for name, (_, _, history) in algorithm_results.items():
        plt.plot(history, label=name)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.title('算法收敛曲线比较')
    plt.legend()
    plt.grid(True)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/algorithm_comparison.png', dpi=300)
    plt.show()


def test_solution_quality():
    """测试解的质量"""
    print("\n=== 测试解的质量 ===")
    
    # 创建系统
    system = create_simple_system()
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 优化解决方案
    # tlbo_plus = TLBOPlus(
    #     system_model=system,
    #     delay_model=delay_model,
    #     energy_model=energy_model,
    #     aoi_model=aoi_model,
    #     max_iter=50,
    #     population_size=20,
    #     w_energy=0.4,
    #     w_delay=0.3,
    #     w_aoi=0.3,
    #     verbose=False
    # )
    tlbo = TLBO(
        system_model=system,
        delay_model=delay_model,
        energy_model=energy_model,
        aoi_model=aoi_model,
        max_iter=50,
        population_size=20,
        w_energy=0.5,
        w_delay=0.3,
        w_aoi=0.2,
        verbose=False
    )
    best_solution, best_fitness, _ = tlbo.optimize()
    
    # 应用最优解
    system.apply_solution(best_solution)
    task = system.tasks[0]
    
    # 打印最优解
    print(f"最优解：{best_solution}")
    print(f"执行位置：{task.execution_location}")
    print(f"执行节点ID：{task.execution_node_id}")
    print(f"分配的计算资源：{task.allocated_resource/1e9:.2f} GHz")
    
    if task.update_interval is not None:
        print(f"更新间隔：{task.update_interval:.2f} s")
    
    # 计算性能指标
    delay = delay_model.calculate_total_delay(task)
    energy = energy_model.calculate_total_energy(task)
    aoi = aoi_model.calculate_average_aoi(task)
    
    print(f"\n延迟：{delay*1000:.2f} ms")
    print(f"能耗：{energy:.6f} J")
    print(f"AoI：{aoi:.2f} s")


if __name__ == "__main__":
    # 测试模型实现
    test_models()
    
    # 测试优化算法
    test_algorithms()
    
    # 测试解的质量
    test_solution_quality()