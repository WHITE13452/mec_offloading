# src/example_usage.py
import json
from models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from models.delay_model import DelayModel
from models.energy_model import EnergyModel
from models.aoi_model import AoIModel

# 创建一个简单的系统模型
def create_example_system():
    system = SystemModel()
    
    # 添加设备
    device1 = Device(device_id=1, max_cpu_frequency=2.0e9, energy_coefficient=1e-27, transmission_power=0.5)
    system.add_device(device1)
    
    # 添加边缘服务器
    edge_server1 = EdgeServer(server_id=1, max_cpu_frequency=3.0e9, energy_coefficient=2e-27, transmission_power=1.0)
    system.add_edge_server(edge_server1)
    
    # 添加云服务器
    cloud_server1 = CloudServer(server_id=1, max_cpu_frequency=4.0e9, energy_coefficient=3e-27)
    system.add_cloud_server(cloud_server1)
    
    # 添加任务
    task1 = Task(
        task_id=1, 
        data_size=1e6,  # 1MB
        computation_complexity=100,  # 每比特100个CPU周期
        arrival_rate=0.1,  # 每秒0.1个任务
        priority=1.0,
        max_delay=1.0,  # 最大延迟1秒
        update_interval=0.5,  # 更新间隔0.5秒
        max_aoi=1.5,  # 最大可接受AoI 1.5秒
        source_device_id=1  # 设置源设备ID为device1
    )
    system.add_task(task1)
    device1.tasks.append(task1)  # 将任务添加到设备的任务列表
    
    # 设置网络参数
    system.set_device_to_edge_rate(device_id=1, edge_id=1, rate=10e6, bandwidth=20e6)  # 10Mbps, 20MHz
    system.set_edge_to_cloud_rate(edge_id=1, cloud_id=1, rate=100e6, bandwidth=200e6)  # 100Mbps, 200MHz
    
    # 设置任务到达率
    device1.arrival_rates[1] = 0.1  # 任务1在设备1上的到达率
    edge_server1.arrival_rates[1] = 0.05  # 任务1在边缘服务器1上的到达率
    cloud_server1.arrival_rates[1] = 0.01  # 任务1在云服务器1上的到达率
    
    return system

def main():
    # 创建系统模型
    system = create_example_system()
    
    # 创建模型
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    
    # 应用一个示例解决方案
    # 假设任务1在边缘服务器上执行，分配2.5GHz的CPU频率，更新间隔0.5秒
    solution = [[1, 2.5e9, 0.5]]  # loc_i = 1 (第一个边缘服务器), f_i = 2.5GHz, Δ_i = 0.5s
    system.apply_solution(solution)
    
    # 计算延迟
    task = system.get_task_by_id(1)
    total_delay = delay_model.calculate_total_delay(task)
    print(f"Task 1 total delay: {total_delay:.6f} seconds")
    
    # 计算能耗
    total_energy = energy_model.calculate_total_energy(task)
    print(f"Task 1 total energy: {total_energy:.6f} joules")
    
    # 计算AoI
    average_aoi = aoi_model.calculate_average_aoi(task)
    peak_aoi = aoi_model.calculate_peak_aoi(task)
    print(f"Task 1 average AoI: {average_aoi:.6f} seconds")
    print(f"Task 1 peak AoI: {peak_aoi:.6f} seconds")
    
    # 优化更新间隔
    optimal_interval = aoi_model.optimize_update_interval(task)
    print(f"Task 1 optimal update interval: {optimal_interval:.6f} seconds")
    print(f"Task 1 new average AoI: {task.aoi:.6f} seconds")
    
    # 编码当前解决方案
    encoded_solution = system.encode_solution()
    print(f"Encoded solution: {encoded_solution}")

if __name__ == "__main__":
    main()