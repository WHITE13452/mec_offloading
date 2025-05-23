# src/experiments/realistic_system_setup.py
import os
import numpy as np
from ..models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
import matplotlib.pyplot as plt
from src.utils.plotting_utils import init_plotting_style
init_plotting_style()


def create_realistic_edge_computing_system(num_devices=15, num_edge_servers=4, num_cloud_servers=2, num_tasks=30):
    """创建现实的边缘计算系统，让卸载具有明显优势"""
    system = SystemModel()
    
    # ====== 1. 设计异构终端设备 ======
    # 大部分设备性能较弱，电池受限
    
    # 低端IoT设备 (40%) - 传感器、监控设备等
    low_end_count = int(num_devices * 0.4)
    for i in range(low_end_count):
        device = Device(
            device_id=i,
            max_cpu_frequency=np.random.uniform(0.2e9, 0.5e9),  # 200-500 MHz - 很低
            energy_coefficient=np.random.uniform(2.5e-27, 3.5e-27),  # 高能耗系数
            transmission_power=np.random.uniform(0.1, 0.2)  # 低传输功率
        )
        system.add_device(device)
    
    # 中端移动设备 (40%) - 智能手机、平板等
    mid_range_count = int(num_devices * 0.4)
    for i in range(low_end_count, low_end_count + mid_range_count):
        device = Device(
            device_id=i,
            max_cpu_frequency=np.random.uniform(0.8e9, 1.5e9),  # 0.8-1.5 GHz - 中等
            energy_coefficient=np.random.uniform(1.5e-27, 2.5e-27),  # 中等能耗系数
            transmission_power=np.random.uniform(0.2, 0.4)  # 中等传输功率
        )
        system.add_device(device)
    
    # 高端设备 (20%) - 高性能移动设备、笔记本等
    high_end_count = num_devices - low_end_count - mid_range_count
    for i in range(low_end_count + mid_range_count, num_devices):
        device = Device(
            device_id=i,
            max_cpu_frequency=np.random.uniform(2.0e9, 3.0e9),  # 2-3 GHz - 较高
            energy_coefficient=np.random.uniform(1.0e-27, 1.8e-27),  # 较低能耗系数
            transmission_power=np.random.uniform(0.3, 0.5)  # 较高传输功率
        )
        system.add_device(device)
    
    # ====== 2. 设计高性能边缘服务器 ======
    # 边缘服务器性能显著优于终端设备
    for i in range(num_edge_servers):
        server = EdgeServer(
            server_id=i,
            max_cpu_frequency=np.random.uniform(4.0e9, 6.0e9),  # 4-6 GHz - 高性能
            energy_coefficient=np.random.uniform(0.8e-27, 1.2e-27),  # 低能耗系数（更高效）
            transmission_power=np.random.uniform(1.0, 1.5)  # 高传输功率
        )
        system.add_edge_server(server)
    
    # ====== 3. 设计超高性能云服务器 ======
    for i in range(num_cloud_servers):
        server = CloudServer(
            server_id=i,
            max_cpu_frequency=np.random.uniform(8.0e9, 12.0e9),  # 8-12 GHz - 超高性能
            energy_coefficient=np.random.uniform(0.5e-27, 0.8e-27)  # 最低能耗系数（最高效）
        )
        system.add_cloud_server(server)
    
    # ====== 4. 设计多样化的任务 ======
    # 创建不同类型的任务，让某些任务必须卸载才能有效处理
    
    # 计算密集型任务 (30%) - 图像处理、机器学习推理等
    compute_intensive_count = int(num_tasks * 0.3)
    for i in range(compute_intensive_count):
        source_device_id = np.random.randint(0, num_devices)
        task = Task(
            task_id=i,
            data_size=np.random.uniform(1.0e6, 3.0e6),  # 1-3 MB - 中等数据量
            computation_complexity=np.random.uniform(500, 1000),  # 500-1000 cycles/bit - 高计算复杂度
            arrival_rate=np.random.uniform(0.02, 0.05),  # 较低到达率
            priority=np.random.uniform(1.5, 2.0),  # 高优先级
            max_delay=np.random.uniform(2.0, 4.0),  # 2-4秒 - 相对宽松的延迟要求
            source_device_id=source_device_id
        )
        system.add_task(task)
    
    # 数据密集型任务 (25%) - 视频处理、大数据分析等
    data_intensive_count = int(num_tasks * 0.25)
    for i in range(compute_intensive_count, compute_intensive_count + data_intensive_count):
        source_device_id = np.random.randint(0, num_devices)
        task = Task(
            task_id=i,
            data_size=np.random.uniform(5.0e6, 15.0e6),  # 5-15 MB - 大数据量
            computation_complexity=np.random.uniform(200, 400),  # 200-400 cycles/bit - 中等计算复杂度
            arrival_rate=np.random.uniform(0.01, 0.03),  # 低到达率
            priority=np.random.uniform(1.0, 1.5),  # 中等优先级
            max_delay=np.random.uniform(3.0, 6.0),  # 3-6秒 - 宽松的延迟要求
            source_device_id=source_device_id
        )
        system.add_task(task)
    
    # 实时敏感型任务 (25%) - 游戏、AR/VR、实时控制等
    realtime_count = int(num_tasks * 0.25)
    for i in range(compute_intensive_count + data_intensive_count, 
                   compute_intensive_count + data_intensive_count + realtime_count):
        source_device_id = np.random.randint(0, num_devices)
        task = Task(
            task_id=i,
            data_size=np.random.uniform(0.2e6, 1.0e6),  # 0.2-1 MB - 小数据量
            computation_complexity=np.random.uniform(150, 350),  # 150-350 cycles/bit - 中等计算复杂度
            arrival_rate=np.random.uniform(0.1, 0.2),  # 高到达率
            priority=np.random.uniform(1.8, 2.0),  # 高优先级
            max_delay=np.random.uniform(0.1, 0.5),  # 0.1-0.5秒 - 严格的延迟要求
            source_device_id=source_device_id
        )
        system.add_task(task)
    
    # 轻量级任务 (20%) - 简单查询、状态更新等
    lightweight_count = num_tasks - compute_intensive_count - data_intensive_count - realtime_count
    for i in range(compute_intensive_count + data_intensive_count + realtime_count, num_tasks):
        source_device_id = np.random.randint(0, num_devices)
        task = Task(
            task_id=i,
            data_size=np.random.uniform(0.05e6, 0.3e6),  # 50-300 KB - 很小数据量
            computation_complexity=np.random.uniform(20, 80),  # 20-80 cycles/bit - 低计算复杂度
            arrival_rate=np.random.uniform(0.2, 0.4),  # 很高到达率
            priority=np.random.uniform(0.5, 1.0),  # 低优先级
            max_delay=np.random.uniform(0.5, 1.5),  # 0.5-1.5秒 - 中等延迟要求
            source_device_id=source_device_id
        )
        system.add_task(task)
    
    # 将任务添加到相应设备的任务列表
    for task in system.tasks:
        device = system.get_device_by_id(task.source_device_id)
        if device:
            device.tasks.append(task)
    
    # ====== 5. 设计现实的网络环境 ======
    
    # 设备到边缘服务器的连接 - 考虑距离和设备类型
    for device in system.devices:
        for edge_server in system.edge_servers:
            # 根据设备类型设置不同的传输速率
            if device.device_id < low_end_count:  # 低端设备
                rate = np.random.uniform(2e6, 5e6)  # 2-5 Mbps - 较低
                bandwidth = np.random.uniform(5e6, 10e6)  # 5-10 MHz
            elif device.device_id < low_end_count + mid_range_count:  # 中端设备
                rate = np.random.uniform(8e6, 20e6)  # 8-20 Mbps - 中等
                bandwidth = np.random.uniform(15e6, 30e6)  # 15-30 MHz
            else:  # 高端设备
                rate = np.random.uniform(25e6, 50e6)  # 25-50 Mbps - 较高
                bandwidth = np.random.uniform(30e6, 50e6)  # 30-50 MHz
            
            system.set_device_to_edge_rate(device.device_id, edge_server.server_id, rate, bandwidth)
    
    # 边缘服务器到云服务器的连接 - 高速光纤连接
    for edge_server in system.edge_servers:
        for cloud_server in system.cloud_servers:
            rate = np.random.uniform(100e6, 500e6)  # 100-500 Mbps - 很高
            bandwidth = np.random.uniform(200e6, 500e6)  # 200-500 MHz
            system.set_edge_to_cloud_rate(edge_server.server_id, cloud_server.server_id, rate, bandwidth)
    
    # ====== 6. 设置现实的任务到达率 ======
    
    # 设备的任务到达率 - 考虑设备处理能力
    for device in system.devices:
        base_rate = 0.05 if device.device_id < low_end_count else (0.08 if device.device_id < low_end_count + mid_range_count else 0.12)
        for task in system.tasks:
            device.arrival_rates[task.task_id] = np.random.uniform(base_rate, base_rate + 0.03)
    
    # 边缘服务器的任务到达率 - 较低，因为它们主要处理卸载的任务
    for edge_server in system.edge_servers:
        for task in system.tasks:
            edge_server.arrival_rates[task.task_id] = np.random.uniform(0.01, 0.04)
    
    # 云服务器的任务到达率 - 最低，主要处理重度计算任务
    for cloud_server in system.cloud_servers:
        for task in system.tasks:
            cloud_server.arrival_rates[task.task_id] = np.random.uniform(0.005, 0.02)
    
    return system


def create_battery_constrained_scenario(base_system):
    """创建电池受限场景 - 强化卸载的必要性"""
    system = base_system
    
    # 进一步降低设备的处理能力，增加能耗系数
    for device in system.devices:
        device.max_cpu_frequency *= 0.6  # 降低60%的处理能力
        device.energy_coefficient *= 2.0  # 增加100%的能耗系数
    
    # 为任务添加电池消耗约束
    total_battery_capacity = 10000  # 假设总电池容量为10000 mAh
    
    for device in system.devices:
        device.battery_capacity = total_battery_capacity
        device.current_battery = total_battery_capacity * np.random.uniform(0.3, 0.8)  # 30%-80%的电量
    
    return system


def create_network_optimized_scenario(base_system):
    """创建网络优化场景 - 良好的网络条件促进卸载"""
    system = base_system
    
    # 提升网络传输速率
    for key in system.device_to_edge_rates:
        system.device_to_edge_rates[key] *= 2.0  # 提升2倍
    
    for key in system.edge_to_cloud_rates:
        system.edge_to_cloud_rates[key] *= 1.5  # 提升1.5倍
    
    # 降低传输功率消耗
    for device in system.devices:
        device.transmission_power *= 0.5  # 降低50%的传输功耗
    
    for edge_server in system.edge_servers:
        edge_server.transmission_power *= 0.7  # 降低30%的传输功耗
    
    return system


def create_computation_heavy_scenario(base_system):
    """创建计算密集型场景 - 重计算任务必须卸载"""
    system = base_system
    
    # 增加所有任务的计算复杂度
    for task in system.tasks:
        if task.task_id < len(system.tasks) * 0.5:  # 50%的任务变为超重计算任务
            task.computation_complexity *= 3.0  # 增加3倍计算复杂度
            task.max_delay *= 1.5  # 稍微放宽延迟要求
    
    return system


def analyze_task_execution_feasibility(system):
    """分析任务在不同位置执行的可行性"""
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    
    analysis_results = []
    
    for task in system.tasks:
        task_analysis = {
            'task_id': task.task_id,
            'task_type': get_task_type(task),
            'execution_options': {}
        }
        
        # 分析在不同位置执行的情况
        locations = ['device', 'edge', 'cloud']
        
        for location in locations:
            # 临时设置任务执行位置
            if location == 'device':
                task.execution_location = 'device'
                task.execution_node_id = task.source_device_id
                # 获取源设备
                device = system.get_device_by_id(task.source_device_id)
                task.allocated_resource = device.max_cpu_frequency * 0.8  # 使用80%的CPU
            elif location == 'edge':
                task.execution_location = 'edge'
                task.execution_node_id = system.edge_servers[0].server_id  # 使用第一个边缘服务器
                task.allocated_resource = system.edge_servers[0].max_cpu_frequency * 0.6  # 使用60%的CPU
            else:  # cloud
                task.execution_location = 'cloud'
                task.execution_node_id = system.cloud_servers[0].server_id  # 使用第一个云服务器
                task.allocated_resource = system.cloud_servers[0].max_cpu_frequency * 0.4  # 使用40%的CPU
            
            try:
                # 计算延迟和能耗
                delay = delay_model.calculate_total_delay(task)
                energy = energy_model.calculate_total_energy(task)
                
                # 检查是否满足延迟约束
                meets_deadline = delay <= task.max_delay
                
                task_analysis['execution_options'][location] = {
                    'delay': delay,
                    'energy': energy,
                    'meets_deadline': meets_deadline,
                    'feasible': meets_deadline
                }
                
            except Exception as e:
                task_analysis['execution_options'][location] = {
                    'delay': float('inf'),
                    'energy': float('inf'),
                    'meets_deadline': False,
                    'feasible': False,
                    'error': str(e)
                }
        
        analysis_results.append(task_analysis)
    
    return analysis_results


def get_task_type(task):
    """根据任务特征确定任务类型"""
    if task.computation_complexity > 500:
        return 'compute_intensive'
    elif task.data_size > 5e6:
        return 'data_intensive'
    elif task.max_delay < 0.6:
        return 'realtime_sensitive'
    else:
        return 'lightweight'


def print_feasibility_analysis(analysis_results):
    """打印可行性分析结果"""
    print("="*80)
    print("任务执行可行性分析")
    print("="*80)
    
    # 统计不同类型任务的执行情况
    task_type_stats = {}
    
    for task_analysis in analysis_results:
        task_type = task_analysis['task_type']
        if task_type not in task_type_stats:
            task_type_stats[task_type] = {
                'total': 0,
                'device_feasible': 0,
                'edge_feasible': 0,
                'cloud_feasible': 0,
                'only_offload_feasible': 0  # 只有卸载可行，本地不可行
            }
        
        stats = task_type_stats[task_type]
        stats['total'] += 1
        
        # 检查可行性
        device_feasible = task_analysis['execution_options']['device']['feasible']
        edge_feasible = task_analysis['execution_options']['edge']['feasible']
        cloud_feasible = task_analysis['execution_options']['cloud']['feasible']
        
        if device_feasible:
            stats['device_feasible'] += 1
        if edge_feasible:
            stats['edge_feasible'] += 1
        if cloud_feasible:
            stats['cloud_feasible'] += 1
        
        # 只有卸载可行的情况
        if not device_feasible and (edge_feasible or cloud_feasible):
            stats['only_offload_feasible'] += 1
    
    # 打印统计结果
    for task_type, stats in task_type_stats.items():
        print(f"\n{task_type.upper()} 任务 (总数: {stats['total']}):")
        print(f"  设备执行可行: {stats['device_feasible']}/{stats['total']} ({stats['device_feasible']/stats['total']*100:.1f}%)")
        print(f"  边缘执行可行: {stats['edge_feasible']}/{stats['total']} ({stats['edge_feasible']/stats['total']*100:.1f}%)")
        print(f"  云端执行可行: {stats['cloud_feasible']}/{stats['total']} ({stats['cloud_feasible']/stats['total']*100:.1f}%)")
        print(f"  仅卸载可行: {stats['only_offload_feasible']}/{stats['total']} ({stats['only_offload_feasible']/stats['total']*100:.1f}%)")
    
    # 显示一些具体的任务示例
    print(f"\n{'='*80}")
    print("具体任务示例:")
    print(f"{'='*80}")
    
    for i, task_analysis in enumerate(analysis_results[:5]):  # 只显示前5个任务
        print(f"\n任务 {task_analysis['task_id']} ({task_analysis['task_type']}):")
        for location, result in task_analysis['execution_options'].items():
            if result['feasible']:
                status = "✓ 可行"
            else:
                status = "✗ 不可行"
            print(f"  {location:6}: {status:8} - 延迟: {result['delay']:.3f}s, 能耗: {result['energy']:.3f}J")


def plot_task_characteristics(system, save_path='results/task_characteristics.png'):
    """绘制任务特征分布图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 提取任务数据
    task_types = [get_task_type(task) for task in system.tasks]
    data_sizes = [task.data_size / 1e6 for task in system.tasks]  # 转换为MB
    complexities = [task.computation_complexity for task in system.tasks]
    max_delays = [task.max_delay for task in system.tasks]
    
    # 1. 任务类型分布
    type_counts = {}
    for t_type in task_types:
        type_counts[t_type] = type_counts.get(t_type, 0) + 1
    
    ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax1.set_title('任务类型分布')
    
    # 2. 数据大小分布
    ax2.hist(data_sizes, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('数据大小 (MB)')
    ax2.set_ylabel('任务数量')
    ax2.set_title('任务数据大小分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 计算复杂度分布
    ax3.hist(complexities, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('计算复杂度 (cycles/bit)')
    ax3.set_ylabel('任务数量')
    ax3.set_title('任务计算复杂度分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 最大延迟分布
    ax4.hist(max_delays, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.set_xlabel('最大延迟要求 (s)')
    ax4.set_ylabel('任务数量')
    ax4.set_title('任务最大延迟要求分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_device_capabilities(system, save_path='results/device_capabilities.png'):
    """绘制设备能力分布图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 提取设备数据
    device_freqs = [device.max_cpu_frequency / 1e9 for device in system.devices]  # 转换为GHz
    device_energy_coeffs = [device.energy_coefficient * 1e27 for device in system.devices]  # 转换为合适单位
    device_tx_powers = [device.transmission_power for device in system.devices]
    
    # 提取边缘服务器数据
    edge_freqs = [server.max_cpu_frequency / 1e9 for server in system.edge_servers]
    edge_energy_coeffs = [server.energy_coefficient * 1e27 for server in system.edge_servers]
    
    # 提取云服务器数据
    cloud_freqs = [server.max_cpu_frequency / 1e9 for server in system.cloud_servers]
    cloud_energy_coeffs = [server.energy_coefficient * 1e27 for server in system.cloud_servers]
    
    # 1. CPU频率比较
    ax1.hist([device_freqs, edge_freqs, cloud_freqs], bins=10, alpha=0.7, 
             label=['设备', '边缘服务器', '云服务器'], color=['lightblue', 'lightgreen', 'lightcoral'])
    ax1.set_xlabel('CPU频率 (GHz)')
    ax1.set_ylabel('数量')
    ax1.set_title('不同节点CPU频率分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 能耗系数比较
    ax2.hist([device_energy_coeffs, edge_energy_coeffs, cloud_energy_coeffs], bins=10, alpha=0.7,
             label=['设备', '边缘服务器', '云服务器'], color=['lightblue', 'lightgreen', 'lightcoral'])
    ax2.set_xlabel('能耗系数 (×10^-27)')
    ax2.set_ylabel('数量')
    ax2.set_title('不同节点能耗系数分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 设备传输功率分布
    ax3.hist(device_tx_powers, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('传输功率 (W)')
    ax3.set_ylabel('设备数量')
    ax3.set_title('设备传输功率分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 计算能力对比
    all_nodes = ['设备'] * len(device_freqs) + ['边缘'] * len(edge_freqs) + ['云端'] * len(cloud_freqs)
    all_freqs = device_freqs + edge_freqs + cloud_freqs
    
    # 箱线图显示计算能力差异
    device_data = [device_freqs, edge_freqs, cloud_freqs]
    ax4.boxplot(device_data, labels=['设备', '边缘服务器', '云服务器'])
    ax4.set_ylabel('CPU频率 (GHz)')
    ax4.set_title('节点计算能力对比')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 使用示例和测试函数
def main():
    """测试新的系统设计"""
    # 创建基础系统
    print("创建现实边缘计算系统...")
    system = create_realistic_edge_computing_system()
    
    # 绘制系统特征
    plot_task_characteristics(system)
    plot_device_capabilities(system)
    
    # 分析任务执行可行性
    print("分析任务执行可行性...")
    analysis_results = analyze_task_execution_feasibility(system)
    print_feasibility_analysis(analysis_results)
    
    # 测试不同场景
    print("\n" + "="*80)
    print("测试电池受限场景...")
    battery_system = create_battery_constrained_scenario(system)
    battery_analysis = analyze_task_execution_feasibility(battery_system)
    print_feasibility_analysis(battery_analysis)
    
    print("\n" + "="*80)
    print("测试计算密集型场景...")
    compute_system = create_computation_heavy_scenario(system)
    compute_analysis = analyze_task_execution_feasibility(compute_system)
    print_feasibility_analysis(compute_analysis)


if __name__ == "__main__":
    main()


# 继续 src/experiments/realistic_system_setup.py

def create_dynamic_workload_scenario(base_system, time_slots=24):
    """创建动态工作负载场景 - 模拟一天中不同时段的负载变化"""
    system = base_system
    
    # 为系统添加时间维度
    system.time_slots = time_slots
    system.current_time_slot = 0
    
    # 定义一天中的负载模式
    # 早高峰 (7-9点)，午高峰 (12-14点)，晚高峰 (18-21点)
    load_patterns = {
        'compute_intensive': [0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.9, 1.2, 1.0, 0.8, 0.7, 0.8,  # 0-11点
                             1.1, 1.3, 1.0, 0.9, 0.8, 0.9, 1.4, 1.6, 1.5, 1.2, 0.9, 0.7], # 12-23点
        'data_intensive': [0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.6, 0.7,    # 0-11点
                          1.2, 1.4, 1.1, 0.8, 0.7, 0.8, 1.0, 1.2, 1.3, 1.1, 0.8, 0.5],   # 12-23点
        'realtime_sensitive': [0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 1.3, 1.0, 0.9, 1.0, # 0-11点
                              1.2, 1.1, 0.9, 0.8, 0.9, 1.3, 1.8, 2.0, 1.7, 1.4, 1.0, 0.6], # 12-23点
        'lightweight': [0.8, 0.6, 0.4, 0.4, 0.6, 0.9, 1.3, 1.5, 1.4, 1.2, 1.1, 1.3,       # 0-11点
                       1.5, 1.4, 1.2, 1.1, 1.2, 1.6, 1.9, 1.8, 1.6, 1.4, 1.2, 1.0]        # 12-23点
    }
    
    # 为每个任务分配时变的到达率
    for task in system.tasks:
        task_type = get_task_type(task)
        task.hourly_arrival_rates = []
        
        base_rate = task.arrival_rate
        for hour in range(24):
            multiplier = load_patterns.get(task_type, [1.0] * 24)[hour]
            task.hourly_arrival_rates.append(base_rate * multiplier)
    
    return system


def create_resource_contention_scenario(base_system):
    """创建资源竞争场景 - 模拟多用户竞争有限资源"""
    system = base_system
    
    # 为边缘服务器和云服务器添加资源限制
    for edge_server in system.edge_servers:
        # 边缘服务器总CPU资源有限
        edge_server.total_cpu_capacity = edge_server.max_cpu_frequency
        edge_server.available_cpu_capacity = edge_server.total_cpu_capacity
        edge_server.max_concurrent_tasks = 5  # 最多同时处理5个任务
        edge_server.current_task_count = 0
        
        # 内存限制
        edge_server.total_memory = np.random.uniform(8e9, 16e9)  # 8-16 GB
        edge_server.available_memory = edge_server.total_memory
    
    for cloud_server in system.cloud_servers:
        # 云服务器资源更充足但也有限制
        cloud_server.total_cpu_capacity = cloud_server.max_cpu_frequency
        cloud_server.available_cpu_capacity = cloud_server.total_cpu_capacity
        cloud_server.max_concurrent_tasks = 20  # 最多同时处理20个任务
        cloud_server.current_task_count = 0
        
        # 内存限制
        cloud_server.total_memory = np.random.uniform(32e9, 64e9)  # 32-64 GB
        cloud_server.available_memory = cloud_server.total_memory
    
    # 为任务添加内存需求
    for task in system.tasks:
        task_type = get_task_type(task)
        if task_type == 'compute_intensive':
            task.memory_requirement = np.random.uniform(1e9, 4e9)  # 1-4 GB
        elif task_type == 'data_intensive':
            task.memory_requirement = np.random.uniform(2e9, 8e9)  # 2-8 GB
        elif task_type == 'realtime_sensitive':
            task.memory_requirement = np.random.uniform(0.5e9, 2e9)  # 0.5-2 GB
        else:  # lightweight
            task.memory_requirement = np.random.uniform(0.1e9, 0.5e9)  # 0.1-0.5 GB
    
    return system


def create_network_congestion_scenario(base_system, congestion_factor=0.3):
    """创建网络拥塞场景 - 降低网络传输速率模拟拥塞"""
    system = base_system
    
    # 随机选择一些链路进行拥塞
    device_edge_links = list(system.device_to_edge_rates.keys())
    edge_cloud_links = list(system.edge_to_cloud_rates.keys())
    
    # 选择拥塞的链路
    num_congested_de_links = int(len(device_edge_links) * congestion_factor)
    num_congested_ec_links = int(len(edge_cloud_links) * congestion_factor)
    
    congested_de_links = np.random.choice(len(device_edge_links), num_congested_de_links, replace=False)
    congested_ec_links = np.random.choice(len(edge_cloud_links), num_congested_ec_links, replace=False)
    
    # 应用拥塞 - 降低传输速率
    for idx in congested_de_links:
        link = device_edge_links[idx]
        system.device_to_edge_rates[link] *= np.random.uniform(0.2, 0.5)  # 降低50-80%
    
    for idx in congested_ec_links:
        link = edge_cloud_links[idx]
        system.edge_to_cloud_rates[link] *= np.random.uniform(0.3, 0.6)  # 降低40-70%
    
    return system


def enhanced_feasibility_analysis(system, scenarios=None):
    """增强的可行性分析 - 包含多种场景"""
    if scenarios is None:
        scenarios = ['base']
    
    all_results = {}
    
    for scenario_name in scenarios:
        print(f"\n{'='*80}")
        print(f"分析场景: {scenario_name.upper()}")
        print(f"{'='*80}")
        
        if scenario_name == 'base':
            test_system = system
        elif scenario_name == 'battery_constrained':
            test_system = create_battery_constrained_scenario(system)
        elif scenario_name == 'network_optimized':
            test_system = create_network_optimized_scenario(system)
        elif scenario_name == 'computation_heavy':
            test_system = create_computation_heavy_scenario(system)
        elif scenario_name == 'resource_contention':
            test_system = create_resource_contention_scenario(system)
        elif scenario_name == 'network_congestion':
            test_system = create_network_congestion_scenario(system)
        else:
            test_system = system
        
        # 分析任务执行可行性
        analysis_results = analyze_task_execution_feasibility(test_system)
        all_results[scenario_name] = analysis_results
        
        # 打印分析结果
        print_feasibility_analysis(analysis_results)
        
        # 计算关键统计信息
        offload_necessity = calculate_offload_necessity(analysis_results)
        print(f"\n卸载必要性分析:")
        print(f"  必须卸载的任务: {offload_necessity['must_offload']}/{offload_necessity['total']} ({offload_necessity['must_offload_rate']:.1f}%)")
        print(f"  本地不可行的任务: {offload_necessity['local_infeasible']}/{offload_necessity['total']} ({offload_necessity['local_infeasible_rate']:.1f}%)")
        print(f"  卸载更优的任务: {offload_necessity['offload_better']}/{offload_necessity['total']} ({offload_necessity['offload_better_rate']:.1f}%)")
    
    return all_results


def calculate_offload_necessity(analysis_results):
    """计算卸载必要性统计"""
    total_tasks = len(analysis_results)
    must_offload = 0  # 必须卸载（本地不可行）
    local_infeasible = 0  # 本地不可行
    offload_better = 0  # 卸载更优（在满足约束条件下能耗或延迟更低）
    
    for task_analysis in analysis_results:
        device_option = task_analysis['execution_options']['device']
        edge_option = task_analysis['execution_options']['edge']
        cloud_option = task_analysis['execution_options']['cloud']
        
        # 本地不可行
        if not device_option['feasible']:
            local_infeasible += 1
            
            # 边缘或云可行，则必须卸载
            if edge_option['feasible'] or cloud_option['feasible']:
                must_offload += 1
        
        # 本地可行但卸载更优
        elif device_option['feasible']:
            device_cost = device_option['energy'] + device_option['delay']
            
            if edge_option['feasible']:
                edge_cost = edge_option['energy'] + edge_option['delay']
                if edge_cost < device_cost * 0.8:  # 卸载成本低20%以上
                    offload_better += 1
                    continue
            
            if cloud_option['feasible']:
                cloud_cost = cloud_option['energy'] + cloud_option['delay']
                if cloud_cost < device_cost * 0.8:  # 卸载成本低20%以上
                    offload_better += 1
    
    return {
        'total': total_tasks,
        'must_offload': must_offload,
        'local_infeasible': local_infeasible,
        'offload_better': offload_better,
        'must_offload_rate': (must_offload / total_tasks) * 100,
        'local_infeasible_rate': (local_infeasible / total_tasks) * 100,
        'offload_better_rate': (offload_better / total_tasks) * 100
    }


def plot_scenario_comparison(all_results, save_path='results/scenario_comparison.png'):
    """绘制不同场景下的任务分配可行性对比"""
    scenarios = list(all_results.keys())
    metrics = ['local_infeasible_rate', 'must_offload_rate', 'offload_better_rate']
    metric_labels = ['本地不可行率 (%)', '必须卸载率 (%)', '卸载更优率 (%)']
    
    # 计算每个场景的统计数据
    scenario_stats = {}
    for scenario, results in all_results.items():
        stats = calculate_offload_necessity(results)
        scenario_stats[scenario] = stats
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 柱状图对比
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [scenario_stats[scenario][metric] for scenario in scenarios]
        ax1.bar(x + i * width, values, width, label=metric_labels[i], alpha=0.8)
    
    ax1.set_xlabel('场景')
    ax1.set_ylabel('百分比 (%)')
    ax1.set_title('不同场景下的任务卸载必要性对比')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 饼图显示基础场景的任务分配情况
    base_stats = scenario_stats.get('base', scenario_stats[list(scenario_stats.keys())[0]])
    
    # 计算不同类型任务的数量
    feasible_local = base_stats['total'] - base_stats['local_infeasible']
    must_offload = base_stats['must_offload']
    prefer_offload = base_stats['offload_better']
    only_local = feasible_local - prefer_offload
    
    labels = ['只能本地执行', '本地更优', '卸载更优', '必须卸载']
    sizes = [only_local, feasible_local - prefer_offload - must_offload, prefer_offload, must_offload]
    colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
    
    # 过滤掉0值
    filtered_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
    if filtered_data:
        labels_f, sizes_f, colors_f = zip(*filtered_data)
        ax2.pie(sizes_f, labels=labels_f, autopct='%1.1f%%', colors=colors_f, startangle=90)
    
    ax2.set_title('基础场景任务执行偏好分布')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_experiment_config(system, config_name='realistic_config'):
    """生成实验配置文件，便于复现实验"""
    config = {
        'system_name': config_name,
        'devices': [],
        'edge_servers': [],
        'cloud_servers': [],
        'tasks': [],
        'device_to_edge_links': [],
        'edge_to_cloud_links': []
    }
    
    # 设备配置
    for device in system.devices:
        config['devices'].append({
            'id': device.device_id,
            'max_cpu_frequency': device.max_cpu_frequency,
            'energy_coefficient': device.energy_coefficient,
            'transmission_power': device.transmission_power
        })
    
    # 边缘服务器配置
    for server in system.edge_servers:
        config['edge_servers'].append({
            'id': server.server_id,
            'max_cpu_frequency': server.max_cpu_frequency,
            'energy_coefficient': server.energy_coefficient,
            'transmission_power': server.transmission_power
        })
    
    # 云服务器配置
    for server in system.cloud_servers:
        config['cloud_servers'].append({
            'id': server.server_id,
            'max_cpu_frequency': server.max_cpu_frequency,
            'energy_coefficient': server.energy_coefficient
        })
    
    # 任务配置
    for task in system.tasks:
        config['tasks'].append({
            'id': task.task_id,
            'data_size': task.data_size,
            'computation_complexity': task.computation_complexity,
            'arrival_rate': task.arrival_rate,
            'priority': task.priority,
            'max_delay': task.max_delay,
            'source_device_id': task.source_device_id
        })
    
    # 网络链路配置
    for (device_id, edge_id), rate in system.device_to_edge_rates.items():
        bandwidth = system.device_to_edge_bandwidth.get((device_id, edge_id), rate * 2)
        config['device_to_edge_links'].append({
            'device_id': device_id,
            'edge_id': edge_id,
            'rate': rate,
            'bandwidth': bandwidth
        })
    
    for (edge_id, cloud_id), rate in system.edge_to_cloud_rates.items():
        bandwidth = system.edge_to_cloud_bandwidth.get((edge_id, cloud_id), rate * 2)
        config['edge_to_cloud_links'].append({
            'edge_id': edge_id,
            'cloud_id': cloud_id,
            'rate': rate,
            'bandwidth': bandwidth
        })
    
    return config


def save_config_to_file(config, filename='configs/realistic_system_config.json'):
    """保存配置到JSON文件"""
    import json
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {filename}")


# 完整的测试和分析主函数
def comprehensive_analysis():
    """进行全面的系统分析"""
    print("="*100)
    print("全面边缘计算系统分析")
    print("="*100)
    
    # 1. 创建基础系统
    print("\n1. 创建现实边缘计算系统...")
    system = create_realistic_edge_computing_system(
        num_devices=20, 
        num_edge_servers=4, 
        num_cloud_servers=2, 
        num_tasks=40
    )
    
    # 2. 绘制系统特征
    print("\n2. 生成系统特征图...")
    plot_task_characteristics(system)
    plot_device_capabilities(system)
    
    # 3. 多场景分析
    print("\n3. 进行多场景可行性分析...")
    scenarios = [
        'base',
        'battery_constrained', 
        'network_optimized',
        'computation_heavy',
        'resource_contention',
        'network_congestion'
    ]
    
    all_results = enhanced_feasibility_analysis(system, scenarios)
    
    # 4. 绘制场景对比
    print("\n4. 生成场景对比图...")
    plot_scenario_comparison(all_results)
    
    # 5. 生成配置文件
    print("\n5. 生成实验配置文件...")
    config = generate_experiment_config(system)
    save_config_to_file(config)
    
    # 6. 推荐实验参数
    print("\n6. 实验参数推荐:")
    print("="*60)
    
    base_stats = calculate_offload_necessity(all_results['base'])
    
    print(f"基础系统统计:")
    print(f"  总任务数: {base_stats['total']}")
    print(f"  必须卸载任务: {base_stats['must_offload']} ({base_stats['must_offload_rate']:.1f}%)")
    print(f"  卸载更优任务: {base_stats['offload_better']} ({base_stats['offload_better_rate']:.1f}%)")
    
    print(f"\n推荐的算法权重配置:")
    print(f"  w_energy = 0.3-0.4 (能耗权重)")
    print(f"  w_delay = 0.6-0.7 (延迟权重，因为有严格延迟约束)")
    print(f"  hho_prob = 0.2-0.4 (HHO使用概率)")
    
    print(f"\n推荐的实验设置:")
    print(f"  max_iter = 150-200 (足够的迭代次数)")
    print(f"  population_size = 40-60 (平衡搜索能力和计算成本)")
    print(f"  多次运行取平均 = 10-15次 (确保结果稳定性)")
    
    return system, all_results


if __name__ == "__main__":
    comprehensive_analysis()