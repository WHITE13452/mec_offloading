# src/experiments/realistic_system_setup.py
import os
import numpy as np
from ..models.system_model import SystemModel, Device, EdgeServer, CloudServer, Task
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
import matplotlib.pyplot as plt
from src.utils.plotting_utils import init_plotting_style
init_plotting_style()

# ============ 添加统一的样式定义 ============
# 全局变量控制是否使用英文标签
USE_ENGLISH_LABELS = True

# 设置全局字体和字号 - 增大字号以匹配文档样式
FONT_SIZE_TITLE = 26    # 标题字体
FONT_SIZE_LABEL = 24    # 轴标签字体
FONT_SIZE_TICK = 22     # 刻度字体
FONT_SIZE_LEGEND = 22   # 图例字体
FONT_SIZE_TEXT = 22     # 文本标注字体

# 根据提供的RGB值设置颜色
COLORS = [
    (19/255, 33/255, 60/255),    # 深蓝色 R:019, G:033, B:060
    (252/255, 163/255, 17/255),  # 黄色 R:252, G:163, B:017
    (136/255, 179/255, 214/255), # 浅蓝色 R:136, G:179, B:214
    (200/255, 97/255, 52/255),   # 棕红色 R:200, G:097, B:052
]

# 不同的填充样式，适合黑白打印
HATCHES = ['/', '\\', 'x', '+', 'o', 'O', '.', '*']

# 标签字典
LABELS = {
    'zh': {
        'task_type_distribution': '任务类型分布',
        'task_data_size': '任务数据大小分布',
        'task_complexity': '任务计算复杂度分布',
        'task_delay_req': '任务延迟要求分布',
        'node_computing': '节点计算能力',
        'network_rates': '网络传输速率',
        'energy_params': '能耗参数',
    },
    'en': {
        'task_type_distribution': 'Task Type Distribution',
        'task_data_size': 'Task Data Size Distribution',
        'task_complexity': 'Task Compute Complexity Distribution',
        'task_delay_req': 'Task Delay Requirement Distribution',
        'node_computing': 'Node Computing Capability',
        'network_rates': 'Network Transmission Rates',
        'energy_params': 'Energy Consumption Parameters',
    }
}

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

def get_label(key):
    """获取标签文本"""
    lang = 'en' if USE_ENGLISH_LABELS else 'zh'
    return LABELS[lang].get(key, key)
# ============ 样式定义结束 ============


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
    """绘制任务特征图"""
    setup_plot_style()  # 应用Times New Roman字体
    
    fig = plt.figure(figsize=(16, 12))
    
    # 任务类型分布
    ax1 = plt.subplot(2, 2, 1)
    task_types = []
    for task in system.tasks:
        task_types.append(get_task_type(task))
    
    from collections import Counter
    type_counts = Counter(task_types)
    colors_pie = [COLORS[i % len(COLORS)] for i in range(len(type_counts))]
    wedges, texts, autotexts = ax1.pie(type_counts.values(), labels=type_counts.keys(), 
                                       autopct='%1.1f%%', colors=colors_pie,
                                       textprops={'fontsize': FONT_SIZE_TEXT-2})
    # 添加边框和图案
    for i, wedge in enumerate(wedges):
        wedge.set_edgecolor('black')
        wedge.set_linewidth(2)
        wedge.set_hatch(HATCHES[i % len(HATCHES)])
    
    ax1.set_title(get_label('task_type_distribution'), 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    # 任务数据量分布
    ax2 = plt.subplot(2, 2, 2)
    data_sizes = [task.data_size / 1024 / 1024 for task in system.tasks]  # 转换为MB
    ax2.hist(data_sizes, bins=15, alpha=0.7, color=COLORS[0], edgecolor='black', linewidth=2, hatch=HATCHES[0])
    ax2.set_xlabel('Data Size (MB)', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Number of Tasks', fontsize=FONT_SIZE_LABEL)
    ax2.set_title(get_label('task_data_size'), fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    ax2.grid(True, alpha=0.3)
    
    # 任务计算复杂度分布
    ax3 = plt.subplot(2, 2, 3)
    compute_complexities = [task.computation_complexity for task in system.tasks]  # cycles/bit
    ax3.hist(compute_complexities, bins=15, alpha=0.7, color=COLORS[1], edgecolor='black', linewidth=2, hatch=HATCHES[1])
    ax3.set_xlabel('Compute Complexity (cycles/bit)', fontsize=FONT_SIZE_LABEL)
    ax3.set_ylabel('Number of Tasks', fontsize=FONT_SIZE_LABEL)
    ax3.set_title(get_label('task_complexity'), fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax3.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    ax3.grid(True, alpha=0.3)
    
    # 任务延迟要求分布
    ax4 = plt.subplot(2, 2, 4)
    delay_requirements = [task.max_delay for task in system.tasks]
    ax4.hist(delay_requirements, bins=15, alpha=0.7, color=COLORS[2], edgecolor='black', linewidth=2, hatch=HATCHES[2])
    ax4.set_xlabel('Max Delay Requirement (s)', fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel('Number of Tasks', fontsize=FONT_SIZE_LABEL)
    ax4.set_title(get_label('task_delay_req'), fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax4.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"任务特征图已保存到: {save_path}")


def plot_device_capabilities(system, save_path='results/device_capabilities.png'):
    """绘制设备能力分布图"""
    setup_plot_style()  # 应用Times New Roman字体
    
    fig = plt.figure(figsize=(16, 8))
    
    # 设备频率对比
    ax1 = plt.subplot(1, 3, 1)
    device_freqs = [d.max_cpu_frequency / 1e9 for d in system.devices]
    edge_freqs = [e.max_cpu_frequency / 1e9 for e in system.edge_servers]
    cloud_freqs = [c.max_cpu_frequency / 1e9 for c in system.cloud_servers]
    
    data = [device_freqs, edge_freqs, cloud_freqs]
    positions = [1, 2, 3]
    bp1 = ax1.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    # 设置箱线图样式
    for i, (patch, pos) in enumerate(zip(bp1['boxes'], positions)):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
        patch.set_hatch(HATCHES[i % len(HATCHES)])
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp1[element]:
            item.set_color('black')
            item.set_linewidth(2 if element == 'medians' else 1.5)
    
    ax1.set_xticklabels(['Device', 'Edge', 'Cloud'], fontsize=FONT_SIZE_TICK)
    ax1.set_ylabel('CPU Frequency (GHz)', fontsize=FONT_SIZE_LABEL)
    ax1.set_title(get_label('node_computing'), fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax1.tick_params(axis='y', which='major', labelsize=FONT_SIZE_TICK)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 传输速率对比 - 使用平均值
    ax2 = plt.subplot(1, 3, 2)
    
    # 计算平均传输速率
    device_edge_rates = []
    for (device_id, edge_id), rate in system.device_to_edge_rates.items():
        device_edge_rates.append(rate / 1e6)  # 转换为 Mbps
    avg_device_edge_rate = np.mean(device_edge_rates) if device_edge_rates else 0
    
    edge_cloud_rates = []
    for (edge_id, cloud_id), rate in system.edge_to_cloud_rates.items():
        edge_cloud_rates.append(rate / 1e6)  # 转换为 Mbps
    avg_edge_cloud_rate = np.mean(edge_cloud_rates) if edge_cloud_rates else 0
    
    x = [1, 2]
    bars = ax2.bar(x, [avg_device_edge_rate, avg_edge_cloud_rate], 
                   color=[COLORS[0], COLORS[1]], alpha=0.8,
                   edgecolor='black', linewidth=2)
    
    # 添加图案
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCHES[i])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Device-Edge', 'Edge-Cloud'], fontsize=FONT_SIZE_TICK)
    ax2.set_ylabel('Transmission Rate (Mbps)', fontsize=FONT_SIZE_LABEL)
    ax2.set_title(get_label('network_rates'), fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax2.tick_params(axis='y', which='major', labelsize=FONT_SIZE_TICK)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{height:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=FONT_SIZE_TEXT-4)
    
    # 设备能耗参数对比
    ax3 = plt.subplot(1, 3, 3)
    device_powers = [d.energy_coefficient * 1e27 for d in system.devices]  # 转换为合适单位
    edge_powers = [e.energy_coefficient * 1e27 for e in system.edge_servers]
    cloud_powers = [c.energy_coefficient * 1e27 for c in system.cloud_servers]
    
    data = [device_powers, edge_powers, cloud_powers]
    bp2 = ax3.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    # 设置箱线图样式
    for i, (patch, pos) in enumerate(zip(bp2['boxes'], positions)):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
        patch.set_hatch(HATCHES[i % len(HATCHES)])
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp2[element]:
            item.set_color('black')
            item.set_linewidth(2 if element == 'medians' else 1.5)
    
    ax3.set_xticklabels(['Device', 'Edge', 'Cloud'], fontsize=FONT_SIZE_TICK)
    ax3.set_ylabel('Energy Coefficient (×10$^{-27}$)', fontsize=FONT_SIZE_LABEL)
    ax3.set_title(get_label('energy_params'), fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax3.tick_params(axis='y', which='major', labelsize=FONT_SIZE_TICK)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"设备能力图已保存到: {save_path}")


# 继续添加缺失的函数
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