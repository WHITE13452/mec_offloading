# src/models/system_model.py
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class Task:
    """表示计算任务的类"""
    
    def __init__(self, task_id: int, data_size: float, computation_complexity: float, 
                 arrival_rate: float, priority: float, max_delay: float, 
                 update_interval: Optional[float] = None, max_aoi: Optional[float] = None):
        """
        初始化一个计算任务
        
        Parameters:
        -----------
        task_id : int
            任务唯一标识符
        data_size : float
            任务数据大小（比特）
        computation_complexity : float
            计算复杂度（CPU周期/比特）
        arrival_rate : float
            任务到达率（任务/秒）
        priority : float
            任务优先级因子
        max_delay : float
            最大可接受延迟
        update_interval : float, optional
            任务更新间隔（仅用于第二部分，考虑AoI）
        max_aoi : float, optional
            最大可接受AoI（仅用于第二部分）
        """
        self.task_id = task_id
        self.data_size = data_size
        self.computation_complexity = computation_complexity
        self.arrival_rate = arrival_rate
        self.priority = priority
        self.max_delay = max_delay
        self.update_interval = update_interval
        self.max_aoi = max_aoi
        
        # 任务位置和资源分配的决策变量（由优化算法设置）
        self.execution_location = None  # 'device', 'edge', 或 'cloud'
        self.execution_node_id = None   # 执行节点的ID
        self.allocated_resource = None  # 分配的计算资源（CPU频率）
        
        # 任务性能指标（由模型计算）
        self.delay = None
        self.energy = None
        self.aoi = None


class Device:
    """表示终端设备的类"""
    
    def __init__(self, device_id: int, max_cpu_frequency: float, 
                 energy_coefficient: float, transmission_power: float):
        """
        初始化一个终端设备
        
        Parameters:
        -----------
        device_id : int
            设备唯一标识符
        max_cpu_frequency : float
            最大CPU频率（Hz）
        energy_coefficient : float
            能耗系数（用于计算能耗）
        transmission_power : float
            传输功率（W）
        """
        self.device_id = device_id
        self.max_cpu_frequency = max_cpu_frequency
        self.energy_coefficient = energy_coefficient
        self.transmission_power = transmission_power
        self.tasks = []  # 设备上的任务列表
        self.arrival_rates = {}  # 任务到达率字典 {task_id: arrival_rate}


class EdgeServer:
    """表示边缘服务器的类"""
    
    def __init__(self, server_id: int, max_cpu_frequency: float, 
                 energy_coefficient: float, transmission_power: float):
        """
        初始化一个边缘服务器
        
        Parameters:
        -----------
        server_id : int
            服务器唯一标识符
        max_cpu_frequency : float
            最大CPU频率（Hz）
        energy_coefficient : float
            能耗系数
        transmission_power : float
            传输功率（W）
        """
        self.server_id = server_id
        self.max_cpu_frequency = max_cpu_frequency
        self.energy_coefficient = energy_coefficient
        self.transmission_power = transmission_power
        self.tasks = []  # 服务器上的任务列表
        self.arrival_rates = {}  # 任务到达率字典


class CloudServer:
    """表示云服务器的类"""
    
    def __init__(self, server_id: int, max_cpu_frequency: float, 
                 energy_coefficient: float):
        """
        初始化一个云服务器
        
        Parameters:
        -----------
        server_id : int
            服务器唯一标识符
        max_cpu_frequency : float
            最大CPU频率（Hz）
        energy_coefficient : float
            能耗系数
        """
        self.server_id = server_id
        self.max_cpu_frequency = max_cpu_frequency
        self.energy_coefficient = energy_coefficient
        self.tasks = []  # 服务器上的任务列表
        self.arrival_rates = {}  # 任务到达率字典


class SystemModel:
    """系统整体模型，包含所有设备、服务器和任务"""
    
    def __init__(self):
        """初始化系统模型"""
        self.devices = []
        self.edge_servers = []
        self.cloud_servers = []
        self.tasks = []
        
        # 网络连接参数
        self.device_to_edge_rates = {}  # {(device_id, edge_id): rate}
        self.edge_to_cloud_rates = {}   # {(edge_id, cloud_id): rate}
        self.device_to_edge_bandwidth = {}  # {(device_id, edge_id): bandwidth}
        self.edge_to_cloud_bandwidth = {}   # {(edge_id, cloud_id): bandwidth}
    
    def add_device(self, device: Device) -> None:
        """添加设备到系统"""
        self.devices.append(device)
    
    def add_edge_server(self, server: EdgeServer) -> None:
        """添加边缘服务器到系统"""
        self.edge_servers.append(server)
    
    def add_cloud_server(self, server: CloudServer) -> None:
        """添加云服务器到系统"""
        self.cloud_servers.append(server)
    
    def add_task(self, task: Task) -> None:
        """添加任务到系统"""
        self.tasks.append(task)
    
    def set_device_to_edge_rate(self, device_id: int, edge_id: int, rate: float, bandwidth: float) -> None:
        """设置设备到边缘服务器的传输速率和带宽"""
        self.device_to_edge_rates[(device_id, edge_id)] = rate
        self.device_to_edge_bandwidth[(device_id, edge_id)] = bandwidth
    
    def set_edge_to_cloud_rate(self, edge_id: int, cloud_id: int, rate: float, bandwidth: float) -> None:
        """设置边缘服务器到云服务器的传输速率和带宽"""
        self.edge_to_cloud_rates[(edge_id, cloud_id)] = rate
        self.edge_to_cloud_bandwidth[(edge_id, cloud_id)] = bandwidth
    
    def get_device_by_id(self, device_id: int) -> Optional[Device]:
        """通过ID获取设备"""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_edge_server_by_id(self, server_id: int) -> Optional[EdgeServer]:
        """通过ID获取边缘服务器"""
        for server in self.edge_servers:
            if server.server_id == server_id:
                return server
        return None
    
    def get_cloud_server_by_id(self, server_id: int) -> Optional[CloudServer]:
        """通过ID获取云服务器"""
        for server in self.cloud_servers:
            if server.server_id == server_id:
                return server
        return None
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """通过ID获取任务"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def apply_solution(self, solution: List[List[Union[int, float]]]) -> None:
        """
        应用优化算法的解决方案到系统模型
        
        Parameters:
        -----------
        solution : List[List[Union[int, float]]]
            解决方案列表，每个元素是一个任务的解决方案 [loc_i, f_i, Δ_i]
            loc_i是执行位置，f_i是分配的计算资源，Δ_i是更新间隔(可选)
        """
        num_devices = len(self.devices)
        num_edge_servers = len(self.edge_servers)
        
        for i, task_solution in enumerate(solution):
            if i >= len(self.tasks):
                break
                
            task = self.tasks[i]
            loc_i = int(task_solution[0])  # 执行位置
            f_i = task_solution[1]         # 分配的计算资源
            
            # 设置任务更新间隔（如果有）
            if len(task_solution) > 2 and task.update_interval is not None:
                task.update_interval = task_solution[2]
            
            # 根据loc_i设置执行位置和节点ID
            if loc_i < num_devices:  # 在设备上执行
                task.execution_location = 'device'
                task.execution_node_id = self.devices[loc_i].device_id
            elif loc_i < num_devices + num_edge_servers:  # 在边缘服务器上执行
                task.execution_location = 'edge'
                task.execution_node_id = self.edge_servers[loc_i - num_devices].server_id
            else:  # 在云服务器上执行
                task.execution_location = 'cloud'
                task.execution_node_id = self.cloud_servers[loc_i - num_devices - num_edge_servers].server_id
            
            # 设置分配的计算资源
            task.allocated_resource = f_i
    
    def encode_solution(self) -> List[List[Union[int, float]]]:
        """
        将当前系统状态编码为解决方案，用于优化算法
        
        Returns:
        --------
        List[List[Union[int, float]]]
            编码后的解决方案
        """
        solution = []
        num_devices = len(self.devices)
        num_edge_servers = len(self.edge_servers)
        
        for task in self.tasks:
            if task.execution_location is None:
                # 如果任务尚未分配，使用默认值
                solution.append([0, 0.0])
                continue
                
            # 计算loc_i
            if task.execution_location == 'device':
                for i, device in enumerate(self.devices):
                    if device.device_id == task.execution_node_id:
                        loc_i = i
                        break
            elif task.execution_location == 'edge':
                for i, server in enumerate(self.edge_servers):
                    if server.server_id == task.execution_node_id:
                        loc_i = num_devices + i
                        break
            else:  # 'cloud'
                for i, server in enumerate(self.cloud_servers):
                    if server.server_id == task.execution_node_id:
                        loc_i = num_devices + num_edge_servers + i
                        break
            
            # 创建解决方案
            if task.update_interval is not None:
                solution.append([loc_i, task.allocated_resource, task.update_interval])
            else:
                solution.append([loc_i, task.allocated_resource])
        
        return solution
    
    def from_config(cls, config: Dict) -> 'SystemModel':
        """
        从配置字典创建系统模型
        
        Parameters:
        -----------
        config : Dict
            配置字典，包含设备、服务器和任务的参数
            
        Returns:
        --------
        SystemModel
            创建的系统模型
        """
        system = cls()
        
        # 创建设备
        for device_config in config.get('devices', []):
            device = Device(
                device_id=device_config['id'],
                max_cpu_frequency=device_config['max_cpu_frequency'],
                energy_coefficient=device_config['energy_coefficient'],
                transmission_power=device_config['transmission_power']
            )
            system.add_device(device)
        
        # 创建边缘服务器
        for server_config in config.get('edge_servers', []):
            server = EdgeServer(
                server_id=server_config['id'],
                max_cpu_frequency=server_config['max_cpu_frequency'],
                energy_coefficient=server_config['energy_coefficient'],
                transmission_power=server_config['transmission_power']
            )
            system.add_edge_server(server)
        
        # 创建云服务器
        for server_config in config.get('cloud_servers', []):
            server = CloudServer(
                server_id=server_config['id'],
                max_cpu_frequency=server_config['max_cpu_frequency'],
                energy_coefficient=server_config['energy_coefficient']
            )
            system.add_cloud_server(server)
        
        # 创建任务
        for task_config in config.get('tasks', []):
            task = Task(
                task_id=task_config['id'],
                data_size=task_config['data_size'],
                computation_complexity=task_config['computation_complexity'],
                arrival_rate=task_config['arrival_rate'],
                priority=task_config['priority'],
                max_delay=task_config['max_delay'],
                update_interval=task_config.get('update_interval'),
                max_aoi=task_config.get('max_aoi')
            )
            system.add_task(task)
        
        # 设置网络参数
        for link in config.get('device_to_edge_links', []):
            system.set_device_to_edge_rate(
                device_id=link['device_id'],
                edge_id=link['edge_id'],
                rate=link['rate'],
                bandwidth=link['bandwidth']
            )
        
        for link in config.get('edge_to_cloud_links', []):
            system.set_edge_to_cloud_rate(
                edge_id=link['edge_id'],
                cloud_id=link['cloud_id'],
                rate=link['rate'],
                bandwidth=link['bandwidth']
            )
        
        return system