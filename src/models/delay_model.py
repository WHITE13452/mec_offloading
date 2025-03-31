# src/models/delay_model.py
from typing import Dict, Optional, Tuple
from .system_model import SystemModel, Task


class DelayModel:
    """计算各种延迟的模型"""
    
    def __init__(self, system_model: SystemModel):
        """
        初始化延迟模型
        
        Parameters:
        -----------
        system_model : SystemModel
            系统模型实例
        """
        self.system_model = system_model
    
    def calculate_computation_delay(self, task: Task) -> float:
        """
        计算给定任务的计算延迟
        
        Parameters:
        -----------
        task : Task
            要计算延迟的任务
            
        Returns:
        --------
        float
            计算延迟（单位：秒）
        """
        if task.execution_location is None or task.allocated_resource is None:
            raise ValueError(f"Task {task.task_id} has not been allocated.")
        
        # 计算延迟 = 数据大小 * 计算复杂度 / 分配的计算资源
        return task.data_size * task.computation_complexity / task.allocated_resource
    
    def calculate_transmission_delay(self, task: Task) -> Tuple[float, float]:
        """
        计算给定任务的传输延迟（设备到边缘 和/或 边缘到云）
        
        Parameters:
        -----------
        task : Task
            要计算延迟的任务
            
        Returns:
        --------
        Tuple[float, float]
            (设备到边缘的传输延迟, 边缘到云的传输延迟)，如果不适用则为0
        """
        if task.execution_location is None or task.execution_node_id is None:
            raise ValueError(f"Task {task.task_id} has not been allocated.")
        
        d_to_e_delay = 0.0
        e_to_c_delay = 0.0
        
        if task.execution_location == 'edge':
            # 需要从设备传输到边缘
            device = next((d for d in self.system_model.devices if d.device_id == task.task_id), None)
            if device is None:
                raise ValueError(f"Task {task.task_id} is not associated with any device.")
            
            rate = self.system_model.device_to_edge_rates.get((device.device_id, task.execution_node_id))
            if rate is None:
                raise ValueError(f"No transmission rate defined between device {device.device_id} and edge server {task.execution_node_id}.")
            
            d_to_e_delay = task.data_size / rate
        
        elif task.execution_location == 'cloud':
            # 需要从设备传输到边缘，再从边缘传输到云
            device = next((d for d in self.system_model.devices if d.device_id == task.task_id), None)
            if device is None:
                raise ValueError(f"Task {task.task_id} is not associated with any device.")
            
            # 找到最近的边缘服务器（简化：假设使用第一个边缘服务器）
            if not self.system_model.edge_servers:
                raise ValueError("No edge servers available.")
            
            edge_server = self.system_model.edge_servers[0]
            
            # 设备到边缘的延迟
            d_to_e_rate = self.system_model.device_to_edge_rates.get((device.device_id, edge_server.server_id))
            if d_to_e_rate is None:
                raise ValueError(f"No transmission rate defined between device {device.device_id} and edge server {edge_server.server_id}.")
            
            d_to_e_delay = task.data_size / d_to_e_rate
            
            # 边缘到云的延迟
            e_to_c_rate = self.system_model.edge_to_cloud_rates.get((edge_server.server_id, task.execution_node_id))
            if e_to_c_rate is None:
                raise ValueError(f"No transmission rate defined between edge server {edge_server.server_id} and cloud server {task.execution_node_id}.")
            
            e_to_c_delay = task.data_size / e_to_c_rate
        
        return d_to_e_delay, e_to_c_delay
    
    def calculate_queue_delay(self, task: Task) -> float:
        """
        使用M/M/1队列模型计算排队延迟
        
        Parameters:
        -----------
        task : Task
            要计算延迟的任务
            
        Returns:
        --------
        float
            排队延迟（单位：秒）
        """
        if task.execution_location is None or task.execution_node_id is None:
            raise ValueError(f"Task {task.task_id} has not been allocated.")
        
        # 根据执行位置获取到达率
        if task.execution_location == 'device':
            node = next((d for d in self.system_model.devices if d.device_id == task.execution_node_id), None)
            arrival_rate = node.arrival_rates.get(task.task_id, 0) if node else 0
        elif task.execution_location == 'edge':
            node = next((s for s in self.system_model.edge_servers if s.server_id == task.execution_node_id), None)
            arrival_rate = node.arrival_rates.get(task.task_id, 0) if node else 0
        else:  # 'cloud'
            node = next((s for s in self.system_model.cloud_servers if s.server_id == task.execution_node_id), None)
            arrival_rate = node.arrival_rates.get(task.task_id, 0) if node else 0
        
        if node is None:
            raise ValueError(f"No node found with id {task.execution_node_id} for location {task.execution_location}.")
        
        # 使用M/M/1队列模型计算排队延迟
        # 队列延迟 = λ * (D * C)² / (f * (f - λ * D * C))
        service_time = task.data_size * task.computation_complexity
        service_rate = task.allocated_resource / service_time
        
        # 确保系统稳定 (λ < μ)
        if arrival_rate >= service_rate:
            # 系统不稳定，返回一个大值
            return float('inf')
        
        queue_delay = arrival_rate * service_time**2 / (task.allocated_resource * (task.allocated_resource - arrival_rate * service_time))
        return queue_delay
    
    def calculate_total_delay(self, task: Task) -> float:
        """
        计算任务的总延迟
        
        Parameters:
        -----------
        task : Task
            要计算延迟的任务
            
        Returns:
        --------
        float
            总延迟（单位：秒）
        """
        computation_delay = self.calculate_computation_delay(task)
        d_to_e_delay, e_to_c_delay = self.calculate_transmission_delay(task)
        queue_delay = self.calculate_queue_delay(task)
        
        total_delay = computation_delay + d_to_e_delay + e_to_c_delay + queue_delay
        
        # 更新任务的延迟字段
        task.delay = total_delay
        
        return total_delay