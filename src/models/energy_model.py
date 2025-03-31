# src/models/energy_model.py
from typing import Dict, Optional
from .system_model import SystemModel, Task


class EnergyModel:
    """计算能耗的模型"""
    
    def __init__(self, system_model: SystemModel):
        """
        初始化能耗模型
        
        Parameters:
        -----------
        system_model : SystemModel
            系统模型实例
        """
        self.system_model = system_model
    
    def calculate_computation_energy(self, task: Task) -> float:
        """
        计算给定任务的计算能耗
        
        Parameters:
        -----------
        task : Task
            要计算能耗的任务
            
        Returns:
        --------
        float
            计算能耗（单位：焦耳）
        """
        if task.execution_location is None or task.allocated_resource is None:
            raise ValueError(f"Task {task.task_id} has not been allocated.")
        
        # 找到执行任务的节点
        if task.execution_location == 'device':
            node = next((d for d in self.system_model.devices if d.device_id == task.execution_node_id), None)
            if node is None:
                raise ValueError(f"No device found with id {task.execution_node_id}.")
            
            # 计算能耗 = κ * f² * D * C
            energy = node.energy_coefficient * (task.allocated_resource ** 2) * task.data_size * task.computation_complexity
        
        elif task.execution_location == 'edge':
            node = next((s for s in self.system_model.edge_servers if s.server_id == task.execution_node_id), None)
            if node is None:
                raise ValueError(f"No edge server found with id {task.execution_node_id}.")
            
            energy = node.energy_coefficient * (task.allocated_resource ** 2) * task.data_size * task.computation_complexity
        
        else:  # 'cloud'
            node = next((s for s in self.system_model.cloud_servers if s.server_id == task.execution_node_id), None)
            if node is None:
                raise ValueError(f"No cloud server found with id {task.execution_node_id}.")
            
            energy = node.energy_coefficient * (task.allocated_resource ** 2) * task.data_size * task.computation_complexity
        
        return energy
    
    def calculate_transmission_energy(self, task: Task) -> float:
        """
        计算给定任务的传输能耗
        
        Parameters:
        -----------
        task : Task
            要计算能耗的任务
            
        Returns:
        --------
        float
            传输能耗（单位：焦耳）
        """
        if task.execution_location is None or task.execution_node_id is None:
            raise ValueError(f"Task {task.task_id} has not been allocated.")
        
        # 如果在本地执行，则没有传输能耗
        if task.execution_location == 'device':
            return 0.0
        
        # 找到任务关联的设备
        device = next((d for d in self.system_model.devices if d.device_id == task.task_id), None)
        if device is None:
            raise ValueError(f"Task {task.task_id} is not associated with any device.")
        
        # 如果在边缘服务器执行
        if task.execution_location == 'edge':
            # 找到与设备连接的边缘服务器
            rate = self.system_model.device_to_edge_rates.get((device.device_id, task.execution_node_id))
            if rate is None:
                raise ValueError(f"No transmission rate defined between device {device.device_id} and edge server {task.execution_node_id}.")
            
            # 传输能耗 = P * D / R
            energy = device.transmission_power * task.data_size / rate
            return energy
        
        # 如果在云服务器执行
        else:  # 'cloud'
            # 找到最近的边缘服务器（简化：假设使用第一个边缘服务器）
            if not self.system_model.edge_servers:
                raise ValueError("No edge servers available.")
            
            edge_server = self.system_model.edge_servers[0]
            
            # 设备到边缘的能耗
            d_to_e_rate = self.system_model.device_to_edge_rates.get((device.device_id, edge_server.server_id))
            if d_to_e_rate is None:
                raise ValueError(f"No transmission rate defined between device {device.device_id} and edge server {edge_server.server_id}.")
            
            d_to_e_energy = device.transmission_power * task.data_size / d_to_e_rate
            
            # 边缘到云的能耗
            e_to_c_rate = self.system_model.edge_to_cloud_rates.get((edge_server.server_id, task.execution_node_id))
            if e_to_c_rate is None:
                raise ValueError(f"No transmission rate defined between edge server {edge_server.server_id} and cloud server {task.execution_node_id}.")
            
            e_to_c_energy = edge_server.transmission_power * task.data_size / e_to_c_rate
            
            return d_to_e_energy + e_to_c_energy
    
    def calculate_total_energy(self, task: Task) -> float:
        """
        计算任务的总能耗
        
        Parameters:
        -----------
        task : Task
            要计算能耗的任务
            
        Returns:
        --------
        float
            总能耗（单位：焦耳）
        """
        computation_energy = self.calculate_computation_energy(task)
        transmission_energy = self.calculate_transmission_energy(task)
        
        total_energy = computation_energy + transmission_energy
        
        # 更新任务的能耗字段
        task.energy = total_energy
        
        return total_energy