# src/algorithms/base_algorithm.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ..models.system_model import SystemModel


class BaseAlgorithm(ABC):
    """所有优化算法的基类"""
    
    def __init__(self, system_model: SystemModel, max_iter: int = 100, 
                 population_size: int = 50, verbose: bool = False):
        """
        初始化算法基类
        
        Parameters:
        -----------
        system_model : SystemModel
            系统模型实例
        max_iter : int
            最大迭代次数
        population_size : int
            种群大小
        verbose : bool
            是否输出详细信息
        """
        self.system_model = system_model
        self.max_iter = max_iter
        self.population_size = population_size
        self.verbose = verbose
        
        # 解的维度和范围
        self.num_tasks = len(system_model.tasks)
        self.num_devices = len(system_model.devices)
        self.num_edge_servers = len(system_model.edge_servers)
        self.num_cloud_servers = len(system_model.cloud_servers)
        
        # 设置边界
        self.loc_bounds = (0, self.num_devices + self.num_edge_servers + self.num_cloud_servers - 1)
        
        # 最大CPU频率
        self.device_max_freq = max([d.max_cpu_frequency for d in system_model.devices]) if system_model.devices else 0
        self.edge_max_freq = max([s.max_cpu_frequency for s in system_model.edge_servers]) if system_model.edge_servers else 0
        self.cloud_max_freq = max([s.max_cpu_frequency for s in system_model.cloud_servers]) if system_model.cloud_servers else 0
        
        self.freq_bounds = (1e8, max(self.device_max_freq, self.edge_max_freq, self.cloud_max_freq))  # 最小100MHz，最大为系统中最大频率
        
        # 更新间隔的范围（仅用于第二部分，考虑AoI）
        self.update_interval_bounds = (0.1, 10.0)  # 0.1秒到10秒
        
        # 最优解记录
        self.best_solution = None
        self.best_fitness = float('inf')  # 最小化问题，初始设为无穷大
        
        # 是否考虑AoI
        self.consider_aoi = any(task.update_interval is not None for task in system_model.tasks)
    
    @abstractmethod
    def initialize_population(self):
        """初始化种群"""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, solution):
        """评估解的适应度（目标函数值）"""
        pass
    
    
    def update_solution(self, solution):
        """更新解"""
        pass
    
    @abstractmethod
    def optimize(self):
        """执行优化"""
        pass
    
    def handle_constraints(self, solution):
        """处理约束，确保解在可行域内"""
        for i, task_solution in enumerate(solution):
            # 处理位置约束
            loc_i = int(round(task_solution[0]))
            loc_i = max(self.loc_bounds[0], min(self.loc_bounds[1], loc_i))
            
            # 处理资源分配约束
            f_i = task_solution[1]
            f_i = max(self.freq_bounds[0], min(self.freq_bounds[1], f_i))
            
            # 更新位置和资源分配
            solution[i][0] = loc_i
            solution[i][1] = f_i
            
            # 如果考虑AoI，处理更新间隔约束
            if self.consider_aoi and len(task_solution) > 2:
                delta_i = task_solution[2]
                delta_i = max(self.update_interval_bounds[0], min(self.update_interval_bounds[1], delta_i))
                solution[i][2] = delta_i
        
        return solution