# src/algorithms/fixed_base_algorithm.py
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from ..models.system_model import SystemModel

class FixedBaseAlgorithm(ABC):
    """修复后的算法基类"""
    
    def __init__(self, system_model: SystemModel, max_iter: int = 100, 
                 population_size: int = 50, verbose: bool = False):
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
        self.device_max_freq = max([d.max_cpu_frequency for d in system_model.devices]) if system_model.devices else 1e9
        self.edge_max_freq = max([s.max_cpu_frequency for s in system_model.edge_servers]) if system_model.edge_servers else 1e9
        self.cloud_max_freq = max([s.max_cpu_frequency for s in system_model.cloud_servers]) if system_model.cloud_servers else 1e9
        
        self.freq_bounds = (1e8, max(self.device_max_freq, self.edge_max_freq, self.cloud_max_freq))
        
        # 更新间隔的范围
        self.update_interval_bounds = (0.1, 10.0)
        
        # 最优解记录
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # 是否考虑AoI
        self.consider_aoi = any(task.update_interval is not None for task in system_model.tasks)
        
        # 预计算归一化因子，避免运行时的数值问题
        self._precompute_normalization_factors()
    
    def _precompute_normalization_factors(self):
        """预计算归一化因子"""
        # 基于系统参数估算合理的归一化因子
        max_energy_per_task = 0
        max_delay_per_task = 0
        
        for task in self.system_model.tasks:
            # 估算最大可能的能耗（最高频率 + 最大数据 + 最高复杂度）
            max_freq = max(self.device_max_freq, self.edge_max_freq, self.cloud_max_freq)
            max_energy_coeff = 3.5e-27  # 基于你的系统设计
            estimated_energy = max_energy_coeff * (max_freq ** 2) * task.data_size * task.computation_complexity
            max_energy_per_task = max(max_energy_per_task, estimated_energy)
            
            # 估算最大可能的延迟（最低频率 + 传输延迟）
            min_freq = 1e8  # 100MHz
            computation_delay = task.data_size * task.computation_complexity / min_freq
            transmission_delay = task.data_size / 2e6  # 假设最低2Mbps传输
            queue_delay = 5.0  # 假设最大队列延迟5秒
            estimated_delay = computation_delay + transmission_delay + queue_delay
            max_delay_per_task = max(max_delay_per_task, estimated_delay)
        
        # 设置归一化因子为合理的上界
        self.energy_max = max_energy_per_task * len(self.system_model.tasks)
        self.delay_max = max_delay_per_task * len(self.system_model.tasks)
        self.aoi_max = 10.0 * len(self.system_model.tasks)  # 假设最大AoI为10秒
        
        if self.verbose:
            print(f"预设归一化因子 - Energy: {self.energy_max:.2e}, Delay: {self.delay_max:.2e}")
    
    def handle_constraints(self, solution):
        """改进的约束处理"""
        try:
            for i, task_solution in enumerate(solution):
                if not isinstance(task_solution, list) or len(task_solution) < 2:
                    # 修复损坏的解
                    if self.consider_aoi:
                        solution[i] = [0, self.freq_bounds[0], self.update_interval_bounds[0]]
                    else:
                        solution[i] = [0, self.freq_bounds[0]]
                    continue
                
                # 处理位置约束
                loc_i = task_solution[0]
                if not np.isfinite(loc_i):
                    loc_i = 0
                loc_i = int(np.clip(round(loc_i), self.loc_bounds[0], self.loc_bounds[1]))
                
                # 处理资源分配约束
                f_i = task_solution[1]
                if not np.isfinite(f_i) or f_i <= 0:
                    f_i = self.freq_bounds[0]
                f_i = np.clip(f_i, self.freq_bounds[0], self.freq_bounds[1])
                
                # 更新位置和资源分配
                solution[i][0] = loc_i
                solution[i][1] = f_i
                
                # 如果考虑AoI，处理更新间隔约束
                if self.consider_aoi and len(task_solution) > 2:
                    delta_i = task_solution[2]
                    if not np.isfinite(delta_i) or delta_i <= 0:
                        delta_i = self.update_interval_bounds[0]
                    delta_i = np.clip(delta_i, self.update_interval_bounds[0], self.update_interval_bounds[1])
                    solution[i][2] = delta_i
            
            return solution
            
        except Exception as e:
            if self.verbose:
                print(f"Error in handle_constraints: {e}")
            # 返回默认可行解
            default_solution = []
            for _ in range(self.num_tasks):
                if self.consider_aoi:
                    default_solution.append([0, self.freq_bounds[0], self.update_interval_bounds[0]])
                else:
                    default_solution.append([0, self.freq_bounds[0]])
            return default_solution
    
    @abstractmethod
    def initialize_population(self):
        pass
    
    @abstractmethod
    def evaluate_fitness(self, solution):
        pass
    
    @abstractmethod
    def optimize(self):
        pass