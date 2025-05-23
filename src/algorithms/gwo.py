# src/algorithms/gwo.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_algorithm import BaseAlgorithm
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel


class GWO(BaseAlgorithm):
    """灰狼优化算法（Grey Wolf Optimizer）- 修复版"""
    
    def __init__(self, system_model: SystemModel, 
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: Optional[AoIModel] = None,
                 max_iter: int = 100,
                 population_size: int = 50,
                 w_energy: float = 0.4,
                 w_delay: float = 0.6,
                 w_aoi: float = 0.0,
                 verbose: bool = False):
        """
        初始化灰狼优化算法
        """
        super().__init__(system_model, max_iter, population_size, verbose)
        
        self.delay_model = delay_model
        self.energy_model = energy_model
        self.aoi_model = aoi_model
        
        self.w_energy = w_energy
        self.w_delay = w_delay
        self.w_aoi = w_aoi
        
        # 预设归一化因子，避免除零
        self.energy_max = 1e5  # 100,000 J
        self.delay_max = 1e3   # 1,000 s
        self.aoi_max = 1e2     # 100 s
        
        # 迭代历史记录
        self.history = []
        
        # 初始化三个最优狼（α、β、δ）
        self.alpha_pos = None
        self.alpha_score = float('inf')
        
        self.beta_pos = None
        self.beta_score = float('inf')
        
        self.delta_pos = None
        self.delta_score = float('inf')
    
    def initialize_population(self):
        """初始化种群（狼群）"""
        population = []
        
        for _ in range(self.population_size):
            solution = []
            for _ in range(self.num_tasks):
                loc_i = np.random.randint(self.loc_bounds[0], self.loc_bounds[1] + 1)
                f_i = np.random.uniform(self.freq_bounds[0], self.freq_bounds[1])
                
                if self.consider_aoi:
                    delta_i = np.random.uniform(self.update_interval_bounds[0], self.update_interval_bounds[1])
                    solution.append([loc_i, f_i, delta_i])
                else:
                    solution.append([loc_i, f_i])
            
            solution = self.handle_constraints(solution)
            population.append(solution)
        
        return population
    
    def evaluate_fitness(self, solution):
        """
        评估解的适应度（目标函数值）- 修复版
        """
        try:
            # 确保解的格式正确
            if not solution or len(solution) != self.num_tasks:
                return float('inf')
            
            # 应用约束处理
            solution = self.handle_constraints(solution)
            
            # 将解应用到系统模型
            self.system_model.apply_solution(solution)
            
            total_energy = 0.0
            total_delay = 0.0
            total_aoi = 0.0
            
            # 计算所有任务的能耗和延迟
            for task in self.system_model.tasks:
                try:
                    # 计算延迟
                    delay = self.delay_model.calculate_total_delay(task)
                    if not np.isfinite(delay) or delay < 0:
                        return float('inf')
                    total_delay += delay
                    
                    # 计算能耗
                    energy = self.energy_model.calculate_total_energy(task)
                    if not np.isfinite(energy) or energy < 0:
                        return float('inf')
                    total_energy += energy
                    
                    # 计算AoI（如果考虑）
                    if self.consider_aoi and self.aoi_model and task.update_interval is not None:
                        aoi = self.aoi_model.calculate_average_aoi(task)
                        if not np.isfinite(aoi) or aoi < 0:
                            return float('inf')
                        total_aoi += aoi
                        
                except Exception:
                    return float('inf')
            
            # 检查总值
            if not (np.isfinite(total_energy) and np.isfinite(total_delay)):
                return float('inf')
            
            # 更新归一化因子（但保持最小值）
            self.energy_max = max(self.energy_max, total_energy, 1.0)
            self.delay_max = max(self.delay_max, total_delay, 1.0)
            if self.consider_aoi:
                self.aoi_max = max(self.aoi_max, total_aoi, 1.0)
            
            # 计算加权目标函数
            fitness = (self.w_energy * total_energy / self.energy_max + 
                      self.w_delay * total_delay / self.delay_max)
            
            if self.consider_aoi and self.aoi_model:
                fitness += self.w_aoi * total_aoi / self.aoi_max
            
            return fitness if np.isfinite(fitness) else float('inf')
            
        except Exception:
            return float('inf')
    
    def update_wolves(self, population, fitness_values):
        """
        更新三个领导狼（α、β、δ）- 修复版
        """
        for i in range(self.population_size):
            if not np.isfinite(fitness_values[i]):
                continue
            
            # 如果优于α
            if fitness_values[i] < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos
                
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos
                
                self.alpha_score = fitness_values[i]
                self.alpha_pos = population[i].copy()
            
            # 如果优于β但不如α
            elif fitness_values[i] < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos
                
                self.beta_score = fitness_values[i]
                self.beta_pos = population[i].copy()
            
            # 如果优于δ但不如α和β
            elif fitness_values[i] < self.delta_score:
                self.delta_score = fitness_values[i]
                self.delta_pos = population[i].copy()
    
    def update_population(self, population, iter_idx):
        """
        更新种群位置 - 修复版
        """
        new_population = []
        
        # 计算参数a，在迭代过程中线性递减（从2到0）
        a = 2 - iter_idx * (2 / self.max_iter)
        
        for i in range(self.population_size):
            try:
                new_position = []
                
                # 为每个任务更新位置
                for j in range(self.num_tasks):
                    task_dim = len(population[i][j])
                    task_new_pos = []
                    
                    for k in range(task_dim):
                        # 计算A和C参数
                        r1 = np.random.random()
                        r2 = np.random.random()
                        
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        
                        r1 = np.random.random()
                        r2 = np.random.random()
                        
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        
                        r1 = np.random.random()
                        r2 = np.random.random()
                        
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        
                        # 计算到三个领导狼的距离
                        D_alpha = abs(C1 * self.alpha_pos[j][k] - population[i][j][k])
                        D_beta = abs(C2 * self.beta_pos[j][k] - population[i][j][k])
                        D_delta = abs(C3 * self.delta_pos[j][k] - population[i][j][k])
                        
                        # 计算三个领导狼引导的新位置
                        X1 = self.alpha_pos[j][k] - A1 * D_alpha
                        X2 = self.beta_pos[j][k] - A2 * D_beta
                        X3 = self.delta_pos[j][k] - A3 * D_delta
                        
                        # 计算新位置（三个领导狼引导位置的平均）
                        new_pos_k = (X1 + X2 + X3) / 3
                        
                        # 如果是位置变量（第一维），则四舍五入为整数
                        if k == 0:
                            new_pos_k = int(round(new_pos_k))
                        
                        task_new_pos.append(new_pos_k)
                    
                    new_position.append(task_new_pos)
                
                # 确保新位置满足约束
                new_position = self.handle_constraints(new_position)
                new_population.append(new_position)
                
            except Exception:
                # 如果更新失败，保留原位置
                new_population.append(population[i].copy())
        
        return new_population
    
    def optimize(self):
        """
        执行灰狼优化算法 - 修复版
        """
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        fitness_values = [self.evaluate_fitness(solution) for solution in population]
        
        # 找到有效解
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            if self.verbose:
                print("Warning: No valid solutions found in initial population!")
            return None, float('inf'), []
        
        # 初始化领导狼
        # 设置一个有效的初始解作为领导狼
        init_idx = valid_indices[0]
        self.alpha_pos = population[init_idx].copy()
        self.beta_pos = population[init_idx].copy()
        self.delta_pos = population[init_idx].copy()
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        
        # 更新三个领导狼
        self.update_wolves(population, fitness_values)
        
        # 记录最佳解（α狼）
        self.best_solution = self.alpha_pos
        self.best_fitness = self.alpha_score
        
        # 初始化迭代历史
        self.history = [self.best_fitness]
        
        for iter_idx in range(self.max_iter):
            try:
                # 更新种群位置
                population = self.update_population(population, iter_idx)
                
                # 评估新种群
                fitness_values = [self.evaluate_fitness(solution) for solution in population]
                
                # 更新三个领导狼
                self.update_wolves(population, fitness_values)
                
                # 更新最佳解（α狼）
                if self.alpha_score < self.best_fitness:
                    self.best_solution = self.alpha_pos.copy()
                    self.best_fitness = self.alpha_score
                
                # 记录历史
                self.history.append(self.best_fitness)
                
                if self.verbose and (iter_idx + 1) % 10 == 0:
                    print(f"Iteration {iter_idx + 1}/{self.max_iter}, Best fitness: {self.best_fitness:.6f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error in iteration {iter_idx}: {e}")
                self.history.append(self.best_fitness)
        
        if self.verbose:
            print(f"Optimization completed. Best fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness, self.history