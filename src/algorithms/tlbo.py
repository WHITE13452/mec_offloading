# src/algorithms/tlbo.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_algorithm import BaseAlgorithm
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel


class TLBO(BaseAlgorithm):
    """基于教学的优化算法（Teaching-Learning-Based Optimization）"""
    
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
        初始化TLBO算法
        
        Parameters:
        -----------
        system_model : SystemModel
            系统模型实例
        delay_model : DelayModel
            延迟模型实例
        energy_model : EnergyModel
            能耗模型实例
        aoi_model : AoIModel, optional
            AoI模型实例，如果为None则不考虑AoI
        max_iter : int
            最大迭代次数
        population_size : int
            种群大小（学生数量）
        w_energy : float
            能耗的权重系数
        w_delay : float
            延迟的权重系数
        w_aoi : float
            AoI的权重系数
        verbose : bool
            是否输出详细信息
        """
        super().__init__(system_model, max_iter, population_size, verbose)
        
        self.delay_model = delay_model
        self.energy_model = energy_model
        self.aoi_model = aoi_model
        
        self.w_energy = w_energy
        self.w_delay = w_delay
        self.w_aoi = w_aoi
        
        # 归一化因子（用于目标函数）
        self.energy_max = 1.0
        self.delay_max = 1.0
        self.aoi_max = 1.0
        
        # 迭代历史记录
        self.history = []
    
    def initialize_population(self):
        """初始化种群（学生）"""
        population = []
        
        for _ in range(self.population_size):
            # 为每个任务生成随机解
            solution = []
            for _ in range(self.num_tasks):
                # 随机选择执行位置
                loc_i = np.random.randint(self.loc_bounds[0], self.loc_bounds[1] + 1)
                
                # 随机选择计算资源分配
                f_i = np.random.uniform(self.freq_bounds[0], self.freq_bounds[1])
                
                if self.consider_aoi:
                    # 如果考虑AoI，随机选择更新间隔
                    delta_i = np.random.uniform(self.update_interval_bounds[0], self.update_interval_bounds[1])
                    solution.append([loc_i, f_i, delta_i])
                else:
                    solution.append([loc_i, f_i])
            
            # 确保解满足约束
            solution = self.handle_constraints(solution)
            population.append(solution)
        
        return population
    
    def evaluate_fitness(self, solution):
        """
        评估解的适应度（目标函数值）
        
        Parameters:
        -----------
        solution : List[List[Union[int, float]]]
            解决方案
            
        Returns:
        --------
        float
            适应度值（越小越好）
        """
        # 将解应用到系统模型
        self.system_model.apply_solution(solution)
        
        total_energy = 0.0
        total_delay = 0.0
        total_aoi = 0.0
        
        # 计算所有任务的能耗和延迟
        for task in self.system_model.tasks:
            # 计算延迟
            delay = self.delay_model.calculate_total_delay(task)
            total_delay += delay
            
            # 计算能耗
            energy = self.energy_model.calculate_total_energy(task)
            total_energy += energy
            
            # 计算AoI（如果考虑）
            if self.consider_aoi and self.aoi_model and task.update_interval is not None:
                aoi = self.aoi_model.calculate_average_aoi(task)
                total_aoi += aoi
        
        # 更新归一化因子
        self.energy_max = max(self.energy_max, total_energy)
        self.delay_max = max(self.delay_max, total_delay)
        if self.consider_aoi:
            self.aoi_max = max(self.aoi_max, total_aoi)
        
        # 计算加权目标函数
        fitness = (self.w_energy * total_energy / self.energy_max + 
                  self.w_delay * total_delay / self.delay_max)
        
        if self.consider_aoi and self.aoi_model:
            fitness += self.w_aoi * total_aoi / self.aoi_max
        
        return fitness
    
    def teacher_phase(self, population, fitness_values):
        """
        TLBO的教师阶段
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
        fitness_values : List[float]
            当前种群的适应度值
            
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            更新后的种群
        """
        new_population = []
        
        # 找出最好的解（教师）
        best_idx = np.argmin(fitness_values)
        teacher = population[best_idx]
        
        # 计算平均解
        mean_solution = np.mean(population, axis=0)
        
        for i, student in enumerate(population):
            # 生成随机教学因子
            tf = np.random.randint(1, 3)  # 1或2
            
            # 生成随机权重
            r = np.random.rand(*np.array(student).shape)
            
            # 更新学生（通过教师）
            new_student = np.array(student) + r * (np.array(teacher) - tf * mean_solution)
            
            # 处理约束
            new_student = self.handle_constraints(new_student.tolist())
            
            # 评估新解
            new_fitness = self.evaluate_fitness(new_student)
            
            # 如果新解更好，则接受
            if new_fitness < fitness_values[i]:
                new_population.append(new_student)
            else:
                new_population.append(student)
        
        return new_population
    
    def learner_phase(self, population, fitness_values):
        """
        TLBO的学习者阶段
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
        fitness_values : List[float]
            当前种群的适应度值
            
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            更新后的种群
        """
        new_population = []
        
        for i, student in enumerate(population):
            # 随机选择另一个学生
            j = i
            while j == i:
                j = np.random.randint(0, self.population_size)
            
            other_student = population[j]
            
            # 生成随机权重
            r = np.random.rand(*np.array(student).shape)
            
            # 根据适应度比较，更新学习方向
            if fitness_values[i] < fitness_values[j]:
                new_student = np.array(student) + r * (np.array(student) - np.array(other_student))
            else:
                new_student = np.array(student) + r * (np.array(other_student) - np.array(student))
            
            # 处理约束
            new_student = self.handle_constraints(new_student.tolist())
            
            # 评估新解
            new_fitness = self.evaluate_fitness(new_student)
            
            # 如果新解更好，则接受
            if new_fitness < fitness_values[i]:
                new_population.append(new_student)
            else:
                new_population.append(student)
        
        return new_population
    
    def optimize(self):
        """
        执行TLBO优化
        
        Returns:
        --------
        Tuple[List[List[Union[int, float]]], float, List[float]]
            最优解、最优适应度值和迭代历史
        """
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        fitness_values = [self.evaluate_fitness(solution) for solution in population]
        
        # 记录最佳解
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx]
        self.best_fitness = fitness_values[best_idx]
        
        # 初始化迭代历史
        self.history = [self.best_fitness]
        
        for iter_idx in range(self.max_iter):
            # 教师阶段
            population = self.teacher_phase(population, fitness_values)
            
            # 更新适应度值
            fitness_values = [self.evaluate_fitness(solution) for solution in population]
            
            # 学习者阶段
            population = self.learner_phase(population, fitness_values)
            
            # 更新适应度值
            fitness_values = [self.evaluate_fitness(solution) for solution in population]
            
            # 更新最佳解
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.best_fitness:
                self.best_solution = population[best_idx]
                self.best_fitness = fitness_values[best_idx]
            
            # 记录历史
            self.history.append(self.best_fitness)
            
            if self.verbose and (iter_idx + 1) % 10 == 0:
                print(f"Iteration {iter_idx + 1}/{self.max_iter}, Best fitness: {self.best_fitness:.6f}")
        
        if self.verbose:
            print(f"Optimization completed. Best fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness, self.history
    
    # 在TLBO类中添加update_solution方法的实现
    
    def update_solution(self, solution):
        """
        更新解决方案，确保解在有效范围内
        
        Parameters:
        -----------
        solution : List[List[Union[int, float]]]
            需要更新的解决方案
            
        Returns:
        --------
        List[List[Union[int, float]]]
            更新后的解决方案
        """
        num_devices = len(self.system_model.devices)
        num_edge_servers = len(self.system_model.edge_servers)
        num_cloud_servers = len(self.system_model.cloud_servers)
        
        for i, task_solution in enumerate(solution):
            # 更新执行位置
            loc_i = int(task_solution[0])
            max_loc = num_devices + num_edge_servers + num_cloud_servers - 1
            loc_i = max(0, min(loc_i, max_loc))
            
            # 更新计算资源分配
            f_i = task_solution[1]
            
            # 根据执行位置确定最大可用资源
            if loc_i < num_devices:  # 设备
                max_f = self.system_model.devices[loc_i].max_cpu_frequency
            elif loc_i < num_devices + num_edge_servers:  # 边缘服务器
                edge_idx = loc_i - num_devices
                max_f = self.system_model.edge_servers[edge_idx].max_cpu_frequency
            else:  # 云服务器
                cloud_idx = loc_i - num_devices - num_edge_servers
                max_f = self.system_model.cloud_servers[cloud_idx].max_cpu_frequency
            
            # 确保资源分配在有效范围内
            f_i = max(0.1 * max_f, min(f_i, max_f))
            
            # 更新解决方案
            solution[i][0] = loc_i
            solution[i][1] = f_i
            
            # 如果有更新间隔参数，也进行更新
            if len(task_solution) > 2:
                update_interval = task_solution[2]
                # 确保更新间隔在合理范围内
                task = self.system_model.tasks[i]
                if task.max_aoi is not None:
                    # 更新间隔不应超过最大AoI
                    update_interval = max(0.01, min(update_interval, task.max_aoi))
                else:
                    # 如果没有最大AoI限制，使用一个合理的默认范围
                    update_interval = max(0.01, min(update_interval, 2.0))
                
                solution[i][2] = update_interval
        
        return solution