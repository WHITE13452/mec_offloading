# src/algorithms/tlbo_plus.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .tlbo import TLBO
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from scipy import special


class TLBOPlus(TLBO):
    """改进的基于教学的优化算法（TLBO+）"""
    
    def __init__(self, system_model: SystemModel, 
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: Optional[AoIModel] = None,
                 max_iter: int = 100,
                 population_size: int = 50,
                 w_energy: float = 0.4,
                 w_delay: float = 0.6,
                 w_aoi: float = 0.0,
                 levy_alpha: float = 1.5,
                 verbose: bool = False):
        """
        初始化TLBO+算法
        
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
        levy_alpha : float
            莱维飞行的参数alpha
        verbose : bool
            是否输出详细信息
        """
        super().__init__(system_model, delay_model, energy_model, aoi_model,
                         max_iter, population_size, w_energy, w_delay, w_aoi, verbose)
        
        self.levy_alpha = levy_alpha
        self.elite_solution = None
        self.elite_fitness = float('inf')
    
    def levy_flight(self, size):
        """
        生成莱维飞行步长
        
        Parameters:
        -----------
        size : tuple
            生成的步长数组形状
            
        Returns:
        --------
        numpy.ndarray
            莱维飞行步长
        """
        sigma = (special.gamma(1 + self.levy_alpha) * np.sin(np.pi * self.levy_alpha / 2) / 
                (special.gamma((1 + self.levy_alpha) / 2) * self.levy_alpha * 2 ** ((self.levy_alpha - 1) / 2))) ** (1 / self.levy_alpha)
        
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        
        step = u / np.abs(v) ** (1 / self.levy_alpha)
        
        # 归一化步长以控制搜索范围
        step = step / np.max(np.abs(step)) * 2.0
        
        return step
    
    def adaptive_tf(self, teacher_fitness, avg_fitness):
        """
        计算自适应教学因子
        
        Parameters:
        -----------
        teacher_fitness : float
            教师的适应度值
        avg_fitness : float
            种群平均适应度值
            
        Returns:
        --------
        float
            自适应教学因子
        """
        r = np.random.random()
        
        # 确保不会出现除零错误
        if avg_fitness == 0:
            return 1 + r
        
        # 教师与平均值差距越大，教学因子越大
        tf = 1 + r * np.exp(-teacher_fitness / avg_fitness)
        
        return tf
    
    def teacher_phase(self, population, fitness_values):
        """
        TLBO+的教师阶段，引入莱维飞行和自适应教学因子
        
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
        teacher_fitness = fitness_values[best_idx]
        
        # 计算平均解和平均适应度
        mean_solution = np.mean(population, axis=0)
        avg_fitness = np.mean(fitness_values)
        
        # 计算自适应教学因子
        tf = self.adaptive_tf(teacher_fitness, avg_fitness)
        
        for i, student in enumerate(population):
            # 生成莱维飞行步长
            levy_step = self.levy_flight(np.array(student).shape)
            
            # 生成随机权重
            r = np.random.rand(*np.array(student).shape)
            
            # 更新学生（通过教师），结合莱维飞行
            new_student = np.array(student) + levy_step * r * (np.array(teacher) - tf * mean_solution)
            
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
        执行TLBO+优化，增加精英保留策略
        
        Returns:
        --------
        Tuple[List[List[Union[int, float]]], float, List[float]]
            最优解、最优适应度值和迭代历史
        """
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        fitness_values = [self.evaluate_fitness(solution) for solution in population]
        
        # 记录最佳解和精英解
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx]
        self.best_fitness = fitness_values[best_idx]
        self.elite_solution = self.best_solution
        self.elite_fitness = self.best_fitness
        
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
            
            # 精英保留策略
            if self.best_fitness < self.elite_fitness:
                self.elite_solution = self.best_solution
                self.elite_fitness = self.best_fitness
            else:
                # 将精英解替换种群中的最差解
                worst_idx = np.argmax(fitness_values)
                population[worst_idx] = self.elite_solution
                fitness_values[worst_idx] = self.elite_fitness
            
            # 记录历史
            self.history.append(self.elite_fitness)
            
            if self.verbose and (iter_idx + 1) % 10 == 0:
                print(f"Iteration {iter_idx + 1}/{self.max_iter}, Best fitness: {self.elite_fitness:.6f}")
        
        if self.verbose:
            print(f"Optimization completed. Best fitness: {self.elite_fitness:.6f}")
        
        return self.elite_solution, self.elite_fitness, self.history