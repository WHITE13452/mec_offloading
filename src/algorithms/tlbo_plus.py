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
        """初始化TLBO+算法"""
        super().__init__(system_model, delay_model, energy_model, aoi_model,
                         max_iter, population_size, w_energy, w_delay, w_aoi, verbose)
        
        self.levy_alpha = levy_alpha
        self.elite_solution = None
        self.elite_fitness = float('inf')
        
        # 新增：初始化多个精英解（精英池）- 减小数量到2个，避免过多多样性
        self.elite_pool_size = 2
        self.elite_pool = []
        self.elite_pool_fitness = []
    
    def levy_flight(self, size, iter_idx=None):
        """生成莱维飞行步长，减小步长"""
        sigma = (special.gamma(1 + self.levy_alpha) * np.sin(np.pi * self.levy_alpha / 2) / 
                (special.gamma((1 + self.levy_alpha) / 2) * self.levy_alpha * 2 ** ((self.levy_alpha - 1) / 2))) ** (1 / self.levy_alpha)
        
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        
        step = u / np.abs(v) ** (1 / self.levy_alpha)
        
        # 更温和的步长控制
        if iter_idx is not None:
            scale_factor = 0.5 * (1.0 - iter_idx / self.max_iter) ** 2
        else:
            scale_factor = 0.5
            
        max_step = np.max(np.abs(step))
        if max_step > 0:
            step = step / max_step * scale_factor
        
        return step
    
    def adaptive_tf(self, teacher_fitness, avg_fitness, iter_idx):
        """简化自适应教学因子"""
        # 固定教学因子为1，避免过度影响
        return 1.0
    
    def teacher_phase(self, population, fitness_values, iter_idx):
        """更稳定的教师阶段"""
        new_population = []
        
        # 找出最好的解（教师）
        best_idx = np.argmin(fitness_values)
        teacher = population[best_idx]
        teacher_fitness = fitness_values[best_idx]
        
        # 计算平均解和平均适应度
        mean_solution = np.mean(population, axis=0)
        
        # 固定教学因子为1
        tf = 1.0
        
        for i, student in enumerate(population):
            # 生成莱维飞行步长
            levy_step = self.levy_flight(np.array(student).shape, iter_idx)
            
            # 生成随机权重
            r = np.random.rand(*np.array(student).shape)
            
            # 基本TLBO更新公式，添加莱维飞行
            new_student = np.array(student) + levy_step * r * (np.array(teacher) - tf * mean_solution)
            
            # 处理约束
            new_student = self.handle_constraints(new_student.tolist())
            
            # 评估新解
            new_fitness = self.evaluate_fitness(new_student)
            
            # 如果新解更好，则接受
            if new_fitness < fitness_values[i]:
                new_population.append(new_student)
                
                # 更新精英解
                if new_fitness < self.best_fitness:
                    self.best_solution = new_student.copy()
                    self.best_fitness = new_fitness
                    
                    # 更新精英池
                    if new_fitness < self.elite_fitness:
                        self.elite_solution = new_student.copy()
                        self.elite_fitness = new_fitness
                        self.update_elite_pool(new_student.copy(), new_fitness)
            else:
                new_population.append(student)
        
        return new_population
    
    def learner_phase(self, population, fitness_values):
        """更稳定的学习者阶段"""
        new_population = []
        
        for i, student in enumerate(population):
            # 随机选择另一个学生
            j = i
            while j == i:
                j = np.random.randint(0, self.population_size)
            
            other_student = population[j]
            
            # 生成随机权重
            r = np.random.rand(*np.array(student).shape)
            
            # 使用更小的学习强度
            learning_factor = 1.0 + 0.1 * np.random.random()
            
            # 根据适应度比较，更新学习方向
            if fitness_values[i] < fitness_values[j]:
                new_student = np.array(student) + learning_factor * r * (np.array(student) - np.array(other_student))
            else:
                new_student = np.array(student) + learning_factor * r * (np.array(other_student) - np.array(student))
            
            # 处理约束
            new_student = self.handle_constraints(new_student.tolist())
            
            # 评估新解
            new_fitness = self.evaluate_fitness(new_student)
            
            # 如果新解更好，则接受
            if new_fitness < fitness_values[i]:
                new_population.append(new_student)
                
                # 更新精英解
                if new_fitness < self.best_fitness:
                    self.best_solution = new_student.copy()
                    self.best_fitness = new_fitness
                    
                    # 更新精英池
                    if new_fitness < self.elite_fitness:
                        self.elite_solution = new_student.copy()
                        self.elite_fitness = new_fitness
                        self.update_elite_pool(new_student.copy(), new_fitness)
            else:
                new_population.append(student)
        
        return new_population
    
    def update_elite_pool(self, solution, fitness):
        """更新精英池"""
        # 检查解是否已经在精英池中
        for i, elite_solution in enumerate(self.elite_pool):
            if np.array_equal(solution, elite_solution):
                return
        
        # 检查精英池是否已满
        if len(self.elite_pool) < self.elite_pool_size:
            self.elite_pool.append(solution)
            self.elite_pool_fitness.append(fitness)
        else:
            # 找出精英池中最差的解
            worst_idx = np.argmax(self.elite_pool_fitness)
            worst_fitness = self.elite_pool_fitness[worst_idx]
            
            # 如果新解比最差的解好，则替换
            if fitness < worst_fitness:
                self.elite_pool[worst_idx] = solution
                self.elite_pool_fitness[worst_idx] = fitness
    
    def optimize(self):
        """执行TLBO+优化"""
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        fitness_values = [self.evaluate_fitness(solution) for solution in population]
        
        # 记录最佳解和精英解
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        self.elite_solution = self.best_solution.copy()
        self.elite_fitness = self.best_fitness
        
        # 初始化精英池
        self.elite_pool = [self.best_solution.copy()]
        self.elite_pool_fitness = [self.best_fitness]
        
        # 初始化迭代历史
        self.history = [self.best_fitness]
        
        # 记录无改进迭代次数
        no_improvement_count = 0
        
        for iter_idx in range(self.max_iter):
            # 教师阶段
            population = self.teacher_phase(population, fitness_values, iter_idx)
            
            # 更新适应度值
            fitness_values = [self.evaluate_fitness(solution) for solution in population]
            
            # 学习者阶段
            population = self.learner_phase(population, fitness_values)
            
            # 更新适应度值
            fitness_values = [self.evaluate_fitness(solution) for solution in population]
            
            # 更新最佳解
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.best_fitness:
                self.best_solution = population[best_idx].copy()
                self.best_fitness = fitness_values[best_idx]
                no_improvement_count = 0
                
                # 更新精英解
                if self.best_fitness < self.elite_fitness:
                    self.elite_solution = self.best_solution.copy()
                    self.elite_fitness = self.best_fitness
            else:
                no_improvement_count += 1
            
            # 每10次迭代注入一次精英解，防止震荡影响收敛
            if iter_idx % 10 == 0:
                worst_idx = np.argmax(fitness_values)
                population[worst_idx] = self.elite_solution.copy()
                fitness_values[worst_idx] = self.elite_fitness
            
            # 记录历史
            self.history.append(self.elite_fitness)
            
            if self.verbose and (iter_idx + 1) % 10 == 0:
                print(f"Iteration {iter_idx + 1}/{self.max_iter}, Best fitness: {self.elite_fitness:.6f}")
        
        if self.verbose:
            print(f"Optimization completed. Best fitness: {self.elite_fitness:.6f}")
        
        return self.elite_solution, self.elite_fitness, self.history