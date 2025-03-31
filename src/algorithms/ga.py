# src/algorithms/ga.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .base_algorithm import BaseAlgorithm
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel


class GA(BaseAlgorithm):
    """遗传算法（Genetic Algorithm）"""
    
    def __init__(self, system_model: SystemModel, 
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: Optional[AoIModel] = None,
                 max_iter: int = 100,
                 population_size: int = 50,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 w_energy: float = 0.4,
                 w_delay: float = 0.6,
                 w_aoi: float = 0.0,
                 verbose: bool = False):
        """
        初始化遗传算法
        
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
            种群大小
        crossover_prob : float
            交叉概率
        mutation_prob : float
            变异概率
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
        
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
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
        """初始化种群"""
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
    
    def selection(self, population, fitness_values):
        """
        选择算子：锦标赛选择
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
        fitness_values : List[float]
            当前种群的适应度值
            
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            选择后的种群
        """
        new_population = []
        
        for _ in range(self.population_size):
            # 随机选择两个解进行锦标赛
            idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
            
            # 选择适应度值较小的解（最小化问题）
            if fitness_values[idx1] < fitness_values[idx2]:
                new_population.append(population[idx1])
            else:
                new_population.append(population[idx2])
        
        return new_population
    
    def crossover(self, population):
        """
        交叉算子：均匀交叉
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
            
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            交叉后的种群
        """
        new_population = []
        
        # 随机打乱种群顺序
        np.random.shuffle(population)
        
        # 对相邻的两个解进行交叉
        for i in range(0, self.population_size, 2):
            parent1 = population[i]
            
            if i + 1 < self.population_size:
                parent2 = population[i + 1]
                
                # 根据交叉概率决定是否进行交叉
                if np.random.random() < self.crossover_prob:
                    # 均匀交叉：为每个任务随机选择父解
                    child1 = []
                    child2 = []
                    
                    for j in range(self.num_tasks):
                        if np.random.random() < 0.5:
                            child1.append(parent1[j])
                            child2.append(parent2[j])
                        else:
                            child1.append(parent2[j])
                            child2.append(parent1[j])
                    
                    new_population.append(child1)
                    new_population.append(child2)
                else:
                    # 不交叉，直接复制父解
                    new_population.append(parent1.copy())
                    new_population.append(parent2.copy())
            else:
                # 如果种群大小为奇数，最后一个解直接复制
                new_population.append(parent1.copy())
        
        return new_population
    
    def mutation(self, population):
        """
        变异算子
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
            
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            变异后的种群
        """
        new_population = []
        
        for individual in population:
            mutated_individual = individual.copy()
            
            # 对每个任务，根据变异概率决定是否变异
            for i in range(self.num_tasks):
                if np.random.random() < self.mutation_prob:
                    # 变异执行位置
                    if np.random.random() < 0.3:  # 30%的概率变异位置
                        loc_i = np.random.randint(self.loc_bounds[0], self.loc_bounds[1] + 1)
                        mutated_individual[i][0] = loc_i
                    
                    # 变异计算资源分配
                    if np.random.random() < 0.3:  # 30%的概率变异资源
                        f_i = np.random.uniform(self.freq_bounds[0], self.freq_bounds[1])
                        mutated_individual[i][1] = f_i
                    
                    # 如果考虑AoI，变异更新间隔
                    if self.consider_aoi and len(mutated_individual[i]) > 2:
                        if np.random.random() < 0.3:  # 30%的概率变异更新间隔
                            delta_i = np.random.uniform(self.update_interval_bounds[0], self.update_interval_bounds[1])
                            mutated_individual[i][2] = delta_i
            
            # 确保变异后的解满足约束
            mutated_individual = self.handle_constraints(mutated_individual)
            new_population.append(mutated_individual)
        
        return new_population
    
    def elitism(self, population, fitness_values, elite_size=1):
        """
        精英保留策略
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
        fitness_values : List[float]
            当前种群的适应度值
        elite_size : int
            保留的精英数量
            
        Returns:
        --------
        Tuple[List[List[List[Union[int, float]]]], List[float]]
            精英解和其适应度值
        """
        # 按适应度值排序
        sorted_indices = np.argsort(fitness_values)
        
        # 选择最好的elite_size个解
        elite_indices = sorted_indices[:elite_size]
        elite_solutions = [population[i] for i in elite_indices]
        elite_fitness = [fitness_values[i] for i in elite_indices]
        
        return elite_solutions, elite_fitness
    
    def optimize(self):
        """
        执行遗传算法优化
        
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
            # 选择
            population = self.selection(population, fitness_values)
            
            # 交叉
            population = self.crossover(population)
            
            # 变异
            population = self.mutation(population)
            
            # 评估新种群
            fitness_values = [self.evaluate_fitness(solution) for solution in population]
            
            # 精英保留
            elite_solutions, _ = self.elitism(population, fitness_values)
            
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