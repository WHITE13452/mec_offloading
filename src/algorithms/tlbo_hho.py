# src/algorithms/tlbo_hho.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .tlbo import TLBO
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel


class TLBOHHO(TLBO):
    """基于教学的优化算法与哈里斯鹰优化结合（TLBO-HHO）- 修复版"""
    
    def __init__(self, system_model: SystemModel, 
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: Optional[AoIModel] = None,
                 max_iter: int = 100,
                 population_size: int = 50,
                 w_energy: float = 0.4,
                 w_delay: float = 0.6,
                 w_aoi: float = 0.0,
                 hho_prob: float = 0.3,  # HHO阶段的概率
                 verbose: bool = False):
        """
        初始化TLBO-HHO算法
        
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
        hho_prob : float
            使用HHO更新策略的概率
        verbose : bool
            是否输出详细信息
        """
        super().__init__(system_model, delay_model, energy_model, aoi_model,
                         max_iter, population_size, w_energy, w_delay, w_aoi, verbose)
        
        self.hho_prob = hho_prob
        self.elite_solution = None
        self.elite_fitness = float('inf')
        
        # 预设归一化因子，避免除零 - 这是修复的关键部分
        self.energy_max = 1e5  # 100,000 J
        self.delay_max = 1e3   # 1,000 s
        self.aoi_max = 1e2     # 100 s
    
    def hho_update(self, population, fitness_values, iteration):
        """
        哈里斯鹰优化算法更新策略 - 修复版
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
        fitness_values : List[float]
            当前种群的适应度值
        iteration : int
            当前迭代次数
            
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            更新后的种群
        """
        new_population = []
        
        try:
            # 找出有效的最好的解（兔子/猎物）
            valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
            if not valid_indices:
                return population
            
            best_idx = min(valid_indices, key=lambda i: fitness_values[i])
            best_solution = population[best_idx]
            
            # 迭代控制参数E - 随迭代减小
            E = 2 * (1 - iteration / self.max_iter)
            
            for i, hawk in enumerate(population):
                try:
                    hawk_array = np.array(hawk)
                    best_array = np.array(best_solution)
                    
                    # 随机值
                    r1 = np.random.random()
                    r2 = np.random.random()
                    r3 = np.random.random()
                    r4 = np.random.random()
                    r5 = np.random.random()
                    
                    # 能量E0随机值在[-1,1]
                    E0 = 2 * r1 - 1  
                    
                    # 转义跳跃强度J
                    J = 2 * (1 - r2)
                    
                    # 策略选择的概率q
                    q = np.random.random()
                    
                    if np.abs(E) >= 1:  # 探索阶段
                        if q < 0.5:  # 策略1: 随机游走
                            # 随机选择另一个解
                            random_idx = np.random.randint(0, self.population_size)
                            random_hawk = population[random_idx]
                            random_array = np.array(random_hawk)
                            
                            # 更新位置
                            new_hawk = random_array - r3 * np.abs(random_array - 2 * r4 * hawk_array)
                        else:  # 策略2: 基于差异的随机游走
                            # 计算种群平均位置
                            mean_array = np.mean(population, axis=0)
                            
                            # 更新位置
                            new_hawk = (best_array - mean_array) - r3 * (r4 + (r5 * (2 * r2 - 1))) * np.abs(best_array - hawk_array)
                    else:  # 开发阶段
                        # 飞行能量
                        E1 = 2 * E0 * r2
                        
                        if np.abs(E1) >= 1 and q < 0.5:  # 硬包围
                            new_hawk = best_array - E * np.abs(J * best_array - hawk_array)
                        elif np.abs(E1) >= 1 and q >= 0.5:  # 软包围
                            delta = best_array - hawk_array
                            new_hawk = delta - E * np.abs(J * best_array - hawk_array)
                        elif np.abs(E1) < 1 and q < 0.5:  # 软包围渐进快速俯冲
                            new_hawk = best_array - E * np.abs(J * best_array - hawk_array)
                        else:  # 硬包围渐进快速俯冲
                            LF = 0.01 * r3 * (1 / (r4 + 1e-10))  # 莱维飞行
                            Y = best_array - E * np.abs(J * best_array - hawk_array)
                            Z = Y + r4 * LF
                            
                            # 使用Y和Z中较好的解
                            Y_list = Y.tolist()
                            Z_list = Z.tolist()
                            
                            # 确保Y和Z满足约束
                            Y_list = self.handle_constraints(Y_list)
                            Z_list = self.handle_constraints(Z_list)
                            
                            Y_fitness = self.evaluate_fitness(Y_list)
                            Z_fitness = self.evaluate_fitness(Z_list)
                            
                            if np.isfinite(Y_fitness) and np.isfinite(Z_fitness):
                                if Y_fitness < Z_fitness:
                                    new_hawk = Y
                                else:
                                    new_hawk = Z
                            elif np.isfinite(Y_fitness):
                                new_hawk = Y
                            elif np.isfinite(Z_fitness):
                                new_hawk = Z
                            else:
                                new_hawk = hawk_array  # 保持原位置
                    
                    # 处理约束
                    new_hawk_list = self.handle_constraints(new_hawk.tolist())
                    
                    # 评估新解
                    new_fitness = self.evaluate_fitness(new_hawk_list)
                    
                    # 如果新解更好且有效，则接受
                    if np.isfinite(new_fitness) and new_fitness < fitness_values[i]:
                        new_population.append(new_hawk_list)
                    else:
                        new_population.append(hawk)
                        
                except Exception:
                    # 如果出错，保留原解
                    new_population.append(hawk)
            
            return new_population
            
        except Exception:
            # 如果整个过程出错，返回原种群
            return population

    def adaptive_hho_prob(self, iteration, max_iterations, min_prob=0.1, max_prob=0.6):
        """动态调整HHO阶段的概率"""
        # 早期迭代更多使用HHO进行全局搜索，后期更多使用TLBO进行精细优化
        return max_prob - (max_prob - min_prob) * (iteration / max_iterations)
    
    def optimize(self):
        """
        执行TLBO-HHO优化 - 修复版
        
        Returns:
        --------
        Tuple[List[List[Union[int, float]]], float, List[float]]
            最优解、最优适应度值和迭代历史
        """
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        fitness_values = [self.evaluate_fitness(solution) for solution in population]
        
        # 找到有效的最佳解
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            if self.verbose:
                print("Warning: No valid solutions found in initial population!")
            return None, float('inf'), []
        
        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        self.best_solution = population[best_idx]
        self.best_fitness = fitness_values[best_idx]
        self.elite_solution = self.best_solution
        self.elite_fitness = self.best_fitness
        
        # 初始化迭代历史
        self.history = [self.best_fitness]
        
        for iter_idx in range(self.max_iter):
            try:
                # 动态调整HHO阶段的概率
                current_hho_prob = self.adaptive_hho_prob(iter_idx, self.max_iter)
                
                # 以一定概率选择使用HHO或TLBO
                if np.random.random() < current_hho_prob:
                    # HHO更新
                    population = self.hho_update(population, fitness_values, iter_idx)
                else:
                    # TLBO更新: 教师阶段
                    population = self.teacher_phase(population, fitness_values)
                    
                    # 更新适应度值
                    fitness_values = [self.evaluate_fitness(solution) for solution in population]
                    
                    # 学习者阶段
                    population = self.learner_phase(population, fitness_values)
                
                # 更新适应度值
                fitness_values = [self.evaluate_fitness(solution) for solution in population]
                
                # 更新最佳解
                valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
                if valid_indices:
                    best_idx = min(valid_indices, key=lambda i: fitness_values[i])
                    if fitness_values[best_idx] < self.best_fitness:
                        self.best_solution = population[best_idx]
                        self.best_fitness = fitness_values[best_idx]
                
                # 精英保留策略
                if self.best_fitness < self.elite_fitness:
                    self.elite_solution = self.best_solution
                    self.elite_fitness = self.best_fitness
                else:
                    # 将精英解替换种群中的最差解
                    if valid_indices:
                        worst_idx = max(range(len(fitness_values)), 
                                      key=lambda i: fitness_values[i] if np.isfinite(fitness_values[i]) else -float('inf'))
                        population[worst_idx] = self.elite_solution
                        fitness_values[worst_idx] = self.elite_fitness
                
                # 记录历史
                self.history.append(self.elite_fitness)
                
                if self.verbose and (iter_idx + 1) % 10 == 0:
                    print(f"Iteration {iter_idx + 1}/{self.max_iter}, Best fitness: {self.elite_fitness:.6f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error in iteration {iter_idx}: {e}")
                self.history.append(self.elite_fitness)
        
        if self.verbose:
            print(f"Optimization completed. Best fitness: {self.elite_fitness:.6f}")
        
        return self.elite_solution, self.elite_fitness, self.history