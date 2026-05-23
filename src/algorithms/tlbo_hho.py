# src/algorithms/tlbo_hho.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .tlbo import TLBO
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel


class TLBOHHO(TLBO):
    """基于教学的优化算法与哈里斯鹰优化结合（TLBO-HHO）- 修复版"""
    
    def __init__(self, system_model: SystemModel,
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: Optional[AoIModel] = None,
                 qoe_model: Optional[QoEModel] = None,
                 fairness_model: Optional[FairnessModel] = None,
                 max_iter: int = 100,
                 population_size: int = 50,
                 w_energy: float = 0.15,
                 w_delay: float = 0.15,
                 w_aoi: float = 0.20,
                 w_qoe: float = 0.25,
                 w_fairness: float = 0.25,
                 hho_prob: float = 0.3,  # HHO阶段的概率
                 verbose: bool = False):
        """
        初始化TLBO-HHO算法（五目标优化版本）

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
        qoe_model : QoEModel, optional
            QoE模型实例，如果为None则不考虑QoE
        fairness_model : FairnessModel, optional
            公平性模型实例，如果为None则不考虑公平性
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
        w_qoe : float
            QoE的权重系数
        w_fairness : float
            公平性的权重系数
        hho_prob : float
            使用HHO更新策略的概率
        verbose : bool
            是否输出详细信息
        """
        super().__init__(system_model, delay_model, energy_model, aoi_model,
                         max_iter, population_size, w_energy, w_delay, w_aoi, verbose)

        # 五目标优化模型
        self.qoe_model = qoe_model
        self.fairness_model = fairness_model
        self.w_qoe = w_qoe
        self.w_fairness = w_fairness

        self.hho_prob = hho_prob
        self.elite_solution = None
        self.elite_fitness = float('inf')

        # 预设归一化因子，避免除零 - 与FPA保持一致
        self.energy_max = 1e5  # 100,000 J
        self.delay_max = 1e3   # 1,000 s
        self.aoi_max = 1e2     # 100 s

        # Debug计数器
        self._eval_count = 0
        self._debug_first_eval = True

    def evaluate_fitness(self, solution):
        """
        五目标适应度函数（与FPA保持一致）

        F(X) = w_E*E/E_norm + w_T*T/T_norm + w_AoI*AoI/AoI_norm
             + w_QoE*(1-QoE) + w_F*(1-Fairness) + Penalty

        Parameters:
        -----------
        solution : List[List[Union[int, float]]]
            解向量

        Returns:
        --------
        float
            适应度值（越小越好）
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
            valid_aoi_tasks = 0

            constraint_violations = 0

            # 计算能耗、时延和AoI
            for task in self.system_model.tasks:
                try:
                    # 计算延迟
                    delay = self.delay_model.calculate_total_delay(task)
                    if not np.isfinite(delay) or delay < 0:
                        return float('inf')
                    total_delay += delay

                    # 检查延迟约束
                    if delay > task.max_delay:
                        constraint_violations += 1

                    # 计算能耗
                    energy = self.energy_model.calculate_total_energy(task)
                    if not np.isfinite(energy) or energy < 0:
                        return float('inf')
                    total_energy += energy

                    # 计算AoI（如果考虑）
                    if self.aoi_model is not None and task.update_interval is not None:
                        aoi = self.aoi_model.calculate_average_aoi(task)
                        if np.isfinite(aoi) and aoi >= 0:
                            total_aoi += aoi
                            valid_aoi_tasks += 1

                            # 检查AoI约束
                            if hasattr(task, 'max_aoi') and task.max_aoi is not None:
                                if aoi > task.max_aoi:
                                    constraint_violations += 1

                except Exception:
                    return float('inf')

            # 检查总值有效性
            if not (np.isfinite(total_energy) and np.isfinite(total_delay)):
                return float('inf')

            # 动态更新归一化因子（与FPA保持一致）
            self.energy_max = max(self.energy_max, total_energy, 1.0)
            self.delay_max = max(self.delay_max, total_delay, 1.0)
            if valid_aoi_tasks > 0:
                self.aoi_max = max(self.aoi_max, total_aoi / valid_aoi_tasks, 1.0)

            # 归一化基础目标
            normalized_energy = total_energy / self.energy_max
            normalized_delay = total_delay / self.delay_max
            normalized_aoi = (total_aoi / valid_aoi_tasks / self.aoi_max
                            if valid_aoi_tasks > 0 else 0.0)

            # 计算QoE（如果有QoE模型）
            qoe = 0.0
            if self.qoe_model is not None:
                qoe = self.qoe_model.calculate_system_qoe(solution)

            # 计算公平性（如果有公平性模型）
            fairness = 0.0
            if self.fairness_model is not None:
                fairness = self.fairness_model.calculate_fairness(solution)

            # 约束惩罚（降低惩罚系数，与FPA保持一致的数量级）
            penalty = constraint_violations * 2.0

            # 五目标适应度函数
            # 能耗、时延、AoI最小化（正权重）
            # QoE、公平性最大化（转换为最小化：1-value）
            # 注意：QoE和Fairness已经在[0,1]范围内
            fitness = (self.w_energy * normalized_energy +
                      self.w_delay * normalized_delay +
                      self.w_aoi * normalized_aoi +
                      self.w_qoe * (1.0 - qoe) +  # 转换为最小化
                      self.w_fairness * (1.0 - fairness) +  # 转换为最小化
                      penalty)

            # Debug输出（仅打印第一次评估）
            if self._debug_first_eval:
                avg_aoi = total_aoi / valid_aoi_tasks if valid_aoi_tasks > 0 else 0.0
                print(f"[TLBO-HHO DEBUG] Energy: {total_energy:.2f}, Delay: {total_delay:.4f}, "
                      f"AoI: {avg_aoi:.4f}, QoE: {qoe:.4f}, Fairness: {fairness:.4f}, "
                      f"Violations: {constraint_violations}, Penalty: {penalty:.4f}, "
                      f"Fitness: {fitness:.6f}")
                self._debug_first_eval = False

            return fitness if np.isfinite(fitness) else float('inf')

        except Exception:
            return float('inf')

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