"""
改进混合麻雀搜索算法 (Improved Hybrid Sparrow Search Algorithm with Differential Evolution, IHSSA-DE)
结合Bernoulli混沌映射、微分进化、自适应t分布变异和动态惩罚的五目标优化算法
修订版：引入全环节贪婪选择机制（Greedy Selection）以解决约束丢失问题
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import t as t_dist
from .ssa import SSA
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel


class IHSSADE(SSA):
    """改进混合麻雀搜索算法 - 五目标优化 (修复版)"""

    def __init__(self, system_model: SystemModel,
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: Optional[AoIModel] = None,
                 qoe_model: Optional[QoEModel] = None,
                 fairness_model: Optional[FairnessModel] = None,
                 max_iter: int = 150,
                 population_size: int = 50,
                 w_energy: float = 0.15,
                 w_delay: float = 0.15,
                 w_aoi: float = 0.20,
                 w_qoe: float = 0.25,
                 w_fairness: float = 0.25,
                 PD: float = 0.2,
                 SD: float = 0.1,
                 ST: float = 0.8,
                 F: float = 0.5,    # 调低F值，减少破坏性
                 CR: float = 0.9,   # 调高CR，保留更多优良基因
                 lambda_chaos: float = 0.4,
                 base_penalty: float = 1.0,  # 增大基础惩罚
                 penalty_alpha: float = 2.0,
                 verbose: bool = False):
        """
        初始化IHSSA-DE算法
        """
        super().__init__(system_model, delay_model, energy_model,
                        aoi_model, qoe_model, fairness_model,
                        max_iter, population_size,
                        w_energy, w_delay, w_aoi, w_qoe, w_fairness,
                        PD, SD, ST, verbose)

        # DE参数
        self.F = F    # 缩放因子
        self.CR = CR  # 交叉概率

        # Bernoulli混沌参数
        self.lambda_chaos = lambda_chaos

        # 动态惩罚参数
        self.base_penalty = base_penalty
        self.penalty_alpha = penalty_alpha

        # 当前迭代（用于动态惩罚）
        self.current_iteration = 0

    def bernoulli_chaotic_map(self, z: float) -> float:
        """Bernoulli混沌映射"""
        if 0 < z <= 1 - self.lambda_chaos:
            z_new = z / (1 - self.lambda_chaos)
        else:
            z_new = (z - 1 + self.lambda_chaos) / self.lambda_chaos
        return z_new % 1.0

    def map_chaos_to_solution(self, chaos_vector: np.ndarray) -> List:
        """将混沌序列映射到决策变量"""
        solution = []
        offset = 0

        for task_idx in range(self.num_tasks):
            # 位置
            loc_chaos = chaos_vector[offset]
            loc_i = int(loc_chaos * (self.loc_bounds[1] - self.loc_bounds[0] + 1)) + self.loc_bounds[0]
            loc_i = np.clip(loc_i, self.loc_bounds[0], self.loc_bounds[1])

            # 频率
            freq_chaos = chaos_vector[offset + 1]
            f_i = self.freq_bounds[0] + freq_chaos * (self.freq_bounds[1] - self.freq_bounds[0])

            offset += 2

            if self.consider_aoi:
                # 更新间隔
                delta_chaos = chaos_vector[offset]
                delta_i = self.update_interval_bounds[0] + delta_chaos * (
                    self.update_interval_bounds[1] - self.update_interval_bounds[0])
                solution.append([loc_i, f_i, delta_i])
                offset += 1
            else:
                solution.append([loc_i, f_i])

        return solution

    def initialize_population(self):
        """改进策略1: Bernoulli混沌映射初始化"""
        population = []
        
        # 确定维度
        if self.consider_aoi:
            dim = self.num_tasks * 3
        else:
            dim = self.num_tasks * 2

        # 初始化混沌变量
        z = np.random.rand(dim)

        for _ in range(self.population_size):
            # 应用Bernoulli混沌映射
            for d in range(dim):
                z[d] = self.bernoulli_chaotic_map(z[d])
            
            # 映射到决策变量
            solution = self.map_chaos_to_solution(z.copy())
            
            # 处理约束
            solution = self.handle_constraints(solution)
            population.append(solution)

        return population

    def calculate_dynamic_penalty(self, constraint_violations: int) -> float:
        """改进策略4: 动态约束惩罚机制"""
        if constraint_violations == 0:
            return 0.0
        
        # 迭代进度
        progress = self.current_iteration / self.max_iter if self.max_iter > 0 else 0.0
        
        # 动态惩罚因子：随迭代指数增长，强迫后期收敛到可行域
        penalty_factor = self.base_penalty * ((1 + progress * 2) ** self.penalty_alpha)
        
        return penalty_factor * constraint_violations

    def evaluate_fitness(self, solution):
        """重写适应度函数，使用动态惩罚"""
        try:
            if not solution or len(solution) != self.num_tasks:
                return float('inf')

            # 注意：此处不再次调用handle_constraints，避免改变传入的solution结构
            # handle_constraints应该在产生解的时候调用
            
            # 将解应用到系统模型
            self.system_model.apply_solution(solution)

            total_energy = 0.0
            total_delay = 0.0
            total_aoi = 0.0
            valid_aoi_tasks = 0
            constraint_violations = 0

            # 计算各项指标
            for task in self.system_model.tasks:
                try:
                    # 时延
                    delay = self.delay_model.calculate_total_delay(task)
                    if not np.isfinite(delay) or delay < 0: return float('inf')
                    total_delay += delay
                    if delay > task.max_delay: constraint_violations += 1

                    # 能耗
                    energy = self.energy_model.calculate_total_energy(task)
                    if not np.isfinite(energy) or energy < 0: return float('inf')
                    total_energy += energy

                    # AoI
                    if self.aoi_model is not None and task.update_interval is not None:
                        aoi = self.aoi_model.calculate_average_aoi(task)
                        if np.isfinite(aoi) and aoi >= 0:
                            total_aoi += aoi
                            valid_aoi_tasks += 1
                            if hasattr(task, 'max_aoi') and task.max_aoi is not None:
                                if aoi > task.max_aoi: constraint_violations += 1
                except Exception:
                    return float('inf')

            if not (np.isfinite(total_energy) and np.isfinite(total_delay)):
                return float('inf')

            # 动态更新归一化因子
            self.energy_max = max(self.energy_max, total_energy, 1.0)
            self.delay_max = max(self.delay_max, total_delay, 1.0)
            if valid_aoi_tasks > 0:
                self.aoi_max = max(self.aoi_max, total_aoi / valid_aoi_tasks, 1.0)

            normalized_energy = total_energy / self.energy_max
            normalized_delay = total_delay / self.delay_max
            normalized_aoi = (total_aoi / valid_aoi_tasks / self.aoi_max if valid_aoi_tasks > 0 else 0.0)

            # QoE & Fairness
            qoe = self.qoe_model.calculate_system_qoe(solution) if self.qoe_model else 0.0
            fairness = self.fairness_model.calculate_fairness(solution) if self.fairness_model else 0.0

            # 动态惩罚
            penalty = self.calculate_dynamic_penalty(constraint_violations)

            fitness = (self.w_energy * normalized_energy +
                      self.w_delay * normalized_delay +
                      self.w_aoi * normalized_aoi +
                      self.w_qoe * (1.0 - qoe) +
                      self.w_fairness * (1.0 - fairness) +
                      penalty)

            return fitness if np.isfinite(fitness) else float('inf')

        except Exception:
            return float('inf')

    def de_enhanced_scrounger_update(self, scrounger_idx: int, population: List, 
                                      best_solution: List) -> List:
        """改进策略2: 微分进化（DE）变异"""
        try:
            current = population[scrounger_idx]
            candidates = [i for i in range(len(population)) if i != scrounger_idx]
            if len(candidates) < 2: return current

            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            best_array = np.array(best_solution, dtype=float)
            r1_array = np.array(population[r1], dtype=float)
            r2_array = np.array(population[r2], dtype=float)
            
            # DE/best/1 变异策略
            mutant = best_array + self.F * (r1_array - r2_array)
            
            # 二项交叉
            current_array = np.array(current, dtype=float)
            trial = current_array.copy()
            
            # 随机选择维度进行交叉
            j_rand = np.random.randint(len(current))
            for j in range(len(current)):
                if np.random.random() < self.CR or j == j_rand:
                    trial[j] = mutant[j]
            
            return self.handle_constraints(trial.tolist())

        except Exception:
            return current

    def adaptive_t_distribution_mutation(self, best_solution: List) -> List:
        """改进策略3: 自适应t分布变异"""
        try:
            min_df = 1
            max_df = 30
            df = min_df + self.current_iteration * (max_df - min_df) / self.max_iter
            
            best_array = np.array(best_solution, dtype=float)
            perturbation = t_dist.rvs(df, size=best_array.shape)
            
            # 扰动幅度
            scale = 0.1 * (1 - self.current_iteration / self.max_iter)
            new_solution = best_array + scale * best_array * perturbation
            
            return self.handle_constraints(new_solution.tolist())
        except Exception:
            return best_solution

    def optimize(self):
        """执行IHSSA-DE优化（引入贪婪选择机制）"""
        
        # 1. 混沌初始化
        population = self.initialize_population()
        self.current_iteration = 0
        
        # 评估初始种群
        fitness_values = [self.evaluate_fitness(sol) for sol in population]
        
        # 找初始最优
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices: return None, float('inf'), []
        
        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        # 记录最差解用于标准SSA公式（虽然DE策略可能不完全依赖它，但为了兼容性保留）
        worst_idx = max(valid_indices, key=lambda i: fitness_values[i])
        worst_solution = population[worst_idx]
        worst_fitness = fitness_values[worst_idx]
        
        self.history = [best_fitness]

        # 3. 主迭代循环
        for iteration in range(self.max_iter):
            try:
                self.current_iteration = iteration
                
                # 对种群按适应度排序，确定角色
                sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
                
                n_producers = max(1, int(self.population_size * self.PD))
                n_scouts = max(1, int(self.population_size * self.SD))
                
                # 角色索引集合
                producer_set = set(sorted_indices[:n_producers])
                scout_set = set(sorted_indices[-n_scouts:])
                # 其余为加入者
                
                # 创建新一代种群（为了异步更新，也可以直接在原种群操作，这里采用贪婪选择后直接更新原种群）
                # 注意：DE通常建议同步更新，但贪婪选择配合异步更新收敛更快
                
                for idx in range(self.population_size):
                    old_fitness = fitness_values[idx]
                    candidate = None
                    
                    # --- 生成候选解 ---
                    if idx in producer_set:
                        # 发现者更新
                        candidate = self.update_producer(population[idx], iteration, best_solution)
                    
                    elif idx in scout_set:
                        # 侦察者更新
                        candidate = self.update_scout(population[idx], best_solution, worst_solution,
                                                    old_fitness, best_fitness, worst_fitness)
                    else:
                        # 加入者更新 (使用DE策略)
                        candidate = self.de_enhanced_scrounger_update(idx, population, best_solution)
                    
                    # --- 贪婪选择 (Greedy Selection) ---
                    # 只有当新解更好时，才更新位置
                    if candidate is not None:
                        new_fitness = self.evaluate_fitness(candidate)
                        
                        if np.isfinite(new_fitness) and new_fitness < old_fitness:
                            population[idx] = candidate
                            fitness_values[idx] = new_fitness
                            
                            # 实时更新全局最优
                            if new_fitness < best_fitness:
                                best_solution = candidate
                                best_fitness = new_fitness

                # 更新最差解（用于下一轮计算）
                valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
                if valid_indices:
                    worst_idx = max(valid_indices, key=lambda i: fitness_values[i])
                    worst_solution = population[worst_idx]
                    worst_fitness = fitness_values[worst_idx]

                # d. t分布变异（尝试改进全局最优）
                mutated_solution = self.adaptive_t_distribution_mutation(best_solution)
                mutated_fitness = self.evaluate_fitness(mutated_solution)
                
                if np.isfinite(mutated_fitness) and mutated_fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = mutated_fitness
                    # 替换最差个体以保持种群数量
                    population[worst_idx] = mutated_solution
                    fitness_values[worst_idx] = mutated_fitness

                self.history.append(best_fitness)
                
                if self.verbose and (iteration + 1) % 10 == 0:
                    print(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {best_fitness:.6f}")

            except Exception as e:
                if self.verbose: print(f"Error in iteration {iteration}: {e}")
                self.history.append(best_fitness)

        if self.verbose:
            print(f"Optimization completed. Best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness, self.history