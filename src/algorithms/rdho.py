"""
RDHO算法 (RIME-Dung Beetle Hybrid Optimizer)
融合RIME和DBO的混合优化算法，专为五目标MEC任务卸载优化设计
"""
import numpy as np
from typing import List, Tuple, Optional
from .base_algorithm import BaseAlgorithm
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel


class RDHO(BaseAlgorithm):
    """RDHO混合算法 - 五目标优化"""

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
                 producer_ratio: float = 0.2,
                 follower_ratio: float = 0.7,
                 scout_ratio: float = 0.1,
                 elite_ratio: float = 0.1,
                 base_penalty: float = 1.0,
                 penalty_alpha: float = 2.0,
                 verbose: bool = False):
        """
        初始化RDHO算法

        Args:
            system_model: 系统模型
            delay_model: 时延模型
            energy_model: 能耗模型
            aoi_model: AoI模型
            qoe_model: QoE模型
            fairness_model: 公平性模型
            max_iter: 最大迭代次数
            population_size: 种群规模
            w_energy: 能耗权重
            w_delay: 时延权重
            w_aoi: AoI权重
            w_qoe: QoE权重
            w_fairness: 公平性权重
            producer_ratio: 生产者比例
            follower_ratio: 跟随者比例
            scout_ratio: 侦察者比例
            elite_ratio: 精英解比例
            base_penalty: 基础惩罚系数
            penalty_alpha: 惩罚增长指数
            verbose: 是否输出详细信息
        """
        super().__init__(system_model, max_iter, population_size, verbose)

        # 保存模型引用
        self.delay_model = delay_model
        self.energy_model = energy_model
        self.aoi_model = aoi_model
        self.qoe_model = qoe_model
        self.fairness_model = fairness_model

        # 五目标权重
        self.w_energy = w_energy
        self.w_delay = w_delay
        self.w_aoi = w_aoi
        self.w_qoe = w_qoe
        self.w_fairness = w_fairness

        # 角色比例
        self.producer_ratio = producer_ratio
        self.follower_ratio = follower_ratio
        self.scout_ratio = scout_ratio
        self.elite_ratio = elite_ratio

        # 动态惩罚参数
        self.base_penalty = base_penalty
        self.penalty_alpha = penalty_alpha

        # 决策变量边界已在父类中定义，无需重复设置
        # self.num_tasks, self.loc_bounds, self.freq_bounds 已由 BaseAlgorithm 设置

        # AoI相关（consider_aoi已在父类中设置）
        # self.consider_aoi 已由 BaseAlgorithm 设置
        # self.update_interval_bounds 已由 BaseAlgorithm 设置

        # 动态归一化因子
        self.energy_max = 1.0
        self.delay_max = 1.0
        self.aoi_max = 1.0

        # 优化历史
        self.history = []

        # 当前迭代
        self.current_iteration = 0

    def calculate_dynamic_penalty(self, constraint_violations: int) -> float:
        """
        动态约束惩罚机制（分层惩罚）

        Args:
            constraint_violations: 约束违反次数

        Returns:
            惩罚值
        """
        if constraint_violations == 0:
            return 0.0

        # 迭代进度
        progress = self.current_iteration / self.max_iter if self.max_iter > 0 else 0.0

        # 动态惩罚因子（随迭代指数增长）
        penalty_factor = self.base_penalty * ((1 + progress * 2) ** self.penalty_alpha)

        return penalty_factor * constraint_violations

    def evaluate_fitness(self, solution) -> float:
        """
        评估解的适应度（五目标优化 + 动态惩罚）

        Args:
            solution: 决策解

        Returns:
            适应度值（越小越好）
        """
        try:
            if not solution or len(solution) != self.num_tasks:
                return float('inf')

            self.system_model.apply_solution(solution)

            total_energy = 0.0
            total_delay = 0.0
            total_aoi = 0.0
            valid_aoi_tasks = 0
            constraint_violations = 0

            for task in self.system_model.tasks:
                try:
                    # 时延
                    delay = self.delay_model.calculate_total_delay(task)
                    if not np.isfinite(delay) or delay < 0:
                        return float('inf')
                    total_delay += delay
                    if delay > task.max_delay:
                        constraint_violations += 1

                    # 能耗
                    energy = self.energy_model.calculate_total_energy(task)
                    if not np.isfinite(energy) or energy < 0:
                        return float('inf')
                    total_energy += energy

                    # AoI
                    if self.aoi_model is not None and task.update_interval is not None:
                        aoi = self.aoi_model.calculate_average_aoi(task)
                        if np.isfinite(aoi) and aoi >= 0:
                            total_aoi += aoi
                            valid_aoi_tasks += 1
                            if hasattr(task, 'max_aoi') and task.max_aoi is not None:
                                if aoi > task.max_aoi:
                                    constraint_violations += 1
                except Exception:
                    return float('inf')

            if not (np.isfinite(total_energy) and np.isfinite(total_delay)):
                return float('inf')

            # 动态更新归一化因子
            self.energy_max = max(self.energy_max, total_energy, 1.0)
            self.delay_max = max(self.delay_max, total_delay, 1.0)
            if valid_aoi_tasks > 0:
                self.aoi_max = max(self.aoi_max, total_aoi / valid_aoi_tasks, 1.0)

            # 归一化
            normalized_energy = total_energy / self.energy_max
            normalized_delay = total_delay / self.delay_max
            normalized_aoi = (total_aoi / valid_aoi_tasks / self.aoi_max
                            if valid_aoi_tasks > 0 else 0.0)

            # QoE和公平性
            qoe = self.qoe_model.calculate_system_qoe(solution) if self.qoe_model else 0.0
            fairness = self.fairness_model.calculate_fairness(solution) if self.fairness_model else 0.0

            # 动态惩罚
            penalty = self.calculate_dynamic_penalty(constraint_violations)

            # 五目标适应度
            fitness = (self.w_energy * normalized_energy +
                      self.w_delay * normalized_delay +
                      self.w_aoi * normalized_aoi +
                      self.w_qoe * (1.0 - qoe) +
                      self.w_fairness * (1.0 - fairness) +
                      penalty)

            return fitness if np.isfinite(fitness) else float('inf')

        except Exception:
            return float('inf')

    def initialize_population(self) -> List:
        """
        双源混合初始化（50% RIME软凝华 + 50% DBO滚球）

        Returns:
            初始种群
        """
        population = []
        half_size = self.population_size // 2

        # 50% RIME软凝华风格初始化（高斯分布）
        for _ in range(half_size):
            solution = []
            for task_idx in range(self.num_tasks):
                # 高斯分布采样
                loc = int(np.clip(np.round(np.random.randn() + 1),
                                 self.loc_bounds[0], self.loc_bounds[1]))

                freq_mean = (self.freq_bounds[0] + self.freq_bounds[1]) / 2
                freq_std = (self.freq_bounds[1] - self.freq_bounds[0]) / 6
                freq = np.clip(np.random.randn() * freq_std + freq_mean,
                              self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi:
                    delta_mean = (self.update_interval_bounds[0] + self.update_interval_bounds[1]) / 2
                    delta_std = (self.update_interval_bounds[1] - self.update_interval_bounds[0]) / 6
                    delta = np.clip(np.random.randn() * delta_std + delta_mean,
                                   self.update_interval_bounds[0],
                                   self.update_interval_bounds[1])
                    solution.append([loc, freq, delta])
                else:
                    solution.append([loc, freq])

            solution = self.handle_constraints(solution)
            population.append(solution)

        # 50% DBO滚球风格初始化（均匀分布）
        for _ in range(self.population_size - half_size):
            solution = []
            for task_idx in range(self.num_tasks):
                loc = np.random.randint(self.loc_bounds[0], self.loc_bounds[1] + 1)
                freq = np.random.uniform(self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi:
                    delta = np.random.uniform(self.update_interval_bounds[0],
                                             self.update_interval_bounds[1])
                    solution.append([loc, freq, delta])
                else:
                    solution.append([loc, freq])

            solution = self.handle_constraints(solution)
            population.append(solution)

        return population

    def producer_update_fusion(self, individual: List, best_solution: List,
                               worst_solution: List, iteration: int) -> List:
        """
        生产者更新：RIME软凝华 + DBO滚球融合

        公式: x_new = w·RIME_component + (1-w)·DBO_component
        其中: w = 0.5 + 0.3·cos(π·t/T)

        Args:
            individual: 当前个体
            best_solution: 全局最优解
            worst_solution: 最差解
            iteration: 当前迭代次数

        Returns:
            新解
        """
        try:
            individual_array = np.array(individual, dtype=float)
            best_array = np.array(best_solution, dtype=float)
            worst_array = np.array(worst_solution, dtype=float)

            # 自适应融合权重
            w = 0.5 + 0.3 * np.cos(np.pi * iteration / self.max_iter)

            # RIME软凝华组件
            h = 2.0 * (1.0 - iteration / self.max_iter)
            beta = np.random.randn()
            theta = 2 * np.pi * np.random.rand()

            # DBO滚球组件
            alpha = 1.0 - iteration / self.max_iter
            k = np.random.uniform(-1, 1)
            b = np.random.rand()

            new_solution = []
            for i in range(self.num_tasks):
                # 位置融合
                loc_range = self.loc_bounds[1] - self.loc_bounds[0]
                rime_loc = (best_array[i][0] +
                           beta * np.cos(theta) * h * loc_range)

                dbo_loc = (individual_array[i][0] +
                          alpha * k * (individual_array[i][0] - self.loc_bounds[0]) +
                          b * np.abs(individual_array[i][0] - worst_array[i][0]))

                loc_new = w * rime_loc + (1 - w) * dbo_loc
                loc_new = int(np.clip(np.round(loc_new),
                                     self.loc_bounds[0], self.loc_bounds[1]))

                # 频率融合
                freq_range = self.freq_bounds[1] - self.freq_bounds[0]
                rime_freq = (best_array[i][1] +
                            beta * np.cos(theta + np.pi/4) * h * freq_range)

                dbo_freq = (individual_array[i][1] +
                           alpha * k * (individual_array[i][1] - self.freq_bounds[0]) +
                           b * np.abs(individual_array[i][1] - worst_array[i][1]))

                freq_new = w * rime_freq + (1 - w) * dbo_freq
                freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi and len(individual[i]) > 2:
                    delta_range = (self.update_interval_bounds[1] -
                                  self.update_interval_bounds[0])
                    rime_delta = (best_array[i][2] +
                                 beta * np.cos(theta + np.pi/2) * h * delta_range)

                    dbo_delta = (individual_array[i][2] +
                                alpha * k * (individual_array[i][2] - self.update_interval_bounds[0]) +
                                b * np.abs(individual_array[i][2] - worst_array[i][2]))

                    delta_new = w * rime_delta + (1 - w) * dbo_delta
                    delta_new = np.clip(delta_new,
                                       self.update_interval_bounds[0],
                                       self.update_interval_bounds[1])
                    new_solution.append([loc_new, freq_new, delta_new])
                else:
                    new_solution.append([loc_new, freq_new])

            return self.handle_constraints(new_solution)

        except Exception:
            return individual

    def follower_update_hybrid(self, individual: List, best_solution: List,
                               iteration: int) -> List:
        """
        跟随者更新：RIME硬凝华穿刺 or DBO觅食

        Args:
            individual: 当前个体
            best_solution: 全局最优解
            iteration: 当前迭代次数

        Returns:
            新解
        """
        try:
            individual_array = np.array(individual, dtype=float)
            best_array = np.array(best_solution, dtype=float)

            # 穿刺概率
            E = 2.0 * np.exp(-((4.0 * iteration / self.max_iter) ** 2))

            if np.random.rand() < E:
                # RIME硬凝华穿刺
                new_solution = individual_array.copy()
                j_rand = np.random.randint(self.num_tasks)
                new_solution[j_rand] = best_array[j_rand].copy()
            else:
                # DBO觅食
                C1 = np.random.rand()
                C2 = np.random.rand()

                new_solution = []
                for i in range(self.num_tasks):
                    loc_new = (best_array[i][0] +
                              C1 * (individual_array[i][0] - self.loc_bounds[0]) +
                              C2 * (individual_array[i][0] - self.loc_bounds[1]))
                    loc_new = int(np.clip(np.round(loc_new),
                                         self.loc_bounds[0], self.loc_bounds[1]))

                    freq_new = (best_array[i][1] +
                               C1 * (individual_array[i][1] - self.freq_bounds[0]) +
                               C2 * (individual_array[i][1] - self.freq_bounds[1]))
                    freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                    if self.consider_aoi and len(individual[i]) > 2:
                        delta_new = (best_array[i][2] +
                                    C1 * (individual_array[i][2] - self.update_interval_bounds[0]) +
                                    C2 * (individual_array[i][2] - self.update_interval_bounds[1]))
                        delta_new = np.clip(delta_new,
                                           self.update_interval_bounds[0],
                                           self.update_interval_bounds[1])
                        new_solution.append([loc_new, freq_new, delta_new])
                    else:
                        new_solution.append([loc_new, freq_new])

            return self.handle_constraints(new_solution.tolist() if isinstance(new_solution, np.ndarray)
                                          else new_solution)

        except Exception:
            return individual

    def scout_update_cauchy(self, individual: List, best_solution: List,
                           local_best: List, fitness: float,
                           best_fitness: float) -> List:
        """
        侦察者更新：DBO偷窃 or Cauchy变异

        Args:
            individual: 当前个体
            best_solution: 全局最优解
            local_best: 局部最优解
            fitness: 当前适应度
            best_fitness: 最优适应度

        Returns:
            新解
        """
        try:
            individual_array = np.array(individual, dtype=float)
            best_array = np.array(best_solution, dtype=float)
            local_best_array = np.array(local_best, dtype=float)

            if fitness > best_fitness:
                # DBO偷窃
                theta = np.random.uniform(-np.pi / 4, np.pi / 4,
                                         size=individual_array.shape)

                new_solution = []
                for i in range(self.num_tasks):
                    loc_new = (local_best_array[i][0] +
                              np.tan(theta[i][0]) * np.abs(individual_array[i][0] - local_best_array[i][0]))
                    loc_new = int(np.clip(np.round(loc_new),
                                         self.loc_bounds[0], self.loc_bounds[1]))

                    freq_new = (local_best_array[i][1] +
                               np.tan(theta[i][1]) * np.abs(individual_array[i][1] - local_best_array[i][1]))
                    freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                    if self.consider_aoi and len(individual[i]) > 2:
                        delta_new = (local_best_array[i][2] +
                                    np.tan(theta[i][2]) * np.abs(individual_array[i][2] - local_best_array[i][2]))
                        delta_new = np.clip(delta_new,
                                           self.update_interval_bounds[0],
                                           self.update_interval_bounds[1])
                        new_solution.append([loc_new, freq_new, delta_new])
                    else:
                        new_solution.append([loc_new, freq_new])

                return self.handle_constraints(new_solution)
            else:
                # Cauchy变异
                scale = 0.1 * (1.0 - self.current_iteration / self.max_iter)
                cauchy_noise = np.random.standard_cauchy(size=best_array.shape)

                new_solution = []
                for i in range(self.num_tasks):
                    loc_new = best_array[i][0] + scale * cauchy_noise[i][0]
                    loc_new = int(np.clip(np.round(loc_new),
                                         self.loc_bounds[0], self.loc_bounds[1]))

                    freq_new = best_array[i][1] + scale * best_array[i][1] * cauchy_noise[i][1]
                    freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                    if self.consider_aoi and len(individual[i]) > 2:
                        delta_new = best_array[i][2] + scale * best_array[i][2] * cauchy_noise[i][2]
                        delta_new = np.clip(delta_new,
                                           self.update_interval_bounds[0],
                                           self.update_interval_bounds[1])
                        new_solution.append([loc_new, freq_new, delta_new])
                    else:
                        new_solution.append([loc_new, freq_new])

                return self.handle_constraints(new_solution)

        except Exception:
            return individual

    def optimize(self) -> Tuple[List, float, List]:
        """
        执行RDHO优化（精英保留 + 贪婪选择）

        Returns:
            best_solution: 最优解
            best_fitness: 最优适应度
            history: 优化历史
        """
        # 双源混合初始化
        population = self.initialize_population()
        self.current_iteration = 0

        # 评估初始种群
        fitness_values = [self.evaluate_fitness(sol) for sol in population]

        # 找初始最优和最差
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            return None, float('inf'), []

        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        worst_idx = max(valid_indices, key=lambda i: fitness_values[i])

        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        worst_solution = population[worst_idx]

        self.history = [best_fitness]

        # 主迭代循环
        for iteration in range(self.max_iter):
            try:
                self.current_iteration = iteration

                # 按适应度排序确定角色
                sorted_indices = sorted(valid_indices, key=lambda i: fitness_values[i])

                n_producers = max(1, int(self.population_size * self.producer_ratio))
                n_followers = max(1, int(self.population_size * self.follower_ratio))
                n_scouts = max(1, int(self.population_size * self.scout_ratio))

                producer_set = set(sorted_indices[:n_producers])
                follower_set = set(sorted_indices[n_producers:n_producers + n_followers])
                scout_set = set(sorted_indices[-n_scouts:])

                # 精英解保留
                n_elites = max(1, int(self.population_size * self.elite_ratio))
                elite_set = set(sorted_indices[:n_elites])

                for idx in range(self.population_size):
                    if idx not in valid_indices:
                        continue

                    # 精英解跳过更新
                    if idx in elite_set:
                        continue

                    old_fitness = fitness_values[idx]
                    candidate = None

                    # 根据角色选择更新策略
                    if idx in producer_set:
                        # 生产者：RIME软凝华 + DBO滚球融合
                        candidate = self.producer_update_fusion(
                            population[idx], best_solution, worst_solution, iteration)

                    elif idx in follower_set:
                        # 跟随者：RIME硬凝华穿刺 or DBO觅食
                        candidate = self.follower_update_hybrid(
                            population[idx], best_solution, iteration)

                    elif idx in scout_set:
                        # 侦察者：DBO偷窃 or Cauchy变异
                        local_best_idx = sorted_indices[min(len(sorted_indices) - 1,
                                                           int(idx * 0.3))]
                        local_best = population[local_best_idx]
                        candidate = self.scout_update_cauchy(
                            population[idx], best_solution, local_best,
                            old_fitness, best_fitness)

                    # 贪婪选择
                    if candidate is not None:
                        new_fitness = self.evaluate_fitness(candidate)
                        if np.isfinite(new_fitness) and new_fitness < old_fitness:
                            population[idx] = candidate
                            fitness_values[idx] = new_fitness

                            # 更新全局最优
                            if new_fitness < best_fitness:
                                best_solution = candidate
                                best_fitness = new_fitness

                # 更新最差解
                valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
                if valid_indices:
                    worst_idx = max(valid_indices, key=lambda i: fitness_values[i])
                    worst_solution = population[worst_idx]

                self.history.append(best_fitness)

                if self.verbose and (iteration + 1) % 10 == 0:
                    print(f"Iteration {iteration + 1}/{self.max_iter}, "
                          f"Best fitness: {best_fitness:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"Error in iteration {iteration}: {e}")
                self.history.append(best_fitness)

        if self.verbose:
            print(f"Optimization completed. Best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness, self.history
