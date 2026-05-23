"""
RIME算法 (Rime-ice Optimization Algorithm)
基于冰凌形成过程的优化算法，包含软凝华搜索和硬凝华穿刺两个阶段
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


class RIME(BaseAlgorithm):
    """RIME算法 - 五目标优化"""

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
                 verbose: bool = False):
        """
        初始化RIME算法

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

        # 决策变量边界已在父类中定义，无需重复设置
        # self.num_tasks, self.loc_bounds, self.freq_bounds 已由 BaseAlgorithm 设置

        # AoI相关（已在父类中设置）
        # self.consider_aoi 已由 BaseAlgorithm 设置
        # self.update_interval_bounds 已由 BaseAlgorithm 设置

        # 动态归一化因子
        self.energy_max = 1.0
        self.delay_max = 1.0
        self.aoi_max = 1.0

        # 优化历史
        self.history = []

        # 当前迭代（用于自适应参数）
        self.current_iteration = 0

    def initialize_population(self) -> List:
        """
        初始化种群（高斯分布）

        Returns:
            初始种群
        """
        population = []
        for _ in range(self.population_size):
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

        return population

    def evaluate_fitness(self, solution) -> float:
        """
        评估解的适应度（五目标优化）

        Args:
            solution: 决策解

        Returns:
            适应度值（越小越好）
        """
        try:
            if not solution or len(solution) != self.num_tasks:
                return float('inf')

            # 应用解到系统模型
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

            # 约束惩罚
            penalty = constraint_violations * 2.0

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

    def soft_rime_search(self, solution: List, best_solution: List,
                         iteration: int) -> List:
        """
        软凝华搜索（探索阶段）

        公式: x_new = x_best + β·cos(θ)·h·(ub - lb)
        其中: h = 2·(1 - t/T)

        Args:
            solution: 当前解
            best_solution: 全局最优解
            iteration: 当前迭代次数

        Returns:
            新解
        """
        try:
            solution_array = np.array(solution, dtype=float)
            best_array = np.array(best_solution, dtype=float)

            # 自适应搜索半径
            h = 2.0 * (1.0 - iteration / self.max_iter)

            new_solution = []
            for i in range(self.num_tasks):
                # 随机参数
                beta = np.random.randn()
                theta = 2 * np.pi * np.random.rand()

                # 位置搜索
                loc_range = self.loc_bounds[1] - self.loc_bounds[0]
                loc_new = (best_array[i][0] +
                          beta * np.cos(theta) * h * loc_range)
                loc_new = np.clip(loc_new, self.loc_bounds[0], self.loc_bounds[1])
                loc_new = int(np.round(loc_new))

                # 频率搜索
                freq_range = self.freq_bounds[1] - self.freq_bounds[0]
                freq_new = (best_array[i][1] +
                           beta * np.cos(theta + np.pi/4) * h * freq_range)
                freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi and len(solution[i]) > 2:
                    # 更新间隔搜索
                    delta_range = (self.update_interval_bounds[1] -
                                  self.update_interval_bounds[0])
                    delta_new = (best_array[i][2] +
                                beta * np.cos(theta + np.pi/2) * h * delta_range)
                    delta_new = np.clip(delta_new,
                                       self.update_interval_bounds[0],
                                       self.update_interval_bounds[1])
                    new_solution.append([loc_new, freq_new, delta_new])
                else:
                    new_solution.append([loc_new, freq_new])

            return self.handle_constraints(new_solution)

        except Exception:
            return solution

    def hard_rime_puncture(self, solution: List, best_solution: List,
                           iteration: int) -> List:
        """
        硬凝华穿刺（开发阶段）

        公式: E = 2·exp(-(4·t/T)²)
        如果 rand < E，则 x_new[j_rand] = x_best[j_rand]

        Args:
            solution: 当前解
            best_solution: 全局最优解
            iteration: 当前迭代次数

        Returns:
            新解
        """
        try:
            solution_array = np.array(solution, dtype=float)
            best_array = np.array(best_solution, dtype=float)

            # 穿刺概率（随迭代递减）
            E = 2.0 * np.exp(-((4.0 * iteration / self.max_iter) ** 2))

            new_solution = solution_array.copy()

            # 以概率E进行穿刺操作
            if np.random.rand() < E:
                # 随机选择维度进行穿刺
                j_rand = np.random.randint(self.num_tasks)
                new_solution[j_rand] = best_array[j_rand].copy()
            else:
                # 局部微调
                for i in range(self.num_tasks):
                    if np.random.rand() < 0.3:
                        # 位置微调
                        if np.random.rand() < 0.5:
                            perturbation = np.random.randint(-1, 2)
                            new_solution[i][0] = np.clip(
                                new_solution[i][0] + perturbation,
                                self.loc_bounds[0], self.loc_bounds[1])

                        # 频率微调
                        else:
                            scale = 0.1 * (1.0 - iteration / self.max_iter)
                            perturbation = np.random.randn() * scale * self.freq_bounds[1]
                            new_solution[i][1] = np.clip(
                                new_solution[i][1] + perturbation,
                                self.freq_bounds[0], self.freq_bounds[1])

            return self.handle_constraints(new_solution.tolist())

        except Exception:
            return solution

    def positive_greedy_selection(self, old_solution: List, old_fitness: float,
                                  new_solution: List) -> Tuple[List, float]:
        """
        正向贪婪选择机制

        Args:
            old_solution: 旧解
            old_fitness: 旧解适应度
            new_solution: 新解

        Returns:
            选择的解和适应度
        """
        new_fitness = self.evaluate_fitness(new_solution)

        if np.isfinite(new_fitness) and new_fitness < old_fitness:
            return new_solution, new_fitness
        else:
            return old_solution, old_fitness

    def optimize(self) -> Tuple[List, float, List]:
        """
        执行RIME优化

        Returns:
            best_solution: 最优解
            best_fitness: 最优适应度
            history: 优化历史
        """
        # 初始化种群
        population = self.initialize_population()
        self.current_iteration = 0

        # 评估初始种群
        fitness_values = [self.evaluate_fitness(sol) for sol in population]

        # 找初始最优
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            return None, float('inf'), []

        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]

        self.history = [best_fitness]

        # 主迭代循环
        for iteration in range(self.max_iter):
            try:
                self.current_iteration = iteration

                # 计算探索/开发阈值
                exploration_rate = 1.0 - iteration / self.max_iter

                for idx in range(self.population_size):
                    old_fitness = fitness_values[idx]

                    # 根据探索率选择策略
                    if np.random.rand() < exploration_rate:
                        # 软凝华搜索（探索）
                        candidate = self.soft_rime_search(
                            population[idx], best_solution, iteration)
                    else:
                        # 硬凝华穿刺（开发）
                        candidate = self.hard_rime_puncture(
                            population[idx], best_solution, iteration)

                    # 正向贪婪选择
                    population[idx], fitness_values[idx] = \
                        self.positive_greedy_selection(
                            population[idx], old_fitness, candidate)

                    # 更新全局最优
                    if fitness_values[idx] < best_fitness:
                        best_solution = population[idx]
                        best_fitness = fitness_values[idx]

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
