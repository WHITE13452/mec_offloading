"""
DBO算法 (Dung Beetle Optimizer)
基于蜣螂行为的优化算法，包含滚球、繁殖、觅食和偷窃四种行为
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


class DBO(BaseAlgorithm):
    """DBO算法 - 五目标优化"""

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
                 rolling_ratio: float = 0.2,
                 breeding_ratio: float = 0.2,
                 foraging_ratio: float = 0.4,
                 verbose: bool = False):
        """
        初始化DBO算法

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
            rolling_ratio: 滚球蜣螂比例
            breeding_ratio: 繁殖蜣螂比例
            foraging_ratio: 觅食蜣螂比例
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

        # 蜣螂角色比例
        self.rolling_ratio = rolling_ratio
        self.breeding_ratio = breeding_ratio
        self.foraging_ratio = foraging_ratio
        self.thief_ratio = 1.0 - rolling_ratio - breeding_ratio - foraging_ratio

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

        # 当前迭代
        self.current_iteration = 0

    def initialize_population(self) -> List:
        """
        初始化种群（均匀分布）

        Returns:
            初始种群
        """
        population = []
        for _ in range(self.population_size):
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

    def ball_rolling(self, beetle: List, worst_solution: List,
                     iteration: int) -> List:
        """
        滚球蜣螂行为（全局探索）

        公式: x_new = x + α·k·(x - lb) + b·|x - x_worst|
        其中: α = 1 - t/T, k ∈ [-1, 1], b ∈ [0, 1]

        Args:
            beetle: 当前蜣螂位置
            worst_solution: 最差解
            iteration: 当前迭代次数

        Returns:
            新位置
        """
        try:
            beetle_array = np.array(beetle, dtype=float)
            worst_array = np.array(worst_solution, dtype=float)

            # 自适应探索因子
            alpha = 1.0 - iteration / self.max_iter
            k = np.random.uniform(-1, 1, size=beetle_array.shape)
            b = np.random.rand()

            new_solution = []
            for i in range(self.num_tasks):
                # 位置滚动
                loc_lb = self.loc_bounds[0]
                loc_new = (beetle_array[i][0] +
                          alpha * k[i][0] * (beetle_array[i][0] - loc_lb) +
                          b * np.abs(beetle_array[i][0] - worst_array[i][0]))
                loc_new = int(np.clip(np.round(loc_new),
                                     self.loc_bounds[0], self.loc_bounds[1]))

                # 频率滚动
                freq_lb = self.freq_bounds[0]
                freq_new = (beetle_array[i][1] +
                           alpha * k[i][1] * (beetle_array[i][1] - freq_lb) +
                           b * np.abs(beetle_array[i][1] - worst_array[i][1]))
                freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi and len(beetle[i]) > 2:
                    delta_lb = self.update_interval_bounds[0]
                    delta_new = (beetle_array[i][2] +
                                alpha * k[i][2] * (beetle_array[i][2] - delta_lb) +
                                b * np.abs(beetle_array[i][2] - worst_array[i][2]))
                    delta_new = np.clip(delta_new,
                                       self.update_interval_bounds[0],
                                       self.update_interval_bounds[1])
                    new_solution.append([loc_new, freq_new, delta_new])
                else:
                    new_solution.append([loc_new, freq_new])

            return self.handle_constraints(new_solution)

        except Exception:
            return beetle

    def breeding(self, beetle: List, best_solution: List,
                 local_best: List) -> List:
        """
        繁殖蜣螂行为（局部开发）

        公式: x_new = x_best + β1·|x - x_local_best|
        其中: β1 ∈ [-1, 1]

        Args:
            beetle: 当前蜣螂位置
            best_solution: 全局最优解
            local_best: 局部最优解

        Returns:
            新位置
        """
        try:
            beetle_array = np.array(beetle, dtype=float)
            best_array = np.array(best_solution, dtype=float)
            local_best_array = np.array(local_best, dtype=float)

            beta1 = np.random.uniform(-1, 1, size=beetle_array.shape)

            new_solution = []
            for i in range(self.num_tasks):
                # 位置繁殖
                loc_new = (best_array[i][0] +
                          beta1[i][0] * np.abs(beetle_array[i][0] - local_best_array[i][0]))
                loc_new = int(np.clip(np.round(loc_new),
                                     self.loc_bounds[0], self.loc_bounds[1]))

                # 频率繁殖
                freq_new = (best_array[i][1] +
                           beta1[i][1] * np.abs(beetle_array[i][1] - local_best_array[i][1]))
                freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi and len(beetle[i]) > 2:
                    delta_new = (best_array[i][2] +
                                beta1[i][2] * np.abs(beetle_array[i][2] - local_best_array[i][2]))
                    delta_new = np.clip(delta_new,
                                       self.update_interval_bounds[0],
                                       self.update_interval_bounds[1])
                    new_solution.append([loc_new, freq_new, delta_new])
                else:
                    new_solution.append([loc_new, freq_new])

            return self.handle_constraints(new_solution)

        except Exception:
            return beetle

    def foraging(self, beetle: List, best_solution: List) -> List:
        """
        小蜣螂觅食行为（均衡搜索）

        公式: x_new = x_best + C1·(x - lb) + C2·(x - ub)
        其中: C1, C2 ∈ [0, 1]

        Args:
            beetle: 当前蜣螂位置
            best_solution: 全局最优解

        Returns:
            新位置
        """
        try:
            beetle_array = np.array(beetle, dtype=float)
            best_array = np.array(best_solution, dtype=float)

            C1 = np.random.rand()
            C2 = np.random.rand()

            new_solution = []
            for i in range(self.num_tasks):
                # 位置觅食
                loc_lb = self.loc_bounds[0]
                loc_ub = self.loc_bounds[1]
                loc_new = (best_array[i][0] +
                          C1 * (beetle_array[i][0] - loc_lb) +
                          C2 * (beetle_array[i][0] - loc_ub))
                loc_new = int(np.clip(np.round(loc_new),
                                     self.loc_bounds[0], self.loc_bounds[1]))

                # 频率觅食
                freq_lb = self.freq_bounds[0]
                freq_ub = self.freq_bounds[1]
                freq_new = (best_array[i][1] +
                           C1 * (beetle_array[i][1] - freq_lb) +
                           C2 * (beetle_array[i][1] - freq_ub))
                freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi and len(beetle[i]) > 2:
                    delta_lb = self.update_interval_bounds[0]
                    delta_ub = self.update_interval_bounds[1]
                    delta_new = (best_array[i][2] +
                                C1 * (beetle_array[i][2] - delta_lb) +
                                C2 * (beetle_array[i][2] - delta_ub))
                    delta_new = np.clip(delta_new,
                                       self.update_interval_bounds[0],
                                       self.update_interval_bounds[1])
                    new_solution.append([loc_new, freq_new, delta_new])
                else:
                    new_solution.append([loc_new, freq_new])

            return self.handle_constraints(new_solution)

        except Exception:
            return beetle

    def thief(self, beetle: List, local_best: List) -> List:
        """
        偷窃蜣螂行为（扰动机制）

        公式: x_new = x_local_best + tan(θ)·|x - x_local_best|
        其中: θ ∈ [-π/4, π/4]

        Args:
            beetle: 当前蜣螂位置
            local_best: 局部最优解

        Returns:
            新位置
        """
        try:
            beetle_array = np.array(beetle, dtype=float)
            local_best_array = np.array(local_best, dtype=float)

            # 随机角度
            theta = np.random.uniform(-np.pi / 4, np.pi / 4,
                                     size=beetle_array.shape)

            new_solution = []
            for i in range(self.num_tasks):
                # 位置偷窃
                loc_new = (local_best_array[i][0] +
                          np.tan(theta[i][0]) * np.abs(beetle_array[i][0] - local_best_array[i][0]))
                loc_new = int(np.clip(np.round(loc_new),
                                     self.loc_bounds[0], self.loc_bounds[1]))

                # 频率偷窃
                freq_new = (local_best_array[i][1] +
                           np.tan(theta[i][1]) * np.abs(beetle_array[i][1] - local_best_array[i][1]))
                freq_new = np.clip(freq_new, self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi and len(beetle[i]) > 2:
                    delta_new = (local_best_array[i][2] +
                                np.tan(theta[i][2]) * np.abs(beetle_array[i][2] - local_best_array[i][2]))
                    delta_new = np.clip(delta_new,
                                       self.update_interval_bounds[0],
                                       self.update_interval_bounds[1])
                    new_solution.append([loc_new, freq_new, delta_new])
                else:
                    new_solution.append([loc_new, freq_new])

            return self.handle_constraints(new_solution)

        except Exception:
            return beetle

    def optimize(self) -> Tuple[List, float, List]:
        """
        执行DBO优化

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

                n_rolling = max(1, int(self.population_size * self.rolling_ratio))
                n_breeding = max(1, int(self.population_size * self.breeding_ratio))
                n_foraging = max(1, int(self.population_size * self.foraging_ratio))

                rolling_set = set(sorted_indices[:n_rolling])
                breeding_set = set(sorted_indices[n_rolling:n_rolling + n_breeding])
                foraging_set = set(sorted_indices[n_rolling + n_breeding:
                                                 n_rolling + n_breeding + n_foraging])
                thief_set = set(sorted_indices) - rolling_set - breeding_set - foraging_set

                for idx in range(self.population_size):
                    if idx not in valid_indices:
                        continue

                    old_fitness = fitness_values[idx]
                    candidate = None

                    # 根据角色选择行为
                    if idx in rolling_set:
                        # 滚球行为
                        candidate = self.ball_rolling(population[idx], worst_solution, iteration)

                    elif idx in breeding_set:
                        # 繁殖行为（局部最优为邻域最优）
                        local_best_idx = sorted_indices[min(len(sorted_indices) - 1,
                                                           int(idx * 0.3))]
                        local_best = population[local_best_idx]
                        candidate = self.breeding(population[idx], best_solution, local_best)

                    elif idx in foraging_set:
                        # 觅食行为
                        candidate = self.foraging(population[idx], best_solution)

                    else:
                        # 偷窃行为
                        local_best_idx = sorted_indices[min(len(sorted_indices) - 1,
                                                           int(idx * 0.5))]
                        local_best = population[local_best_idx]
                        candidate = self.thief(population[idx], local_best)

                    # 贪婪选择
                    if candidate is not None:
                        new_fitness = self.evaluate_fitness(candidate)
                        if np.isfinite(new_fitness) and new_fitness < old_fitness:
                            population[idx] = candidate
                            fitness_values[idx] = new_fitness

                            # 更新全局最优和最差
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
