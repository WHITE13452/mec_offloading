"""
CWTSSA: Chaotic Mapping, Adaptive Weighting, T-distribution mutation SSA
基于论文: A Novel Adaptive Sparrow Search Algorithm Based on Chaotic Mapping
         and T-Distribution Mutation (Applied Sciences 2021)
用于五目标MEC任务卸载优化

Reference:
Yang, W.; Xia, K.; Li, T.; Xie, M.; Gao, F.
A Novel Adaptive Sparrow Search Algorithm Based on Chaotic Mapping and T-Distribution Mutation.
Applied Sciences 2021, 11, 11192.
https://doi.org/10.3390/app112311192
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


class CWTSSA(BaseAlgorithm):
    """CWTSSA - 混沌映射+自适应权重+t分布变异的改进麻雀搜索算法

    三大改进策略:
    1. 混沌映射初始化 - 增强种群多样性
    2. 自适应权重策略 - 平衡探索与开发
    3. 自适应t分布变异 - 避免局部最优
    """

    def __init__(self,
                 system_model: SystemModel,
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
                 w_max: float = 0.9,
                 w_min: float = 0.4,
                 mutation_prob: float = 0.2,
                 mutation_scale: float = 0.5,
                 verbose: bool = False):
        """
        初始化CWTSSA算法

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
            PD: 发现者比例
            SD: 侦察者比例
            ST: 安全阈值
            w_max: 最大惯性权重
            w_min: 最小惯性权重
            mutation_prob: t分布变异概率
            mutation_scale: 变异幅度
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

        # SSA参数
        self.PD = PD
        self.SD = SD
        self.ST = ST

        # CWTSSA参数
        self.w_max = w_max
        self.w_min = w_min
        self.mutation_prob = mutation_prob
        self.mutation_scale = mutation_scale

        # 动态归一化因子
        self.energy_max = 1.0
        self.delay_max = 1.0
        self.aoi_max = 1.0

        # 当前迭代
        self.current_iteration = 0

        # 历史记录
        self.history = []

    # ==================== 改进1: 混沌映射初始化 ====================

    def logistic_chaotic_map(self, x: np.ndarray, mu: float = 4.0) -> np.ndarray:
        """Logistic混沌映射"""
        return mu * x * (1 - x)

    def chaotic_init_vector(self, dim: int) -> np.ndarray:
        """使用混沌映射生成一个初始化向量"""
        x = np.random.uniform(0.001, 0.999, dim)

        for _ in range(10):
            x = self.logistic_chaotic_map(x)

        return x

    def map_chaos_to_solution(self, chaos_vector: np.ndarray) -> List:
        """将混沌向量映射到决策变量空间"""
        solution = []
        offset = 0

        for task_idx in range(self.num_tasks):
            loc_chaos = chaos_vector[offset]
            loc_i = int(loc_chaos * (self.loc_bounds[1] - self.loc_bounds[0] + 1))
            loc_i = np.clip(loc_i + self.loc_bounds[0], self.loc_bounds[0], self.loc_bounds[1])

            freq_chaos = chaos_vector[offset + 1]
            f_i = self.freq_bounds[0] + freq_chaos * (self.freq_bounds[1] - self.freq_bounds[0])

            offset += 2

            if self.consider_aoi:
                delta_chaos = chaos_vector[offset]
                delta_i = self.update_interval_bounds[0] + delta_chaos * (
                    self.update_interval_bounds[1] - self.update_interval_bounds[0])
                solution.append([loc_i, f_i, delta_i])
                offset += 1
            else:
                solution.append([loc_i, f_i])

        return solution

    def initialize_population(self) -> List:
        """改进1: 混沌映射初始化种群"""
        population = []

        if self.consider_aoi:
            dim = self.num_tasks * 3
        else:
            dim = self.num_tasks * 2

        for _ in range(self.population_size):
            chaos_vector = self.chaotic_init_vector(dim)
            solution = self.map_chaos_to_solution(chaos_vector)
            solution = self.handle_constraints(solution)
            population.append(solution)

        return population

    # ==================== 改进2: 自适应权重 ====================

    def adaptive_weight(self) -> float:
        """改进2: 自适应权重（二次衰减）"""
        progress = (self.current_iteration / self.max_iter) ** 2
        w = self.w_max - (self.w_max - self.w_min) * progress
        return w

    # ==================== 改进3: t分布变异 ====================

    def t_distribution_mutation(self, x_best: List) -> List:
        """改进3: 自适应t分布变异"""
        x_best_array = np.array(x_best, dtype=float)

        df = self.current_iteration + 1
        t_random = np.random.standard_t(df, size=x_best_array.shape)
        adaptive_scale = self.mutation_scale * (1 - self.current_iteration / self.max_iter)

        mutation_range = np.zeros_like(x_best_array)
        for i in range(len(x_best)):
            if self.consider_aoi and len(x_best[i]) == 3:
                mutation_range[i] = [
                    self.loc_bounds[1] - self.loc_bounds[0],
                    self.freq_bounds[1] - self.freq_bounds[0],
                    self.update_interval_bounds[1] - self.update_interval_bounds[0]
                ]
            else:
                mutation_range[i] = [
                    self.loc_bounds[1] - self.loc_bounds[0],
                    self.freq_bounds[1] - self.freq_bounds[0]
                ]

        x_new_array = x_best_array + adaptive_scale * t_random * mutation_range

        return self.handle_constraints(x_new_array.tolist())

    # ==================== SSA核心更新 ====================

    def update_producer(self, x: List, x_best: List, x_worst: List, idx: int) -> List:
        """发现者更新（带自适应权重）"""
        try:
            x_array = np.array(x, dtype=float)
            best_array = np.array(x_best, dtype=float)
            worst_array = np.array(x_worst, dtype=float)

            w = self.adaptive_weight()
            R2 = np.random.random()

            if R2 < self.ST:
                alpha = np.random.random()
                exp_factor = np.exp(-self.current_iteration / (alpha * self.max_iter + 1e-10))
                new_position = w * x_array + (1 - w) * best_array * exp_factor
            else:
                Q = np.random.random()
                t_sq = (self.current_iteration + 1) ** 2 + 1e-10
                new_position = w * x_array + Q * np.exp((worst_array - x_array) / t_sq)

            return self.handle_constraints(new_position.tolist())

        except Exception:
            return x

    def update_scrounger(self, x: List, x_producer: List, x_worst: List, idx: int) -> List:
        """跟随者更新"""
        try:
            x_array = np.array(x, dtype=float)
            producer_array = np.array(x_producer, dtype=float)
            worst_array = np.array(x_worst, dtype=float)

            half_pop = self.population_size // 2

            if idx > half_pop:
                Q = np.random.random()
                idx_sq = (idx - half_pop + 1) ** 2 + 1e-10
                new_position = Q * np.exp((worst_array - x_array) / idx_sq)
            else:
                A = np.random.choice([-1, 1], size=x_array.shape)
                A_plus = A / (np.linalg.norm(A) + 1e-10)
                L = np.random.random()
                new_position = producer_array + np.abs(x_array - producer_array) * A_plus * L

            return self.handle_constraints(new_position.tolist())

        except Exception:
            return x

    def update_scout(self, x: List, x_best: List, x_worst: List,
                     f_x: float, f_best: float, f_worst: float) -> List:
        """侦察者更新"""
        try:
            x_array = np.array(x, dtype=float)
            best_array = np.array(x_best, dtype=float)
            worst_array = np.array(x_worst, dtype=float)

            if f_x > f_best:
                beta = np.random.randn()
                new_position = best_array + beta * np.abs(x_array - best_array)
            else:
                K = np.random.uniform(-1, 1)
                denom = abs(f_x - f_worst) + 1e-10
                new_position = x_array + K * np.abs(x_array - worst_array) / denom

            return self.handle_constraints(new_position.tolist())

        except Exception:
            return x

    # ==================== 适应度评估 ====================

    def evaluate_fitness(self, solution: List) -> float:
        """五目标适应度评估"""
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
                    delay = self.delay_model.calculate_total_delay(task)
                    if not np.isfinite(delay) or delay < 0:
                        return float('inf')
                    total_delay += delay
                    if delay > task.max_delay:
                        constraint_violations += 1

                    energy = self.energy_model.calculate_total_energy(task)
                    if not np.isfinite(energy) or energy < 0:
                        return float('inf')
                    total_energy += energy

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

            self.energy_max = max(self.energy_max, total_energy, 1.0)
            self.delay_max = max(self.delay_max, total_delay, 1.0)
            if valid_aoi_tasks > 0:
                self.aoi_max = max(self.aoi_max, total_aoi / valid_aoi_tasks, 1.0)

            normalized_energy = total_energy / self.energy_max
            normalized_delay = total_delay / self.delay_max
            normalized_aoi = (total_aoi / valid_aoi_tasks / self.aoi_max
                            if valid_aoi_tasks > 0 else 0.0)

            qoe = self.qoe_model.calculate_system_qoe(solution) if self.qoe_model else 0.0
            fairness = self.fairness_model.calculate_fairness(solution) if self.fairness_model else 0.0

            penalty = constraint_violations * 2.0

            fitness = (self.w_energy * normalized_energy +
                      self.w_delay * normalized_delay +
                      self.w_aoi * normalized_aoi +
                      self.w_qoe * (1.0 - qoe) +
                      self.w_fairness * (1.0 - fairness) +
                      penalty)

            return fitness if np.isfinite(fitness) else float('inf')

        except Exception:
            return float('inf')

    # ==================== 主优化循环 ====================

    def optimize(self) -> Tuple[List, float, List]:
        """执行CWTSSA优化"""

        population = self.initialize_population()
        self.current_iteration = 0

        fitness_values = [self.evaluate_fitness(sol) for sol in population]

        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            return None, float('inf'), []

        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        worst_idx = max(valid_indices, key=lambda i: fitness_values[i])

        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        worst_solution = population[worst_idx]
        worst_fitness = fitness_values[worst_idx]

        self.history = [best_fitness]

        n_producers = max(1, int(self.population_size * self.PD))
        n_scouts = max(1, int(self.population_size * self.SD))

        for iteration in range(self.max_iter):
            self.current_iteration = iteration

            sorted_indices = sorted(range(len(fitness_values)),
                                   key=lambda i: fitness_values[i])

            # 发现者更新
            for i in range(n_producers):
                idx = sorted_indices[i]
                new_solution = self.update_producer(
                    population[idx], best_solution, worst_solution, idx)
                new_fitness = self.evaluate_fitness(new_solution)

                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness

            # 跟随者更新
            for i in range(n_producers, self.population_size - n_scouts):
                idx = sorted_indices[i]
                producer_idx = sorted_indices[np.random.randint(n_producers)]

                new_solution = self.update_scrounger(
                    population[idx], population[producer_idx], worst_solution, i)
                new_fitness = self.evaluate_fitness(new_solution)

                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness

            # 侦察者更新
            for i in range(self.population_size - n_scouts, self.population_size):
                idx = sorted_indices[i]
                new_solution = self.update_scout(
                    population[idx], best_solution, worst_solution,
                    fitness_values[idx], best_fitness, worst_fitness)
                new_fitness = self.evaluate_fitness(new_solution)

                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness

            # t分布变异
            if np.random.random() < self.mutation_prob:
                mutated = self.t_distribution_mutation(best_solution)
                mutated_fitness = self.evaluate_fitness(mutated)

                if np.isfinite(mutated_fitness) and mutated_fitness < best_fitness:
                    worst_idx_current = max(range(len(fitness_values)),
                                           key=lambda i: fitness_values[i])
                    population[worst_idx_current] = mutated
                    fitness_values[worst_idx_current] = mutated_fitness
                    best_solution = mutated
                    best_fitness = mutated_fitness

            # 更新全局最优和最差
            valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
            if valid_indices:
                current_best_idx = min(valid_indices, key=lambda i: fitness_values[i])
                if fitness_values[current_best_idx] < best_fitness:
                    best_solution = population[current_best_idx]
                    best_fitness = fitness_values[current_best_idx]

                current_worst_idx = max(valid_indices, key=lambda i: fitness_values[i])
                worst_solution = population[current_worst_idx]
                worst_fitness = fitness_values[current_worst_idx]

            self.history.append(best_fitness)

            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {best_fitness:.6f}")

        if self.verbose:
            print(f"CWTSSA optimization completed. Best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness, self.history
