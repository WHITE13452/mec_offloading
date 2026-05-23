"""
MSSA: Modified Sparrow Search Algorithm
基于论文: MSSAMTO-IoV: modified sparrow search algorithm for
         multi-hop task offloading for IoV (Journal of Supercomputing, 2023)
用于五目标MEC任务卸载优化

Reference:
Alseid, M., El-Moursy, A.A., Alfawaz, O. et al.
MSSAMTO-IoV: modified sparrow search algorithm for multi-hop task offloading for IoV.
Journal of Supercomputing 79, 20769–20789 (2023).
https://doi.org/10.1007/s11227-023-05446-2
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


class MSSA(BaseAlgorithm):
    """Modified Sparrow Search Algorithm (MSSA) - 改进麻雀搜索算法

    四项改进策略:
    1. Logistic Map混沌初始化
    2. 自适应惯性权重
    3. 均值终止准则
    4. 变异策略
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
                 mutation_rate: float = 0.1,
                 mutation_scale: float = 0.5,
                 termination_window: int = 20,
                 termination_threshold: float = 1e-6,
                 verbose: bool = False):
        """
        初始化MSSA算法

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
            mutation_rate: 变异概率
            mutation_scale: 变异幅度
            termination_window: 终止检查窗口
            termination_threshold: 终止阈值
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

        # MSSA特有参数
        self.w_max = w_max
        self.w_min = w_min
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.termination_window = termination_window
        self.termination_threshold = termination_threshold

        # 动态归一化因子
        self.energy_max = 1.0
        self.delay_max = 1.0
        self.aoi_max = 1.0

        # 当前迭代
        self.current_iteration = 0

        # 历史记录
        self.history = []

    # ==================== 改进1: Logistic混沌初始化 ====================

    def logistic_map(self, x: float, mu: float = 4.0) -> float:
        """Logistic混沌映射"""
        return mu * x * (1 - x)

    def logistic_init(self, dim: int) -> np.ndarray:
        """使用Logistic混沌映射生成一个解向量"""
        x = np.random.rand(dim)
        x = np.clip(x, 0.001, 0.999)

        for _ in range(10):
            for d in range(dim):
                x[d] = self.logistic_map(x[d])

        return x

    def map_chaos_to_solution(self, chaos_vector: np.ndarray) -> List:
        """将混沌向量映射到决策变量空间"""
        solution = []
        offset = 0

        for task_idx in range(self.num_tasks):
            loc_chaos = chaos_vector[offset]
            loc_i = int(loc_chaos * (self.loc_bounds[1] - self.loc_bounds[0] + 1)) + self.loc_bounds[0]
            loc_i = np.clip(loc_i, self.loc_bounds[0], self.loc_bounds[1])

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
        """改进1: Logistic Map混沌初始化"""
        population = []

        if self.consider_aoi:
            dim = self.num_tasks * 3
        else:
            dim = self.num_tasks * 2

        for _ in range(self.population_size):
            chaos_vector = self.logistic_init(dim)
            solution = self.map_chaos_to_solution(chaos_vector)
            solution = self.handle_constraints(solution)
            population.append(solution)

        return population

    # ==================== 改进2: 自适应惯性权重 ====================

    def adaptive_inertia_weight(self) -> float:
        """改进2: 自适应惯性权重（线性衰减）"""
        progress = self.current_iteration / self.max_iter
        w = self.w_max - (self.w_max - self.w_min) * progress
        return w

    # ==================== 改进3: 均值终止准则 ====================

    def mean_termination_check(self, history: List[float]) -> bool:
        """改进3: 均值终止准则"""
        if len(history) < self.termination_window:
            return False

        recent = history[-self.termination_window:]
        half = self.termination_window // 2

        first_half_mean = np.mean(recent[:half])
        second_half_mean = np.mean(recent[half:])
        mean_change = abs(first_half_mean - second_half_mean)

        return mean_change < self.termination_threshold

    # ==================== 改进4: 变异策略 ====================

    def mutation_operator(self, x: List, x_best: List) -> List:
        """改进4: 变异操作"""
        if np.random.rand() < self.mutation_rate:
            x_array = np.array(x, dtype=float)
            best_array = np.array(x_best, dtype=float)

            mutation = np.random.randn(*x_array.shape) * self.mutation_scale
            x_new = x_array + mutation * (best_array - x_array)

            return self.handle_constraints(x_new.tolist())
        return x

    # ==================== SSA核心更新 ====================

    def update_producer(self, producer: List, best_solution: List) -> List:
        """发现者更新（带惯性权重）"""
        try:
            producer_array = np.array(producer, dtype=float)
            best_array = np.array(best_solution, dtype=float)

            w = self.adaptive_inertia_weight()
            R2 = np.random.random()

            if R2 < self.ST:
                alpha = np.random.random()
                new_position = (w * producer_array +
                               (1 - w) * (best_array + alpha * np.abs(producer_array - best_array)))
            else:
                Q = np.random.random()
                rand_factor = np.random.random() * self.max_iter + 1e-10
                new_position = w * producer_array + Q * np.exp(-self.current_iteration / rand_factor)

            return self.handle_constraints(new_position.tolist())

        except Exception:
            return producer

    def update_scrounger(self, scrounger: List, producer: List,
                         worst_solution: List, idx: int) -> List:
        """跟随者更新"""
        try:
            scrounger_array = np.array(scrounger, dtype=float)
            producer_array = np.array(producer, dtype=float)
            worst_array = np.array(worst_solution, dtype=float)

            half_pop = self.population_size // 2

            if idx > half_pop:
                Q = np.random.random()
                idx_sq = (idx - half_pop + 1) ** 2 + 1e-10
                new_position = Q * np.exp((worst_array - scrounger_array) / idx_sq)
            else:
                A = np.random.choice([-1, 1], size=scrounger_array.shape)
                A_plus = A / (np.linalg.norm(A) + 1e-10)
                L = np.random.random()
                new_position = producer_array + np.abs(scrounger_array - producer_array) * A_plus * L

            return self.handle_constraints(new_position.tolist())

        except Exception:
            return scrounger

    def update_scout(self, scout: List, best_solution: List,
                     worst_solution: List, scout_fitness: float,
                     best_fitness: float, worst_fitness: float) -> List:
        """侦察者更新"""
        try:
            scout_array = np.array(scout, dtype=float)
            best_array = np.array(best_solution, dtype=float)
            worst_array = np.array(worst_solution, dtype=float)

            if scout_fitness > best_fitness:
                beta = np.random.randn()
                new_position = best_array + beta * np.abs(scout_array - best_array)
            else:
                K = np.random.uniform(-1, 1)
                denom = abs(scout_fitness - worst_fitness) + 1e-10
                new_position = scout_array + K * np.abs(scout_array - worst_array) / denom

            return self.handle_constraints(new_position.tolist())

        except Exception:
            return scout

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
        """执行MSSA优化"""

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
                new_solution = self.update_producer(population[idx], best_solution)
                new_fitness = self.evaluate_fitness(new_solution)

                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness

            # 跟随者更新
            for i in range(n_producers, self.population_size - n_scouts):
                idx = sorted_indices[i]
                producer_idx = sorted_indices[np.random.randint(n_producers)]

                new_solution = self.update_scrounger(
                    population[idx], population[producer_idx],
                    worst_solution, i)
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

            # 变异操作
            for i in range(self.population_size):
                mutated = self.mutation_operator(population[i], best_solution)
                if mutated != population[i]:
                    mutated_fitness = self.evaluate_fitness(mutated)
                    if np.isfinite(mutated_fitness) and mutated_fitness < fitness_values[i]:
                        population[i] = mutated
                        fitness_values[i] = mutated_fitness

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

            # 均值终止检查
            if self.mean_termination_check(self.history):
                if self.verbose:
                    print(f"Early termination at iteration {iteration + 1}")
                break

            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {best_fitness:.6f}")

        if self.verbose:
            print(f"MSSA optimization completed. Best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness, self.history
