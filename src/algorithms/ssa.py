"""
麻雀搜索算法 (Sparrow Search Algorithm, SSA)
基于麻雀觅食和反捕食行为的元启发式优化算法
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


class SSA(BaseAlgorithm):
    """标准麻雀搜索算法 - 支持五目标优化"""

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
                 verbose: bool = False):
        """
        初始化SSA算法

        Parameters:
        -----------
        system_model : SystemModel
            系统模型
        delay_model : DelayModel
            时延模型
        energy_model : EnergyModel
            能耗模型
        aoi_model : AoIModel, optional
            AoI模型
        qoe_model : QoEModel, optional
            QoE模型
        fairness_model : FairnessModel, optional
            公平性模型
        max_iter : int
            最大迭代次数 (default=150)
        population_size : int
            种群大小 (default=50)
        w_energy : float
            能耗权重 (default=0.15)
        w_delay : float
            时延权重 (default=0.15)
        w_aoi : float
            AoI权重 (default=0.20)
        w_qoe : float
            QoE权重 (default=0.25)
        w_fairness : float
            公平性权重 (default=0.25)
        PD : float
            发现者（生产者）比例 (default=0.2)
        SD : float
            侦察者比例 (default=0.1)
        ST : float
            安全阈值 (default=0.8)
        verbose : bool
            是否输出详细信息
        """
        super().__init__(system_model, max_iter, population_size, verbose)

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
        self.PD = PD  # 发现者比例
        self.SD = SD  # 侦察者比例
        self.ST = ST  # 安全阈值

        # 归一化因子
        self.energy_max = 1e5  # 100,000 J
        self.delay_max = 1e3   # 1,000 s
        self.aoi_max = 1e2     # 100 s

        # 迭代历史
        self.history = []

    def initialize_population(self):
        """初始化种群（麻雀）"""
        population = []

        for _ in range(self.population_size):
            solution = []
            for _ in range(self.num_tasks):
                loc_i = np.random.randint(self.loc_bounds[0], self.loc_bounds[1] + 1)
                f_i = np.random.uniform(self.freq_bounds[0], self.freq_bounds[1])

                if self.consider_aoi:
                    delta_i = np.random.uniform(
                        self.update_interval_bounds[0],
                        self.update_interval_bounds[1]
                    )
                    solution.append([loc_i, f_i, delta_i])
                else:
                    solution.append([loc_i, f_i])

            solution = self.handle_constraints(solution)
            population.append(solution)

        return population

    def evaluate_fitness(self, solution):
        """
        五目标适应度函数（与FPA保持一致）

        F(X) = w_E*E/E_norm + w_T*T/T_norm + w_AoI*AoI/AoI_norm
             + w_QoE*(1-QoE) + w_F*(1-Fairness) + Penalty

        Returns:
        --------
        float : 适应度值（越小越好）
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

            # 动态更新归一化因子
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

            # 约束惩罚
            penalty = constraint_violations * 2.0

            # 五目标适应度函数
            fitness = (self.w_energy * normalized_energy +
                      self.w_delay * normalized_delay +
                      self.w_aoi * normalized_aoi +
                      self.w_qoe * (1.0 - qoe) +
                      self.w_fairness * (1.0 - fairness) +
                      penalty)

            return fitness if np.isfinite(fitness) else float('inf')

        except Exception:
            return float('inf')

    def update_producer(self, sparrow: List, iteration: int, best_sparrow: List) -> List:
        """
        发现者（生产者）更新策略

        当R2 < ST时（无捕食者）:
            X[i] = X[i] * exp(-i / (alpha * max_iter))

        当R2 >= ST时（有捕食者威胁）:
            X[i] = X[i] + Q * L

        Parameters:
        -----------
        sparrow : List
            当前麻雀
        iteration : int
            当前迭代次数
        best_sparrow : List
            全局最优麻雀

        Returns:
        --------
        List : 更新后的麻雀
        """
        try:
            R2 = np.random.random()  # 预警值
            alpha = 0.5  # 缩放因子

            sparrow_array = np.array(sparrow, dtype=float)

            if R2 < self.ST:
                # 无捕食者威胁，正常搜索
                decay = np.exp(-iteration / (alpha * self.max_iter))
                new_sparrow = sparrow_array * decay
            else:
                # 有捕食者威胁，随机位置
                Q = np.random.random()
                L = np.ones_like(sparrow_array)  # 1×d的全1矩阵
                new_sparrow = sparrow_array + Q * L

            return self.handle_constraints(new_sparrow.tolist())

        except Exception:
            return sparrow

    def update_scrounger(self, scrounger_idx: int, population: List,
                        best_sparrow: List, worst_sparrow: List) -> List:
        """
        加入者（觅食者）更新策略

        当i > n/2时（饥饿的加入者）:
            X[i] = Q * exp((X_worst - X[i]) / i^2)

        当i <= n/2时:
            X[i] = X_best + |X[i] - X_best| * A * L

        Parameters:
        -----------
        scrounger_idx : int
            加入者索引
        population : List[List]
            当前种群
        best_sparrow : List
            全局最优麻雀
        worst_sparrow : List
            全局最差麻雀

        Returns:
        --------
        List : 更新后的麻雀
        """
        try:
            scrounger = population[scrounger_idx]
            n = len(population)

            scrounger_array = np.array(scrounger, dtype=float)
            best_array = np.array(best_sparrow, dtype=float)
            worst_array = np.array(worst_sparrow, dtype=float)

            if scrounger_idx > n / 2:
                # 饥饿的加入者，向最差个体反方向移动
                Q = np.random.random()
                exponent = (worst_array - scrounger_array) / ((scrounger_idx + 1) ** 2)
                new_scrounger = Q * np.exp(exponent)
            else:
                # 普通加入者，跟随最优个体
                A = np.random.choice([-1, 1], size=scrounger_array.shape)  # +1或-1
                L = np.ones_like(scrounger_array)
                new_scrounger = best_array + np.abs(scrounger_array - best_array) * A * L

            return self.handle_constraints(new_scrounger.tolist())

        except Exception:
            return scrounger

    def update_scout(self, scout: List, best_sparrow: List, worst_sparrow: List,
                    scout_fitness: float, best_fitness: float, worst_fitness: float) -> List:
        """
        侦察者更新策略

        当f[i] > f_g时（适应度差于全局最优）:
            X[i] = X_best + beta * |X[i] - X_best|

        当f[i] = f_g时（处于危险边缘）:
            X[i] = X[i] + K * ((X[i] - X_worst) / (f[i] - f_worst + eps))

        Parameters:
        -----------
        scout : List
            侦察者
        best_sparrow : List
            全局最优麻雀
        worst_sparrow : List
            全局最差麻雀
        scout_fitness : float
            侦察者适应度
        best_fitness : float
            全局最优适应度
        worst_fitness : float
            全局最差适应度

        Returns:
        --------
        List : 更新后的麻雀
        """
        try:
            scout_array = np.array(scout, dtype=float)
            best_array = np.array(best_sparrow, dtype=float)
            worst_array = np.array(worst_sparrow, dtype=float)

            if scout_fitness > best_fitness:
                # 适应度较差，向最优个体移动
                beta = np.random.random()
                new_scout = best_array + beta * np.abs(scout_array - best_array)
            else:
                # 处于危险边缘，利用位置和适应度信息逃逸
                K = np.random.uniform(-1, 1)
                eps = 1e-10
                denominator = scout_fitness - worst_fitness + eps

                if abs(denominator) > eps:
                    step = K * ((scout_array - worst_array) / denominator)
                    new_scout = scout_array + step
                else:
                    new_scout = scout_array

            return self.handle_constraints(new_scout.tolist())

        except Exception:
            return scout

    def optimize(self):
        """
        执行SSA优化

        主循环流程:
        1. 初始化种群
        2. 评估适应度
        3. 划分角色（发现者、加入者、侦察者）
        4. 分别更新不同角色
        5. 更新全局最优
        6. 重复2-5直到达到最大迭代次数

        Returns:
        --------
        Tuple[List, float, List] : (最优解, 最优适应度, 迭代历史)
        """
        # 初始化种群
        population = self.initialize_population()

        # 评估初始种群
        fitness_values = [self.evaluate_fitness(solution) for solution in population]

        # 找初始最优和最差
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            if self.verbose:
                print("Warning: No valid solutions found in initial population!")
            return None, float('inf'), []

        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        worst_idx = max(valid_indices, key=lambda i: fitness_values[i])

        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        worst_solution = population[worst_idx]
        worst_fitness = fitness_values[worst_idx]

        self.history = [best_fitness]

        # 主迭代循环
        for iteration in range(self.max_iter):
            try:
                # 对种群按适应度排序
                sorted_indices = sorted(valid_indices, key=lambda i: fitness_values[i])

                # 划分角色
                n_producers = max(1, int(self.population_size * self.PD))  # 至少1个
                n_scouts = max(1, int(self.population_size * self.SD))     # 至少1个

                producer_indices = sorted_indices[:n_producers]
                scout_indices = sorted_indices[-n_scouts:]
                # 剩余的是加入者
                scrounger_indices = sorted_indices[n_producers:len(sorted_indices)-n_scouts]

                new_population = population.copy()

                # 更新发现者
                for idx in producer_indices:
                    new_population[idx] = self.update_producer(
                        population[idx], iteration, best_solution)

                # 更新加入者
                for idx in scrounger_indices:
                    new_population[idx] = self.update_scrounger(
                        idx, population, best_solution, worst_solution)

                # 更新侦察者
                for idx in scout_indices:
                    new_population[idx] = self.update_scout(
                        population[idx], best_solution, worst_solution,
                        fitness_values[idx], best_fitness, worst_fitness)

                # 更新种群
                population = new_population

                # 重新评估适应度
                fitness_values = [self.evaluate_fitness(solution) for solution in population]

                # 更新有效索引
                valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
                if not valid_indices:
                    if self.verbose:
                        print(f"Warning: No valid solutions at iteration {iteration}")
                    self.history.append(best_fitness)
                    continue

                # 更新最优和最差
                current_best_idx = min(valid_indices, key=lambda i: fitness_values[i])
                current_worst_idx = max(valid_indices, key=lambda i: fitness_values[i])

                if fitness_values[current_best_idx] < best_fitness:
                    best_solution = population[current_best_idx]
                    best_fitness = fitness_values[current_best_idx]

                worst_solution = population[current_worst_idx]
                worst_fitness = fitness_values[current_worst_idx]

                # 记录历史
                self.history.append(best_fitness)

                if self.verbose and (iteration + 1) % 10 == 0:
                    print(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {best_fitness:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"Error in iteration {iteration}: {e}")
                self.history.append(best_fitness)

        if self.verbose:
            print(f"Optimization completed. Best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness, self.history
