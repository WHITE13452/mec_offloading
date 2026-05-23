"""
花粉授粉算法 (Flower Pollination Algorithm, FPA)
基于花粉授粉行为的元启发式优化算法
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy.special import gamma
from .base_algorithm import BaseAlgorithm
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel


class FPA(BaseAlgorithm):
    """标准花粉授粉算法 - 支持五目标优化"""

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
                 switch_probability: float = 0.8,
                 levy_lambda: float = 1.5,
                 verbose: bool = False):
        """
        初始化FPA算法

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
        switch_probability : float
            全局/局部授粉切换概率 (default=0.8)
        levy_lambda : float
            Lévy飞行参数λ (default=1.5)
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

        # FPA参数
        self.switch_probability = switch_probability  # 切换概率p
        self.levy_lambda = levy_lambda  # Lévy飞行参数

        # 归一化因子
        self.energy_max = 1e5  # 100,000 J
        self.delay_max = 1e3   # 1,000 s
        self.aoi_max = 1e2     # 100 s

        # 迭代历史
        self.history = []

        # Debug计数器
        self._eval_count = 0
        self._debug_first_eval = True

    def initialize_population(self):
        """初始化种群（花粉）"""
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

    def levy_flight(self, dim: int) -> np.ndarray:
        """
        生成Lévy飞行步长

        基于Mantegna方法:
        L(λ) ~ λΓ(λ)sin(πλ/2) / (π * s^(1+λ))

        Parameters:
        -----------
        dim : int
            维度

        Returns:
        --------
        np.ndarray : Lévy飞行步长向量
        """
        try:
            lambda_param = self.levy_lambda

            # 计算sigma_u和sigma_v
            numerator = gamma(1 + lambda_param) * np.sin(np.pi * lambda_param / 2)
            denominator = gamma((1 + lambda_param) / 2) * lambda_param * (2 ** ((lambda_param - 1) / 2))

            sigma_u = (numerator / denominator) ** (1 / lambda_param)
            sigma_v = 1.0

            # 生成u和v
            u = np.random.normal(0, sigma_u, dim)
            v = np.random.normal(0, sigma_v, dim)

            # 计算Lévy飞行步长
            step = u / (np.abs(v) ** (1 / lambda_param))

            return step

        except Exception:
            # 如果出错，返回标准正态分布
            return np.random.normal(0, 0.01, dim)

    def global_pollination(self, pollen: List, best_pollen: List) -> List:
        """
        全局授粉 - 使用Lévy飞行

        公式: x_i^{new} = x_i + L(λ) * (x_best - x_i)

        Parameters:
        -----------
        pollen : List
            当前花粉(解)
        best_pollen : List
            全局最优花粉

        Returns:
        --------
        List : 更新后的花粉
        """
        try:
            pollen_array = np.array(pollen, dtype=float)
            best_array = np.array(best_pollen, dtype=float)

            # 获取维度
            dim = pollen_array.size

            # 生成Lévy飞行步长
            levy_step = self.levy_flight(dim)

            # 将步长reshape为与解相同的形状
            levy_step = levy_step.reshape(pollen_array.shape)

            # 修复：添加步长缩放因子，限制最大步长，避免步长过大导致探索过度
            step_scale = 0.01  # 缩放因子
            levy_step = np.clip(levy_step * step_scale, -1.0, 1.0)  # 限制步长范围

            # 全局授粉更新
            new_pollen_array = pollen_array + levy_step * (best_array - pollen_array)

            # 转换回列表格式
            new_pollen = new_pollen_array.tolist()

            # 处理约束
            new_pollen = self.handle_constraints(new_pollen)

            return new_pollen

        except Exception:
            return pollen

    def local_pollination(self, pollen: List, population: List[List]) -> List:
        """
        局部授粉 - 邻域内花粉交互

        公式: x_i^{new} = x_i + ε * (x_j - x_k)

        Parameters:
        -----------
        pollen : List
            当前花粉
        population : List[List]
            当前种群

        Returns:
        --------
        List : 更新后的花粉
        """
        try:
            pollen_array = np.array(pollen, dtype=float)

            # 随机选择两个不同的花粉
            indices = np.random.choice(len(population), 2, replace=False)
            pollen_j = np.array(population[indices[0]], dtype=float)
            pollen_k = np.array(population[indices[1]], dtype=float)

            # 随机缩放因子ε ∈ [0, 1]
            epsilon = np.random.uniform(0, 1)

            # 局部授粉更新
            new_pollen_array = pollen_array + epsilon * (pollen_j - pollen_k)

            # 转换回列表格式
            new_pollen = new_pollen_array.tolist()

            # 处理约束
            new_pollen = self.handle_constraints(new_pollen)

            return new_pollen

        except Exception:
            return pollen

    def evaluate_fitness(self, solution):
        """
        评估五目标适应度

        公式: F(X) = w_E*E/E_norm + w_T*T/T_norm + w_AoI*AoI/AoI_norm
                   - w_QoE*QoE - w_F*Fairness + Penalty

        Returns:
        --------
        float : 综合适应度值（越小越好）
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

            # 动态更新归一化因子（与TLBO-HHO保持一致）
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

            # 约束惩罚（降低惩罚系数，与TLBO-HHO保持一致的数量级）
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
                print(f"[FPA DEBUG] Energy: {total_energy:.2f}, Delay: {total_delay:.4f}, "
                      f"AoI: {avg_aoi:.4f}, QoE: {qoe:.4f}, Fairness: {fairness:.4f}, "
                      f"Violations: {constraint_violations}, Penalty: {penalty:.4f}, "
                      f"Fitness: {fitness:.6f}")
                self._debug_first_eval = False

            return fitness if np.isfinite(fitness) else float('inf')

        except Exception:
            return float('inf')

    def optimize(self):
        """
        执行FPA优化

        Returns:
        --------
        Tuple[List, float, List] : (最优解, 最优适应度, 迭代历史)
        """
        # 初始化种群
        population = self.initialize_population()

        # 评估初始种群
        fitness_values = [self.evaluate_fitness(sol) for sol in population]

        # 找到初始最优解
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            if self.verbose:
                print("Warning: No valid solutions in initial population!")
            return None, float('inf'), []

        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        self.best_solution = population[best_idx]
        self.best_fitness = fitness_values[best_idx]

        # 初始化迭代历史
        self.history = [self.best_fitness]

        # 主迭代循环
        for iteration in range(self.max_iter):
            try:
                new_population = []

                for i, pollen in enumerate(population):
                    # 以概率p进行全局授粉，否则进行局部授粉
                    if np.random.random() < self.switch_probability:
                        # 全局授粉
                        new_pollen = self.global_pollination(pollen, self.best_solution)
                    else:
                        # 局部授粉
                        new_pollen = self.local_pollination(pollen, population)

                    # 评估新解
                    new_fitness = self.evaluate_fitness(new_pollen)

                    # 贪婪选择
                    if np.isfinite(new_fitness) and new_fitness < fitness_values[i]:
                        new_population.append(new_pollen)
                        fitness_values[i] = new_fitness

                        # 更新全局最优
                        if new_fitness < self.best_fitness:
                            self.best_solution = new_pollen
                            self.best_fitness = new_fitness
                    else:
                        new_population.append(pollen)

                population = new_population

                # 记录历史
                self.history.append(self.best_fitness)

                if self.verbose and (iteration + 1) % 10 == 0:
                    print(f"Iteration {iteration + 1}/{self.max_iter}, "
                          f"Best fitness: {self.best_fitness:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"Error in iteration {iteration}: {e}")
                self.history.append(self.best_fitness)

        if self.verbose:
            print(f"Optimization completed. Best fitness: {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness, self.history
