"""
FPA-TS: 花粉授粉算法与禁忌搜索的混合算法
用于MEC五目标任务卸载优化
"""
import numpy as np
import hashlib
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
from .fpa import FPA
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel


class FPATS(FPA):
    """FPA-TS混合算法 - 五目标优化"""

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
                 tabu_tenure: int = 15,
                 tabu_trigger_freq: int = 20,
                 aspiration_threshold: float = 0.95,
                 verbose: bool = False):
        """
        初始化FPA-TS算法

        Parameters:
        -----------
        # FPA参数
        switch_probability : float
            基础切换概率 (default=0.8)
        levy_lambda : float
            Lévy飞行参数 (default=1.5)

        # 禁忌搜索参数
        tabu_tenure : int
            禁忌期限，解在禁忌表中保持的迭代次数 (default=15)
        tabu_trigger_freq : int
            TS触发频率，每N次迭代触发一次TS (default=20，修复后降低触发频率)
        aspiration_threshold : float
            渴望准则阈值 (default=0.95)

        # 五目标权重
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
        """
        super().__init__(system_model, delay_model, energy_model,
                        aoi_model, qoe_model, fairness_model,
                        max_iter, population_size,
                        w_energy, w_delay, w_aoi, w_qoe, w_fairness,
                        switch_probability, levy_lambda, verbose)

        # 禁忌搜索参数
        self.tabu_tenure = tabu_tenure
        self.tabu_trigger_freq = tabu_trigger_freq
        self.aspiration_threshold = aspiration_threshold

        # 禁忌表: 使用deque实现FIFO，自动管理期限
        self.tabu_list = deque(maxlen=tabu_tenure)

        # 全局最优解记录
        self.global_best_solution = None
        self.global_best_fitness = float('inf')

    def calculate_diversity(self, population: List[List]) -> float:
        """
        计算种群多样性

        使用种群中个体之间的平均距离来度量

        Returns:
        --------
        float : 种群多样性指标 [0, 1]
        """
        try:
            if len(population) < 2:
                return 0.0

            # 计算所有个体对之间的距离
            distances = []
            pop_arrays = [np.array(ind, dtype=float).flatten() for ind in population]

            for i in range(len(pop_arrays)):
                for j in range(i + 1, len(pop_arrays)):
                    # 欧氏距离
                    dist = np.linalg.norm(pop_arrays[i] - pop_arrays[j])
                    distances.append(dist)

            if not distances:
                return 0.0

            # 平均距离
            avg_distance = np.mean(distances)

            # 归一化到[0, 1]
            # 使用搜索空间的对角线长度作为参考
            max_distance = np.sqrt(self.num_tasks * 2)  # 粗略估计
            diversity = min(1.0, avg_distance / max_distance)

            return diversity

        except Exception:
            return 0.5  # 默认中等多样性

    def adaptive_switch_probability(self, iteration: int, diversity: float) -> float:
        """
        自适应切换概率

        根据迭代进度和种群多样性动态调整
        - 早期+高多样性：增加全局授粉概率（探索）
        - 后期+低多样性：增加局部授粉概率（利用）

        Parameters:
        -----------
        iteration : int
            当前迭代次数
        diversity : float
            种群多样性

        Returns:
        --------
        float : 自适应切换概率
        """
        try:
            # 基础概率
            base_p = self.switch_probability

            # 迭代进度因子 [1.0 -> 0.0]
            progress = iteration / self.max_iter

            # 根据进度调整：早期更多全局搜索，后期更多局部搜索
            progress_factor = 1.0 - 0.5 * progress

            # 根据多样性调整：低多样性时增加全局搜索
            diversity_factor = 1.0 + 0.3 * (0.5 - diversity)

            # 自适应概率
            adaptive_p = base_p * progress_factor * diversity_factor

            # 确保在合理范围内
            adaptive_p = np.clip(adaptive_p, 0.3, 0.95)

            return adaptive_p

        except Exception:
            return self.switch_probability

    def solution_hash(self, solution: List) -> str:
        """
        计算解的哈希值，用于禁忌表

        Returns:
        --------
        str : 解的哈希标识
        """
        try:
            # 修复：考虑位置和资源分配（量化到整数），避免过于粗糙的哈希
            hash_components = []
            for s in solution:
                loc = int(s[0])
                freq_quantized = int(s[1] * 100)  # 量化频率到整数（保留两位小数）
                hash_components.append(f"{loc}:{freq_quantized}")

            solution_str = ','.join(hash_components)

            # 使用MD5哈希
            hash_obj = hashlib.md5(solution_str.encode())
            return hash_obj.hexdigest()[:16]  # 只取前16位，减少存储开销

        except Exception:
            return str(id(solution))

    def is_tabu(self, solution: List) -> bool:
        """
        检查解是否在禁忌表中

        Returns:
        --------
        bool : 是否被禁忌
        """
        try:
            sol_hash = self.solution_hash(solution)
            return sol_hash in self.tabu_list

        except Exception:
            return False

    def satisfies_aspiration(self, solution: List, fitness: float) -> bool:
        """
        检查是否满足渴望准则

        即使解在禁忌表中，如果其适应度显著优于当前最优解，
        则可以被接受

        Parameters:
        -----------
        solution : List
            候选解
        fitness : float
            候选解适应度

        Returns:
        --------
        bool : 是否满足渴望准则
        """
        try:
            if self.global_best_fitness == float('inf'):
                return True

            # 如果新解的适应度优于当前最优解的95%，则接受
            threshold = self.global_best_fitness * self.aspiration_threshold

            return fitness < threshold

        except Exception:
            return False

    def add_to_tabu_list(self, solution: List):
        """将解添加到禁忌表"""
        try:
            sol_hash = self.solution_hash(solution)
            # deque会自动管理最大长度，旧元素会自动移除
            self.tabu_list.append(sol_hash)

        except Exception:
            pass

    def generate_neighborhood(self, solution: List, neighborhood_size: int = 10) -> List[List]:
        """
        生成解的邻域

        邻域操作包括:
        1. 改变任务的执行位置 (local/edge/cloud)
        2. 调整资源分配
        3. 交换两个任务的执行位置

        Parameters:
        -----------
        solution : List
            当前解
        neighborhood_size : int
            邻域大小

        Returns:
        --------
        List[List] : 邻域解列表
        """
        neighbors = []

        try:
            for _ in range(neighborhood_size):
                # 复制当前解
                neighbor = [s[:] for s in solution]

                # 随机选择邻域操作
                operation = np.random.choice(['change_location', 'adjust_resource', 'swap'])

                if operation == 'change_location':
                    # 改变一个任务的执行位置
                    task_idx = np.random.randint(0, len(neighbor))
                    new_loc = np.random.randint(self.loc_bounds[0], self.loc_bounds[1] + 1)
                    neighbor[task_idx][0] = new_loc

                elif operation == 'adjust_resource':
                    # 调整一个任务的资源分配
                    task_idx = np.random.randint(0, len(neighbor))
                    perturbation = np.random.uniform(-0.2, 0.2) * neighbor[task_idx][1]
                    neighbor[task_idx][1] += perturbation

                elif operation == 'swap':
                    # 交换两个任务的执行位置
                    if len(neighbor) >= 2:
                        idx1, idx2 = np.random.choice(len(neighbor), 2, replace=False)
                        neighbor[idx1][0], neighbor[idx2][0] = neighbor[idx2][0], neighbor[idx1][0]

                # 处理约束
                neighbor = self.handle_constraints(neighbor)
                neighbors.append(neighbor)

        except Exception:
            pass

        return neighbors

    def tabu_search(self, current_solution: List, current_fitness: float) -> Tuple[List, float]:
        """
        禁忌搜索过程

        流程:
        1. 生成当前解的邻域
        2. 筛选不在禁忌表中的候选解（或满足渴望准则）
        3. 选择最优候选解
        4. 更新禁忌表

        Parameters:
        -----------
        current_solution : List
            当前解
        current_fitness : float
            当前适应度

        Returns:
        --------
        Tuple[List, float] : (最优邻域解, 对应适应度)
        """
        try:
            # 生成邻域
            neighbors = self.generate_neighborhood(current_solution, neighborhood_size=20)

            if not neighbors:
                return current_solution, current_fitness

            # 评估邻域解并筛选
            best_neighbor = None
            best_neighbor_fitness = float('inf')

            for neighbor in neighbors:
                # 评估适应度
                neighbor_fitness = self.evaluate_fitness(neighbor)

                if not np.isfinite(neighbor_fitness):
                    continue

                # 检查禁忌状态
                is_tabu_solution = self.is_tabu(neighbor)

                # 如果不在禁忌表中，或满足渴望准则，则考虑该解
                if not is_tabu_solution or self.satisfies_aspiration(neighbor, neighbor_fitness):
                    if neighbor_fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = neighbor_fitness

            # 如果找到了更好的邻域解
            if best_neighbor is not None and best_neighbor_fitness < float('inf'):
                # 将新解加入禁忌表
                self.add_to_tabu_list(best_neighbor)
                return best_neighbor, best_neighbor_fitness
            else:
                return current_solution, current_fitness

        except Exception:
            return current_solution, current_fitness

    def tabu_enhanced_local_pollination(self, pollen: List, population: List[List]) -> List:
        """
        禁忌增强的局部授粉

        在标准局部授粉基础上，检查禁忌表避免循环搜索

        Returns:
        --------
        List : 更新后的花粉
        """
        try:
            # 执行标准局部授粉
            new_pollen = self.local_pollination(pollen, population)

            # 检查是否在禁忌表中
            if self.is_tabu(new_pollen):
                # 如果在禁忌表中，尝试小幅度扰动
                for task_idx in range(len(new_pollen)):
                    perturbation = np.random.uniform(-0.1, 0.1)
                    new_pollen[task_idx][1] *= (1 + perturbation)

                new_pollen = self.handle_constraints(new_pollen)

            return new_pollen

        except Exception:
            return pollen

    def memory_guided_global_pollination(self, pollen: List, best_pollen: List) -> List:
        """
        记忆导向的全局授粉

        检查预期搜索方向是否会导向禁忌区域，
        必要时调整搜索方向

        Returns:
        --------
        List : 更新后的花粉
        """
        try:
            # 执行标准全局授粉
            new_pollen = self.global_pollination(pollen, best_pollen)

            # 检查是否在禁忌表中
            if self.is_tabu(new_pollen):
                # 修复：使用小幅随机扰动，而不是混合完全随机的解
                # 避免破坏解的质量，同时避开禁忌区域
                new_pollen_array = np.array(new_pollen, dtype=float)
                for i in range(len(new_pollen)):
                    # 只对位置做小幅扰动（30%概率）
                    if np.random.random() < 0.3:
                        perturbation = np.random.randint(-2, 3)  # [-2, 2]范围内的整数扰动
                        new_pollen_array[i][0] = np.clip(
                            new_pollen_array[i][0] + perturbation,
                            self.loc_bounds[0], self.loc_bounds[1]
                        )
                new_pollen = self.handle_constraints(new_pollen_array.tolist())

            return new_pollen

        except Exception:
            return pollen

    def optimize(self):
        """
        执行FPA-TS优化

        主循环流程:
        1. 初始化种群和禁忌表
        2. 计算种群多样性和自适应切换概率
        3. 对每个个体执行全局或局部授粉
        4. 每隔N次迭代执行禁忌搜索
        5. 更新最优解和禁忌表

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
        self.global_best_solution = population[best_idx]
        self.global_best_fitness = fitness_values[best_idx]

        # 初始化迭代历史
        history = [self.global_best_fitness]

        # 主迭代循环
        for iteration in range(self.max_iter):
            try:
                # 计算种群多样性
                diversity = self.calculate_diversity(population)

                # 自适应切换概率
                p = self.adaptive_switch_probability(iteration, diversity)

                # FPA主循环
                new_population = []
                for i, pollen in enumerate(population):
                    if np.random.random() < p:
                        # 全局授粉（记忆导向）
                        new_pollen = self.memory_guided_global_pollination(
                            pollen, self.global_best_solution)
                    else:
                        # 局部授粉（禁忌增强）
                        new_pollen = self.tabu_enhanced_local_pollination(
                            pollen, population)

                    # 约束处理
                    new_pollen = self.handle_constraints(new_pollen)

                    # 评估新解
                    new_fitness = self.evaluate_fitness(new_pollen)

                    # 贪婪选择
                    if np.isfinite(new_fitness) and new_fitness < fitness_values[i]:
                        new_population.append(new_pollen)
                        fitness_values[i] = new_fitness

                        # 更新全局最优
                        if new_fitness < self.global_best_fitness:
                            self.global_best_solution = new_pollen
                            self.global_best_fitness = new_fitness
                    else:
                        new_population.append(pollen)

                population = new_population

                # 禁忌搜索触发
                if (iteration + 1) % self.tabu_trigger_freq == 0:
                    ts_solution, ts_fitness = self.tabu_search(
                        self.global_best_solution, self.global_best_fitness)

                    if ts_fitness < self.global_best_fitness:
                        self.global_best_solution = ts_solution
                        self.global_best_fitness = ts_fitness

                # 更新禁忌表
                self.add_to_tabu_list(self.global_best_solution)

                # 记录历史
                history.append(self.global_best_fitness)

                if self.verbose and (iteration + 1) % 10 == 0:
                    print(f"Iteration {iteration+1}/{self.max_iter}, "
                          f"Best fitness: {self.global_best_fitness:.6f}, "
                          f"Diversity: {diversity:.4f}, "
                          f"Switch prob: {p:.3f}")

            except Exception as e:
                if self.verbose:
                    print(f"Error in iteration {iteration}: {e}")
                history.append(self.global_best_fitness)

        if self.verbose:
            print(f"Optimization completed. Best fitness: {self.global_best_fitness:.6f}")

        return self.global_best_solution, self.global_best_fitness, history
