# src/algorithms/mo_tlbo.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .tlbo_plus import TLBOPlus
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel


class MOTLBO(TLBOPlus):
    """多目标教学优化算法（Multi-Objective TLBO）"""
    
    def __init__(self, system_model: SystemModel, 
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: AoIModel,
                 max_iter: int = 100,
                 population_size: int = 50,
                 archive_size: int = 100,
                 levy_alpha: float = 1.5,
                 verbose: bool = False):
        """
        初始化多目标TLBO算法
        
        Parameters:
        -----------
        system_model : SystemModel
            系统模型实例
        delay_model : DelayModel
            延迟模型实例
        energy_model : EnergyModel
            能耗模型实例
        aoi_model : AoIModel
            AoI模型实例
        max_iter : int
            最大迭代次数
        population_size : int
            种群大小（学生数量）
        archive_size : int
            外部档案大小，用于存储非支配解
        levy_alpha : float
            莱维飞行的参数alpha
        verbose : bool
            是否输出详细信息
        """
        super().__init__(system_model, delay_model, energy_model, aoi_model,
                         max_iter, population_size, 0.0, 0.0, 0.0, levy_alpha, verbose)
        
        self.archive_size = archive_size
        self.archive = []  # 外部档案，存储非支配解
        
        # 目标数目
        self.n_obj = 3  # 能耗、延迟、AoI
    
    def evaluate_objectives(self, solution):
        """
        评估解的多个目标函数值
        
        Parameters:
        -----------
        solution : List[List[Union[int, float]]]
            解决方案
            
        Returns:
        --------
        np.ndarray
            目标函数值数组 [energy, delay, aoi]
        """
        # 将解应用到系统模型
        self.system_model.apply_solution(solution)
        
        total_energy = 0.0
        total_delay = 0.0
        total_aoi = 0.0
        
        # 计算所有任务的能耗、延迟和AoI
        for task in self.system_model.tasks:
            # 计算延迟
            delay = self.delay_model.calculate_total_delay(task)
            total_delay += delay
            
            # 计算能耗
            energy = self.energy_model.calculate_total_energy(task)
            total_energy += energy
            
            # 计算AoI
            if task.update_interval is not None:
                aoi = self.aoi_model.calculate_average_aoi(task)
                total_aoi += aoi
        
        return np.array([total_energy, total_delay, total_aoi])
    
    def evaluate_fitness(self, solution):
        """
        评估解的适应度，用于排序
        
        Parameters:
        -----------
        solution : List[List[Union[int, float]]]
            解决方案
            
        Returns:
        --------
        float
            适应度值（越小越好）
        """
        # 对于多目标问题，这里返回目标之和作为标量适应度值
        # 注意：这只用于基本的解排序，不影响多目标优化的Pareto支配关系
        obj = self.evaluate_objectives(solution)
        
        # 归一化目标值并求和
        normalized_obj = obj / np.array([self.energy_max, self.delay_max, self.aoi_max])
        return np.sum(normalized_obj)
    
    def is_dominated(self, obj1, obj2):
        """
        判断obj1是否被obj2支配
        
        Parameters:
        -----------
        obj1 : np.ndarray
            目标函数值数组1
        obj2 : np.ndarray
            目标函数值数组2
            
        Returns:
        --------
        bool
            如果obj1被obj2支配则返回True，否则返回False
        """
        # obj2的所有目标都不劣于obj1的对应目标，且至少一个目标更优
        return np.all(obj2 <= obj1) and np.any(obj2 < obj1)
    
    def crowding_distance(self, objectives):
        """
        计算拥挤度距离
        
        Parameters:
        -----------
        objectives : np.ndarray
            目标函数值数组，形状为 (n_solutions, n_objectives)
            
        Returns:
        --------
        np.ndarray
            拥挤度距离数组
        """
        n_solutions = objectives.shape[0]
        distances = np.zeros(n_solutions)
        
        # 对于每个目标，计算拥挤度
        for i in range(self.n_obj):
            # 按当前目标排序
            idx = np.argsort(objectives[:, i])
            
            # 设置边界点的拥挤度为无穷大
            distances[idx[0]] = float('inf')
            distances[idx[-1]] = float('inf')
            
            # 计算中间点的拥挤度
            if n_solutions > 2:
                obj_range = objectives[idx[-1], i] - objectives[idx[0], i]
                if obj_range > 0:
                    for j in range(1, n_solutions - 1):
                        # src/algorithms/mo_tlbo.py (继续)
                        distances[idx[j]] += (objectives[idx[j+1], i] - objectives[idx[j-1], i]) / obj_range
        
        return distances
    
    def non_dominated_sort(self, population, objectives):
        """
        非支配排序
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
        objectives : np.ndarray
            目标函数值数组，形状为 (population_size, n_objectives)
            
        Returns:
        --------
        List[List]
            分层排序后的解索引
        """
        n_solutions = len(population)
        dominated_count = np.zeros(n_solutions, dtype=int)  # 每个解被支配的次数
        dominating_set = [[] for _ in range(n_solutions)]   # 每个解支配的解的集合
        
        # 计算支配关系
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    if self.is_dominated(objectives[i], objectives[j]):
                        dominated_count[i] += 1
                        dominating_set[j].append(i)
        
        # 分层
        fronts = [[]]
        
        # 找出第一层（非支配解）
        for i in range(n_solutions):
            if dominated_count[i] == 0:
                fronts[0].append(i)
        
        # 找出后续层
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominating_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def select_teacher(self, population, objectives):
        """
        从非支配解集中选择教师
        
        Parameters:
        -----------
        population : List[List[List[Union[int, float]]]]
            当前种群
        objectives : np.ndarray
            目标函数值数组
            
        Returns:
        --------
        List[List[Union[int, float]]]
            选择的教师解
        """
        # 非支配排序
        fronts = self.non_dominated_sort(population, objectives)
        
        # 从第一层（非支配解）中随机选择一个解作为教师
        first_front = fronts[0]
        teacher_idx = np.random.choice(first_front)
        
        return population[teacher_idx]
    
    def update_archive(self, solutions, objectives):
        """
        更新外部档案
        
        Parameters:
        -----------
        solutions : List[List[List[Union[int, float]]]]
            当前解集
        objectives : np.ndarray
            目标函数值数组
            
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            更新后的外部档案
        """
        # 合并当前档案和新解
        combined_solutions = self.archive + solutions
        
        # 计算所有解的目标值
        combined_objectives = np.zeros((len(combined_solutions), self.n_obj))
        for i, solution in enumerate(combined_solutions):
            combined_objectives[i] = self.evaluate_objectives(solution)
        
        # 非支配排序
        fronts = self.non_dominated_sort(combined_solutions, combined_objectives)
        
        # 选择前n个非支配解进入档案
        new_archive = []
        i = 0
        
        while len(new_archive) + len(fronts[i]) <= self.archive_size and i < len(fronts):
            new_archive.extend([combined_solutions[idx] for idx in fronts[i]])
            i += 1
        
        # 如果还有空间，使用拥挤度距离选择最后一层中的解
        if len(new_archive) < self.archive_size and i < len(fronts):
            # 计算最后一层中解的拥挤度距离
            last_front_idx = fronts[i]
            last_front_objectives = combined_objectives[last_front_idx]
            
            # 计算拥挤度距离
            crowding_dist = self.crowding_distance(last_front_objectives)
            
            # 按拥挤度距离降序排序
            sorted_idx = np.argsort(-crowding_dist)
            
            # 选择拥挤度距离最大的解进入档案，直到档案满
            remaining = self.archive_size - len(new_archive)
            last_front_selected = [last_front_idx[i] for i in sorted_idx[:remaining]]
            
            new_archive.extend([combined_solutions[idx] for idx in last_front_selected])
        
        return new_archive
    
    def optimize(self):
        """
        执行多目标TLBO优化
        
        Returns:
        --------
        List[List[List[Union[int, float]]]]
            Pareto最优解集（外部档案）
        """
        # 初始化种群
        population = self.initialize_population()
        
        # 计算目标函数值
        objectives = np.zeros((self.population_size, self.n_obj))
        for i, solution in enumerate(population):
            objectives[i] = self.evaluate_objectives(solution)
            
            # 更新归一化因子
            self.energy_max = max(self.energy_max, objectives[i, 0])
            self.delay_max = max(self.delay_max, objectives[i, 1])
            self.aoi_max = max(self.aoi_max, objectives[i, 2])
        
        # 初始化外部档案
        self.archive = self.update_archive(population, objectives)
        
        for iter_idx in range(self.max_iter):
            new_population = []
            
            # 教师阶段
            teacher = self.select_teacher(population, objectives)
            mean_solution = np.mean(population, axis=0)
            
            for i, student in enumerate(population):
                # 生成莱维飞行步长
                levy_step = self.levy_flight(np.array(student).shape)
                
                # 生成自适应教学因子（简化：这里使用固定值）
                tf = 1 + np.random.random()
                
                # 生成随机权重
                r = np.random.rand(*np.array(student).shape)
                
                # 更新学生
                new_student = np.array(student) + levy_step * r * (np.array(teacher) - tf * mean_solution)
                
                # 处理约束
                new_student = self.handle_constraints(new_student.tolist())
                
                new_population.append(new_student)
            
            # 学习者阶段
            for i, student in enumerate(new_population):
                # 随机选择另一个学生
                j = i
                while j == i:
                    j = np.random.randint(0, self.population_size)
                
                other_student = new_population[j]
                
                # 计算目标函数值
                student_obj = self.evaluate_objectives(student)
                other_obj = self.evaluate_objectives(other_student)
                
                # 判断支配关系
                r = np.random.rand(*np.array(student).shape)
                
                if self.is_dominated(student_obj, other_obj):
                    # 如果student被other_student支配，向other_student学习
                    new_student = np.array(student) + r * (np.array(other_student) - np.array(student))
                else:
                    # 如果student不被支配，远离other_student
                    new_student = np.array(student) + r * (np.array(student) - np.array(other_student))
                
                # 处理约束
                new_student = self.handle_constraints(new_student.tolist())
                
                new_population[i] = new_student
            
            # 更新种群
            population = new_population
            
            # 计算目标函数值
            for i, solution in enumerate(population):
                objectives[i] = self.evaluate_objectives(solution)
                
                # 更新归一化因子
                self.energy_max = max(self.energy_max, objectives[i, 0])
                self.delay_max = max(self.delay_max, objectives[i, 1])
                self.aoi_max = max(self.aoi_max, objectives[i, 2])
            
            # 更新外部档案
            self.archive = self.update_archive(population, objectives)
            
            if self.verbose and (iter_idx + 1) % 10 == 0:
                print(f"Iteration {iter_idx + 1}/{self.max_iter}, Archive size: {len(self.archive)}")
        
        if self.verbose:
            print(f"Optimization completed. Archive size: {len(self.archive)}")
        
        return self.archive