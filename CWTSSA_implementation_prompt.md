# Claude Code Prompt: 实现CWTSSA算法

## 任务概述

请实现一个已发表的改进麻雀搜索算法 **CWTSSA (Chaotic Mapping, Adaptive Weighting, T-distribution mutation SSA)**，用于五目标MEC任务卸载优化，作为RDHO算法的对比基准。

---

## 论文来源

```
Yang, W.; Xia, K.; Li, T.; Xie, M.; Gao, F.
A Novel Adaptive Sparrow Search Algorithm Based on Chaotic Mapping and T-Distribution Mutation.
Applied Sciences 2021, 11, 11192.
https://doi.org/10.3390/app112311192
```

**这是SCIE收录期刊的正式发表论文，可以安全地在学术论文中引用。**

---

## CWTSSA算法核心改进

CWTSSA在标准SSA基础上引入了三项改进：

### 改进1：混沌映射初始化（Chaotic Mapping Initialization）

使用Logistic混沌映射替代随机初始化，增强种群多样性和分布均匀性。

```python
def logistic_chaotic_map(x, mu=4.0):
    """Logistic混沌映射
    
    公式: x_{n+1} = μ * x_n * (1 - x_n)
    当μ=4时，系统处于完全混沌状态
    """
    return mu * x * (1 - x)

def chaotic_init(dim, n_pop, lb, ub):
    """混沌映射初始化种群
    
    参数:
        dim: 决策变量维度
        n_pop: 种群规模
        lb: 下界向量
        ub: 上界向量
    
    返回:
        population: 初始化的种群
    """
    population = []
    
    # 初始化混沌种子（避免边界值）
    x = np.random.uniform(0.001, 0.999, dim)
    
    for _ in range(n_pop):
        # 迭代混沌映射
        x = logistic_chaotic_map(x)
        
        # 映射到搜索空间 [lb, ub]
        solution = lb + x * (ub - lb)
        population.append(solution.copy())
    
    return population
```

### 改进2：自适应权重策略（Adaptive Weighting Strategy）

引入自适应惯性权重来平衡全局探索和局部开发能力。

```python
def adaptive_weight(t, T, w_max=0.9, w_min=0.4):
    """自适应惯性权重
    
    公式: w(t) = w_max - (w_max - w_min) * (t/T)^2
    
    特点：
    - 前期权重大，全局探索能力强
    - 后期权重小，局部开发能力强
    - 二次衰减比线性衰减更平滑
    
    参数:
        t: 当前迭代次数
        T: 最大迭代次数
        w_max: 最大权重
        w_min: 最小权重
    
    返回:
        w: 当前权重值
    """
    progress = (t / T) ** 2  # 二次衰减
    w = w_max - (w_max - w_min) * progress
    return w
```

**应用于发现者位置更新：**
```python
# 发现者位置更新（带自适应权重）
w = adaptive_weight(t, T)

if R2 < ST:  # 安全状态
    # 原始SSA公式：x_new = x * exp(-i / (alpha * T))
    # CWTSSA公式：加入自适应权重
    x_new = w * x + (1 - w) * x_best * np.exp(-t / (alpha * T + eps))
else:  # 危险状态
    x_new = w * x + Q * L  # L为Levy飞行因子
```

### 改进3：自适应t分布变异（Adaptive T-distribution Mutation）

设计自适应t分布变异算子，使用迭代次数t作为自由度参数，动态调整探索与开发能力。

```python
def t_distribution_mutation(x_best, t, T, lb, ub, scale=1.0):
    """自适应t分布变异
    
    t分布特性：
    - 当自由度df小时（迭代初期），分布尾部厚，变异幅度大，利于全局探索
    - 当自由度df大时（迭代后期），分布接近正态，变异幅度小，利于局部开发
    
    公式: x_new = x_best + scale * t_random(df=t)
    
    参数:
        x_best: 当前最优解
        t: 当前迭代次数（作为自由度）
        T: 最大迭代次数
        lb: 下界
        ub: 上界
        scale: 变异幅度缩放因子
    
    返回:
        x_new: 变异后的新解
    """
    # 自由度 = 当前迭代次数 + 1（避免df=0）
    df = t + 1
    
    # 生成t分布随机数
    t_random = np.random.standard_t(df, size=x_best.shape)
    
    # 自适应缩放因子（随迭代递减）
    adaptive_scale = scale * (1 - t / T)
    
    # 生成新解
    x_new = x_best + adaptive_scale * t_random * (ub - lb)
    
    # 边界处理
    x_new = np.clip(x_new, lb, ub)
    
    return x_new
```

---

## 完整算法伪代码

```
Algorithm: CWTSSA for Five-Objective MEC Task Offloading
Input: 系统参数, 最大迭代T, 种群规模N, 发现者比例PD, 侦察者比例SD, 安全阈值ST
Output: 最优解, 最优适应度, 收敛历史

1.  // 改进1: 混沌映射初始化
2.  population ← ChaoticInit(N, dim, lb, ub)
3.  fitness ← Evaluate(population)
4.  
5.  // 找初始最优和最差
6.  x_best ← argmin(fitness)
7.  x_worst ← argmax(fitness)
8.  f_best ← min(fitness)
9.  
10. history ← [f_best]
11. 
12. // 主迭代循环
13. for t = 1 to T:
14.     // 改进2: 计算自适应权重
15.     w ← w_max - (w_max - w_min) * (t/T)^2
16.     
17.     // 按适应度排序，确定角色
18.     sorted_indices ← argsort(fitness)
19.     n_producers ← floor(N * PD)
20.     n_scouts ← floor(N * SD)
21.     
22.     // ===== 发现者更新（带自适应权重）=====
23.     for i = 0 to n_producers-1:
24.         idx ← sorted_indices[i]
25.         R2 ← rand()
26.         
27.         if R2 < ST:  // 安全状态
28.             alpha ← rand()
29.             // 带自适应权重的位置更新
30.             x_new ← w * x[idx] + (1-w) * x_best * exp(-t/(alpha*T+ε))
31.         else:  // 危险状态
32.             Q ← rand()
33.             x_new ← w * x[idx] + Q * exp((x_worst - x[idx]) / t^2)
34.         
35.         x_new ← HandleConstraints(x_new)
36.         f_new ← Evaluate(x_new)
37.         
38.         if f_new < fitness[idx]:
39.             population[idx] ← x_new
40.             fitness[idx] ← f_new
41.     
42.     // ===== 跟随者更新 =====
43.     for i = n_producers to N-n_scouts-1:
44.         idx ← sorted_indices[i]
45.         
46.         // 随机选择一个发现者
47.         producer_idx ← sorted_indices[randint(0, n_producers)]
48.         x_producer ← population[producer_idx]
49.         
50.         A ← random_array(dim) with values in {-1, 1}
51.         A_plus ← A / (norm(A) + ε)
52.         
53.         if i > N/2:  // 饥饿的跟随者
54.             Q ← rand()
55.             x_new ← Q * exp((x_worst - x[idx]) / i^2)
56.         else:  // 正常跟随
57.             L ← rand()
58.             x_new ← x_producer + |x[idx] - x_producer| * A_plus * L
59.         
60.         x_new ← HandleConstraints(x_new)
61.         f_new ← Evaluate(x_new)
62.         
63.         if f_new < fitness[idx]:
64.             population[idx] ← x_new
65.             fitness[idx] ← f_new
66.     
67.     // ===== 侦察者更新 =====
68.     for i = N-n_scouts to N-1:
69.         idx ← sorted_indices[i]
70.         
71.         if fitness[idx] > f_best:  // 当前个体较差
72.             beta ← randn()
73.             x_new ← x_best + beta * |x[idx] - x_best|
74.         else:  // 当前个体较好
75.             K ← rand_uniform(-1, 1)
76.             x_new ← x[idx] + K * |x[idx] - x_worst| / (fitness[idx] - f_worst + ε)
77.         
78.         x_new ← HandleConstraints(x_new)
79.         f_new ← Evaluate(x_new)
80.         
81.         if f_new < fitness[idx]:
82.             population[idx] ← x_new
83.             fitness[idx] ← f_new
84.     
85.     // ===== 改进3: 自适应t分布变异 =====
86.     if rand() < mutation_prob:  // mutation_prob一般设为0.2-0.3
87.         x_mutated ← TDistMutation(x_best, t, T, lb, ub)
88.         x_mutated ← HandleConstraints(x_mutated)
89.         f_mutated ← Evaluate(x_mutated)
90.         
91.         if f_mutated < f_best:
92.             // 替换最差个体
93.             worst_idx ← argmax(fitness)
94.             population[worst_idx] ← x_mutated
95.             fitness[worst_idx] ← f_mutated
96.             x_best ← x_mutated
97.             f_best ← f_mutated
98.     
99.     // 更新全局最优和最差
100.    current_best_idx ← argmin(fitness)
101.    if fitness[current_best_idx] < f_best:
102.        x_best ← population[current_best_idx]
103.        f_best ← fitness[current_best_idx]
104.    
105.    x_worst ← population[argmax(fitness)]
106.    
107.    history.append(f_best)
108.    
109.    if verbose and t % 10 == 0:
110.        print(f"Iteration {t}/{T}, Best fitness: {f_best:.6f}")
111.
112. return x_best, f_best, history
```

---

## 完整代码实现模板

```python
"""
CWTSSA: Chaotic Mapping, Adaptive Weighting, T-distribution mutation SSA
基于论文: A Novel Adaptive Sparrow Search Algorithm Based on Chaotic Mapping 
         and T-Distribution Mutation (Applied Sciences 2021)
用于五目标MEC任务卸载优化
"""
import numpy as np
from typing import List, Tuple, Optional
from .base_algorithm import BaseAlgorithm
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel


class CWTSSA(BaseAlgorithm):
    """CWTSSA - 混沌映射+自适应权重+t分布变异的改进麻雀搜索算法
    
    三大改进策略:
    1. 混沌映射初始化 - 增强种群多样性
    2. 自适应权重策略 - 平衡探索与开发
    3. 自适应t分布变异 - 避免局部最优
    
    Reference:
    Yang, W.; Xia, K.; Li, T.; Xie, M.; Gao, F.
    A Novel Adaptive Sparrow Search Algorithm Based on Chaotic Mapping and T-Distribution Mutation.
    Applied Sciences 2021, 11, 11192.
    https://doi.org/10.3390/app112311192
    """
    
    def __init__(self, 
                 system_model: SystemModel,
                 delay_model: DelayModel,
                 energy_model: EnergyModel,
                 aoi_model: Optional[AoIModel] = None,
                 max_iter: int = 150,
                 population_size: int = 50,
                 # 五目标权重
                 w_energy: float = 0.15,
                 w_delay: float = 0.15,
                 w_aoi: float = 0.20,
                 w_qoe: float = 0.25,
                 w_fairness: float = 0.25,
                 # SSA基本参数
                 PD: float = 0.2,      # 发现者比例
                 SD: float = 0.1,      # 侦察者比例
                 ST: float = 0.8,      # 安全阈值
                 # CWTSSA特有参数
                 w_max: float = 0.9,   # 最大权重
                 w_min: float = 0.4,   # 最小权重
                 mutation_prob: float = 0.2,  # t分布变异概率
                 mutation_scale: float = 0.5,  # 变异幅度
                 verbose: bool = False):
        """初始化CWTSSA算法"""
        super().__init__(system_model, max_iter, population_size, verbose)
        
        # 模型
        self.delay_model = delay_model
        self.energy_model = energy_model
        self.aoi_model = aoi_model
        
        # 权重
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
        
        # 归一化因子
        self.energy_max = 1e5
        self.delay_max = 1e3
        self.aoi_max = 1e2
        
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
        # 初始化随机种子（避免边界值）
        x = np.random.uniform(0.001, 0.999, dim)
        
        # 多次迭代增强混沌性
        for _ in range(10):
            x = self.logistic_chaotic_map(x)
        
        return x
    
    def map_chaos_to_solution(self, chaos_vector: np.ndarray) -> List:
        """将混沌向量映射到决策变量空间"""
        solution = []
        offset = 0
        
        for task_idx in range(self.num_tasks):
            # 位置变量
            loc_chaos = chaos_vector[offset]
            loc_i = int(loc_chaos * (self.loc_bounds[1] - self.loc_bounds[0] + 1))
            loc_i = np.clip(loc_i + self.loc_bounds[0], self.loc_bounds[0], self.loc_bounds[1])
            
            # 频率变量
            freq_chaos = chaos_vector[offset + 1]
            f_i = self.freq_bounds[0] + freq_chaos * (self.freq_bounds[1] - self.freq_bounds[0])
            
            offset += 2
            
            if self.consider_aoi:
                # 更新间隔变量
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
        
        # 确定维度
        if self.consider_aoi:
            dim = self.num_tasks * 3
        else:
            dim = self.num_tasks * 2
        
        for _ in range(self.population_size):
            # 使用混沌映射生成混沌向量
            chaos_vector = self.chaotic_init_vector(dim)
            
            # 映射到决策变量
            solution = self.map_chaos_to_solution(chaos_vector)
            
            # 约束处理
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
        """改进3: 自适应t分布变异
        
        使用当前迭代次数作为自由度参数:
        - 前期自由度小，分布尾部厚，变异幅度大
        - 后期自由度大，分布接近正态，变异小
        """
        x_best_array = np.array(x_best, dtype=float)
        
        # 自由度 = 当前迭代 + 1
        df = self.current_iteration + 1
        
        # 生成t分布随机数
        t_random = np.random.standard_t(df, size=x_best_array.shape)
        
        # 自适应缩放（随迭代递减）
        adaptive_scale = self.mutation_scale * (1 - self.current_iteration / self.max_iter)
        
        # 计算变异范围（基于边界）
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
        
        # 生成变异解
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
                # 安全状态
                alpha = np.random.random()
                exp_factor = np.exp(-self.current_iteration / (alpha * self.max_iter + 1e-10))
                new_position = w * x_array + (1 - w) * best_array * exp_factor
            else:
                # 危险状态
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
                # 饥饿的跟随者
                Q = np.random.random()
                idx_sq = (idx - half_pop + 1) ** 2 + 1e-10
                new_position = Q * np.exp((worst_array - x_array) / idx_sq)
            else:
                # 正常跟随
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
                # 当前个体较差，向最优靠近
                beta = np.random.randn()
                new_position = best_array + beta * np.abs(x_array - best_array)
            else:
                # 当前个体较好，小幅探索
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
            total_energy = 0.0
            total_delay = 0.0
            aoi_values = []
            
            for i, task_solution in enumerate(solution):
                task = self.system_model.tasks[i]
                loc_i = int(round(task_solution[0]))
                f_i = task_solution[1]
                
                # 计算能耗
                energy = self.energy_model.calculate_energy(task, loc_i, f_i)
                total_energy += energy
                
                # 计算时延
                delay = self.delay_model.calculate_total_delay(task, loc_i, f_i)
                total_delay += delay
                
                # 计算AoI
                if self.consider_aoi and len(task_solution) > 2:
                    delta_i = task_solution[2]
                    if self.aoi_model is not None:
                        aoi = self.aoi_model.calculate_average_aoi(delta_i, delay)
                        aoi_values.append(aoi)
            
            # 计算平均AoI
            avg_aoi = np.mean(aoi_values) if aoi_values else 0.0
            
            # 归一化
            norm_energy = total_energy / self.energy_max
            norm_delay = total_delay / self.delay_max
            norm_aoi = avg_aoi / self.aoi_max if self.consider_aoi else 0.0
            
            # 简化的QoE和公平性计算
            qoe = max(0, 1 - norm_delay * 0.5 - norm_aoi * 0.3)
            fairness = self._calculate_fairness(solution)
            
            # 约束违反惩罚
            penalty = self._calculate_penalty(solution)
            
            # 加权适应度
            fitness = (
                self.w_energy * norm_energy +
                self.w_delay * norm_delay +
                self.w_aoi * norm_aoi +
                self.w_qoe * (1 - qoe) +
                self.w_fairness * (1 - fairness) +
                penalty
            )
            
            return fitness
            
        except Exception:
            return float('inf')
    
    def _calculate_fairness(self, solution: List) -> float:
        """计算Jain公平性指数"""
        try:
            resources = []
            for task_solution in solution:
                f_i = task_solution[1]
                resources.append(f_i / self.freq_bounds[1])
            
            if len(resources) == 0 or sum(resources) == 0:
                return 1.0
            
            n = len(resources)
            sum_r = sum(resources)
            sum_r_sq = sum(r**2 for r in resources)
            
            fairness = (sum_r ** 2) / (n * sum_r_sq + 1e-10)
            return min(1.0, max(0.0, fairness))
            
        except Exception:
            return 0.0
    
    def _calculate_penalty(self, solution: List) -> float:
        """计算约束违反惩罚"""
        penalty = 0.0
        
        for i, task_solution in enumerate(solution):
            task = self.system_model.tasks[i]
            loc_i = int(round(task_solution[0]))
            f_i = task_solution[1]
            
            # 时延约束
            delay = self.delay_model.calculate_total_delay(task, loc_i, f_i)
            if hasattr(task, 'deadline') and delay > task.deadline:
                penalty += (delay - task.deadline) / task.deadline
            
            # AoI约束
            if self.consider_aoi and len(task_solution) > 2:
                delta_i = task_solution[2]
                if self.aoi_model is not None:
                    aoi = self.aoi_model.calculate_average_aoi(delta_i, delay)
                    if hasattr(task, 'max_aoi') and aoi > task.max_aoi:
                        penalty += (aoi - task.max_aoi) / task.max_aoi
        
        return penalty
    
    # ==================== 主优化循环 ====================
    
    def optimize(self) -> Tuple[List, float, List]:
        """执行CWTSSA优化"""
        
        # 改进1: 混沌映射初始化
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
        worst_fitness = fitness_values[worst_idx]
        
        self.history = [best_fitness]
        
        # 角色数量
        n_producers = max(1, int(self.population_size * self.PD))
        n_scouts = max(1, int(self.population_size * self.SD))
        
        # 主迭代循环
        for iteration in range(self.max_iter):
            self.current_iteration = iteration
            
            # 按适应度排序
            sorted_indices = sorted(range(len(fitness_values)),
                                   key=lambda i: fitness_values[i])
            
            # ===== 发现者更新 =====
            for i in range(n_producers):
                idx = sorted_indices[i]
                new_solution = self.update_producer(
                    population[idx], best_solution, worst_solution, idx)
                new_fitness = self.evaluate_fitness(new_solution)
                
                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness
            
            # ===== 跟随者更新 =====
            for i in range(n_producers, self.population_size - n_scouts):
                idx = sorted_indices[i]
                producer_idx = sorted_indices[np.random.randint(n_producers)]
                
                new_solution = self.update_scrounger(
                    population[idx], population[producer_idx], worst_solution, i)
                new_fitness = self.evaluate_fitness(new_solution)
                
                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness
            
            # ===== 侦察者更新 =====
            for i in range(self.population_size - n_scouts, self.population_size):
                idx = sorted_indices[i]
                new_solution = self.update_scout(
                    population[idx], best_solution, worst_solution,
                    fitness_values[idx], best_fitness, worst_fitness)
                new_fitness = self.evaluate_fitness(new_solution)
                
                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness
            
            # ===== 改进3: t分布变异 =====
            if np.random.random() < self.mutation_prob:
                mutated = self.t_distribution_mutation(best_solution)
                mutated_fitness = self.evaluate_fitness(mutated)
                
                if np.isfinite(mutated_fitness) and mutated_fitness < best_fitness:
                    # 替换最差个体
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
```

---

## 算法参数建议

| 参数 | 符号 | 建议值 | 说明 |
|------|------|--------|------|
| 种群规模 | N | 50 | 与RDHO一致 |
| 最大迭代 | T | 150 | 与RDHO一致 |
| 发现者比例 | PD | 0.2 | 20%为发现者 |
| 侦察者比例 | SD | 0.1 | 10%为侦察者 |
| 安全阈值 | ST | 0.8 | 标准SSA参数 |
| 最大权重 | w_max | 0.9 | 自适应权重参数 |
| 最小权重 | w_min | 0.4 | 自适应权重参数 |
| 变异概率 | mutation_prob | 0.2 | t分布变异触发概率 |
| 变异幅度 | mutation_scale | 0.5 | t分布变异缩放因子 |

---

## 实验脚本修改

在实验脚本中添加CWTSSA对比：

```python
from src.algorithms.cwtssa import CWTSSA

# 创建CWTSSA实例
cwtssa = CWTSSA(
    system_model=system_model,
    delay_model=delay_model,
    energy_model=energy_model,
    aoi_model=aoi_model,
    max_iter=150,
    population_size=50,
    w_energy=0.15,
    w_delay=0.15,
    w_aoi=0.20,
    w_qoe=0.25,
    w_fairness=0.25,
    verbose=False
)

# 运行优化
best_solution, best_fitness, history = cwtssa.optimize()
```

---

## 论文引用格式

**BibTeX:**
```bibtex
@article{yang2021cwtssa,
  title={A Novel Adaptive Sparrow Search Algorithm Based on Chaotic Mapping and T-Distribution Mutation},
  author={Yang, Wei and Xia, Kewen and Li, Tiejun and Xie, Min and Gao, Feng},
  journal={Applied Sciences},
  volume={11},
  number={23},
  pages={11192},
  year={2021},
  publisher={MDPI},
  doi={10.3390/app112311192}
}
```

**中文引用:**
> Yang等人[x]提出了一种基于混沌映射和t分布变异的自适应麻雀搜索算法(CWTSSA)，通过混沌映射初始化增强种群多样性，利用自适应权重策略平衡探索与开发能力，并引入自适应t分布变异算子避免陷入局部最优。

---

## 验收标准

1. **功能完整**：三项改进策略全部实现
2. **代码规范**：完整注释、类型注解、异常处理
3. **可运行**：能与RDHO在相同实验设置下对比
4. **可复现**：固定随机种子后结果一致

---

请创建文件 `src/algorithms/cwtssa.py`，实现CWTSSA算法，然后将其添加到实验对比中。
