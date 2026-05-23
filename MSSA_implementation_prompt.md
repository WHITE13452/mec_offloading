# Claude Code Prompt: 实现MSSA算法（Modified Sparrow Search Algorithm）

## 任务概述

请实现一个已发表的改进麻雀搜索算法 **MSSA (Modified Sparrow Search Algorithm)**，用于五目标MEC任务卸载优化，作为RDHO算法的对比基准。

---

## 论文来源

```
Alseid, M., El-Moursy, A.A., Alfawaz, O. et al. 
MSSAMTO-IoV: modified sparrow search algorithm for multi-hop task offloading for IoV. 
Journal of Supercomputing 79, 20769–20789 (2023). 
https://doi.org/10.1007/s11227-023-05446-2
```

---

## MSSA算法核心改进

MSSA在标准SSA基础上引入了四项改进：

### 改进1：Logistic Map混沌初始化

替代随机初始化，使用Logistic混沌映射生成初始种群，增强种群多样性和分布均匀性。

```python
def logistic_map(x, mu=4.0):
    """Logistic混沌映射"""
    return mu * x * (1 - x)

def logistic_init(dim, n_pop):
    """使用Logistic混沌映射初始化种群"""
    population = []
    
    # 初始化混沌种子
    x = np.random.rand(dim)
    x = np.clip(x, 0.001, 0.999)  # 避免边界值
    
    for _ in range(n_pop):
        # 迭代混沌映射
        for d in range(dim):
            x[d] = logistic_map(x[d])
        
        # 映射到搜索空间 [lb, ub]
        solution = lb + x * (ub - lb)
        population.append(solution.copy())
    
    return population
```

### 改进2：自适应惯性权重

引入惯性权重来平衡全局探索和局部开发能力，惯性权重随迭代线性递减。

```python
def adaptive_inertia_weight(t, T, w_max=0.9, w_min=0.4):
    """自适应惯性权重，随迭代线性递减"""
    w = w_max - (w_max - w_min) * (t / T)
    return w
```

**应用于发现者更新：**
```python
# 发现者位置更新（带惯性权重）
w = adaptive_inertia_weight(t, T)
if rand < ST:  # 安全值
    x_new = w * x + (1 - w) * (x_best + alpha * abs(x - x_best))
else:
    x_new = w * x + Q * np.exp(-iter / (rand * T + eps))
```

### 改进3：均值终止准则

当种群适应度均值变化小于阈值时提前终止，避免不必要的迭代。

```python
def mean_termination_check(history, window=10, threshold=1e-6):
    """均值终止准则"""
    if len(history) < window:
        return False
    
    recent = history[-window:]
    mean_change = abs(np.mean(recent[:window//2]) - np.mean(recent[window//2:]))
    
    return mean_change < threshold
```

### 改进4：变异策略

对陷入局部最优的个体进行变异操作，帮助跳出局部最优。

```python
def mutation_operator(x, x_best, mutation_rate=0.1, scale=0.5):
    """变异操作"""
    if np.random.rand() < mutation_rate:
        # 高斯变异
        mutation = np.random.randn(*x.shape) * scale
        x_new = x + mutation * (x_best - x)
        return x_new
    return x
```

---

## 完整算法伪代码

```
Algorithm: MSSA for Five-Objective MEC Task Offloading
Input: 系统参数, 最大迭代T, 种群规模N, 发现者比例PD, 安全阈值ST
Output: 最优解, 最优适应度, 收敛历史

1.  // 改进1: Logistic混沌初始化
2.  population ← LogisticMapInit(N, dim)
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
14.     // 改进2: 计算自适应惯性权重
15.     w ← w_max - (w_max - w_min) * (t / T)
16.     
17.     // 角色分配
18.     n_producers ← floor(N * PD)
19.     
20.     // 发现者更新（带惯性权重）
21.     for i = 1 to n_producers:
22.         R2 ← rand()
23.         if R2 < ST:
24.             alpha ← rand()
25.             x_new ← w * x_i + (1-w) * (x_best + alpha * |x_i - x_best|)
26.         else:
27.             Q ← rand()
28.             x_new ← w * x_i + Q * exp(-t / (rand()*T + ε))
29.         
30.         x_new ← HandleConstraints(x_new)
31.         f_new ← Evaluate(x_new)
32.         
33.         if f_new < fitness[i]:
34.             population[i] ← x_new
35.             fitness[i] ← f_new
36.     
37.     // 跟随者更新
38.     for i = n_producers+1 to N:
39.         A ← random_array(dim) with values in {-1, 1}
40.         A_plus ← A' * (A * A')^(-1)
41.         
42.         if i > N/2:
43.             Q ← rand()
44.             x_new ← Q * exp((x_worst - x_i) / i^2)
45.         else:
46.             x_best_producer ← population[random_producer]
47.             x_new ← x_best_producer + |x_i - x_best_producer| * A_plus * L
48.         
49.         x_new ← HandleConstraints(x_new)
50.         f_new ← Evaluate(x_new)
51.         
52.         if f_new < fitness[i]:
53.             population[i] ← x_new
54.             fitness[i] ← f_new
55.     
56.     // 侦察者更新
57.     for k random individuals:
58.         if fitness[k] > f_best:
59.             beta ← randn()
60.             x_new ← x_best + beta * |x_k - x_best|
61.         else:
62.             K ← rand() in [-1, 1]
63.             x_new ← x_k + K * (|x_k - x_worst|) / (fitness[k] - f_worst + ε)
64.         
65.         x_new ← HandleConstraints(x_new)
66.         f_new ← Evaluate(x_new)
67.         
68.         if f_new < fitness[k]:
69.             population[k] ← x_new
70.             fitness[k] ← f_new
71.     
72.     // 改进4: 对部分个体进行变异
73.     for each individual with stagnation:
74.         population[i] ← MutationOperator(population[i], x_best)
75.         fitness[i] ← Evaluate(population[i])
76.     
77.     // 更新全局最优
78.     if min(fitness) < f_best:
79.         x_best ← argmin(fitness)
80.         f_best ← min(fitness)
81.     
82.     history.append(f_best)
83.     
84.     // 改进3: 均值终止检查
85.     if MeanTerminationCheck(history):
86.         break
87. 
88. return x_best, f_best, history
```

---

## 代码实现模板

```python
"""
MSSA: Modified Sparrow Search Algorithm
基于论文: MSSAMTO-IoV (Journal of Supercomputing, 2023)
用于五目标MEC任务卸载优化
"""
import numpy as np
from typing import List, Tuple, Optional
from .ssa import SSA  # 继承自基础SSA
from ..models.system_model import SystemModel
from ..models.delay_model import DelayModel
from ..models.energy_model import EnergyModel
from ..models.aoi_model import AoIModel
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel


class MSSA(SSA):
    """Modified Sparrow Search Algorithm (MSSA) - 改进麻雀搜索算法
    
    改进策略:
    1. Logistic Map混沌初始化
    2. 自适应惯性权重
    3. 均值终止准则
    4. 变异策略
    
    Reference:
    Alseid et al. "MSSAMTO-IoV: modified sparrow search algorithm for 
    multi-hop task offloading for IoV." J Supercomput 79, 20769–20789 (2023)
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
                 # 五目标权重
                 w_energy: float = 0.15,
                 w_delay: float = 0.15,
                 w_aoi: float = 0.20,
                 w_qoe: float = 0.25,
                 w_fairness: float = 0.25,
                 # SSA参数
                 PD: float = 0.2,      # 发现者比例
                 SD: float = 0.1,      # 侦察者比例
                 ST: float = 0.8,      # 安全阈值
                 # MSSA特有参数
                 w_max: float = 0.9,   # 最大惯性权重
                 w_min: float = 0.4,   # 最小惯性权重
                 mutation_rate: float = 0.1,  # 变异概率
                 mutation_scale: float = 0.5,  # 变异幅度
                 termination_window: int = 20,  # 终止检查窗口
                 termination_threshold: float = 1e-6,  # 终止阈值
                 verbose: bool = False):
        """
        初始化MSSA算法
        """
        # 调用父类初始化
        super().__init__(system_model, delay_model, energy_model,
                        aoi_model, qoe_model, fairness_model,
                        max_iter, population_size,
                        w_energy, w_delay, w_aoi, w_qoe, w_fairness,
                        PD, SD, ST, verbose)
        
        # MSSA特有参数
        self.w_max = w_max
        self.w_min = w_min
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.termination_window = termination_window
        self.termination_threshold = termination_threshold
        
        # 当前迭代
        self.current_iteration = 0
    
    def logistic_map(self, x: float, mu: float = 4.0) -> float:
        """Logistic混沌映射"""
        return mu * x * (1 - x)
    
    def logistic_init(self, dim: int) -> np.ndarray:
        """使用Logistic混沌映射生成一个解向量"""
        # 初始化随机种子
        x = np.random.rand(dim)
        x = np.clip(x, 0.001, 0.999)  # 避免边界值导致混沌失效
        
        # 多次迭代增强混沌性
        for _ in range(10):
            for d in range(dim):
                x[d] = self.logistic_map(x[d])
        
        return x
    
    def map_chaos_to_solution(self, chaos_vector: np.ndarray) -> List:
        """将混沌向量映射到决策变量空间"""
        solution = []
        offset = 0
        
        for task_idx in range(self.num_tasks):
            # 位置变量
            loc_chaos = chaos_vector[offset]
            loc_i = int(loc_chaos * (self.loc_bounds[1] - self.loc_bounds[0] + 1)) + self.loc_bounds[0]
            loc_i = np.clip(loc_i, self.loc_bounds[0], self.loc_bounds[1])
            
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
        """改进1: Logistic Map混沌初始化"""
        population = []
        
        # 确定维度
        if self.consider_aoi:
            dim = self.num_tasks * 3
        else:
            dim = self.num_tasks * 2
        
        for _ in range(self.population_size):
            # 使用Logistic混沌映射生成混沌向量
            chaos_vector = self.logistic_init(dim)
            
            # 映射到决策变量
            solution = self.map_chaos_to_solution(chaos_vector)
            
            # 约束处理
            solution = self.handle_constraints(solution)
            population.append(solution)
        
        return population
    
    def adaptive_inertia_weight(self) -> float:
        """改进2: 自适应惯性权重"""
        progress = self.current_iteration / self.max_iter
        w = self.w_max - (self.w_max - self.w_min) * progress
        return w
    
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
    
    def mutation_operator(self, x: List, x_best: List) -> List:
        """改进4: 变异操作"""
        if np.random.rand() < self.mutation_rate:
            x_array = np.array(x, dtype=float)
            best_array = np.array(x_best, dtype=float)
            
            # 高斯变异
            mutation = np.random.randn(*x_array.shape) * self.mutation_scale
            x_new = x_array + mutation * (best_array - x_array)
            
            return self.handle_constraints(x_new.tolist())
        return x
    
    def update_producer(self, producer: List, iteration: int, 
                        best_solution: List) -> List:
        """发现者更新（带惯性权重）"""
        try:
            producer_array = np.array(producer, dtype=float)
            best_array = np.array(best_solution, dtype=float)
            
            # 获取自适应惯性权重
            w = self.adaptive_inertia_weight()
            
            R2 = np.random.random()
            
            if R2 < self.ST:
                # 安全状态，向最优解靠近
                alpha = np.random.random()
                # 带惯性权重的位置更新
                new_position = (w * producer_array + 
                               (1 - w) * (best_array + alpha * np.abs(producer_array - best_array)))
            else:
                # 危险状态，随机逃逸
                Q = np.random.random()
                rand_factor = np.random.random() * self.max_iter + 1e-10
                new_position = w * producer_array + Q * np.exp(-iteration / rand_factor)
            
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
                # 饥饿的跟随者，随机觅食
                Q = np.random.random()
                idx_sq = (idx - half_pop + 1) ** 2 + 1e-10
                new_position = Q * np.exp((worst_array - scrounger_array) / idx_sq)
            else:
                # 跟随发现者
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
                # 当前个体较差，向最优靠近
                beta = np.random.randn()
                new_position = best_array + beta * np.abs(scout_array - best_array)
            else:
                # 当前个体较好，小幅探索
                K = np.random.uniform(-1, 1)
                denom = abs(scout_fitness - worst_fitness) + 1e-10
                new_position = scout_array + K * np.abs(scout_array - worst_array) / denom
            
            return self.handle_constraints(new_position.tolist())
            
        except Exception:
            return scout
    
    def optimize(self) -> Tuple[List, float, List]:
        """执行MSSA优化"""
        
        # 改进1: Logistic混沌初始化
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
            
            # 对种群按适应度排序
            sorted_indices = sorted(range(len(fitness_values)),
                                   key=lambda i: fitness_values[i])
            
            # 发现者更新
            for i in range(n_producers):
                idx = sorted_indices[i]
                new_solution = self.update_producer(
                    population[idx], iteration, best_solution)
                new_fitness = self.evaluate_fitness(new_solution)
                
                # 贪婪选择
                if np.isfinite(new_fitness) and new_fitness < fitness_values[idx]:
                    population[idx] = new_solution
                    fitness_values[idx] = new_fitness
            
            # 跟随者更新
            for i in range(n_producers, self.population_size - n_scouts):
                idx = sorted_indices[i]
                # 随机选择一个发现者
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
            
            # 改进4: 变异操作（对停滞个体）
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
            
            # 改进3: 均值终止检查
            if self.mean_termination_check(self.history):
                if self.verbose:
                    print(f"Early termination at iteration {iteration + 1}")
                break
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best fitness: {best_fitness:.6f}")
        
        if self.verbose:
            print(f"Optimization completed. Best fitness: {best_fitness:.6f}")
        
        return best_solution, best_fitness, self.history
```

---

## 算法参数建议

| 参数 | 符号 | 建议值 | 说明 |
|------|------|--------|------|
| 种群规模 | N | 50 | 与其他算法一致 |
| 最大迭代 | T | 150 | 与其他算法一致 |
| 发现者比例 | PD | 0.2 | 标准SSA参数 |
| 侦察者比例 | SD | 0.1 | 标准SSA参数 |
| 安全阈值 | ST | 0.8 | 标准SSA参数 |
| 最大惯性权重 | w_max | 0.9 | MSSA参数 |
| 最小惯性权重 | w_min | 0.4 | MSSA参数 |
| 变异概率 | mutation_rate | 0.1 | MSSA参数 |
| 变异幅度 | mutation_scale | 0.5 | MSSA参数 |
| 终止窗口 | termination_window | 20 | MSSA参数 |
| 终止阈值 | termination_threshold | 1e-6 | MSSA参数 |

---

## 实验脚本修改

在实验脚本中添加MSSA对比：

```python
from src.algorithms.mssa import MSSA

# 创建MSSA实例
mssa = MSSA(
    system_model=system_model,
    delay_model=delay_model,
    energy_model=energy_model,
    aoi_model=aoi_model,
    qoe_model=qoe_model,
    fairness_model=fairness_model,
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
best_solution, best_fitness, history = mssa.optimize()
```

---

## 五目标适应度函数

MSSA应继承或复用与RDHO相同的五目标适应度评估函数，确保公平对比：

```python
def evaluate_fitness(self, solution):
    """五目标适应度评估（与RDHO保持一致）"""
    # 计算能耗、时延、AoI
    total_energy = ...
    total_delay = ...
    avg_aoi = ...
    
    # 计算QoE和公平性
    qoe = self.qoe_model.calculate_system_qoe(solution)
    fairness = self.fairness_model.calculate_fairness(solution)
    
    # 归一化
    norm_energy = total_energy / self.energy_max
    norm_delay = total_delay / self.delay_max
    norm_aoi = avg_aoi / self.aoi_max
    
    # 约束违反检测与惩罚
    violations = count_violations()
    penalty = calculate_penalty(violations)
    
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
```

---

## 论文引用格式

在论文中引用MSSA算法：

**BibTeX:**
```bibtex
@article{alseid2023mssamto,
  title={MSSAMTO-IoV: modified sparrow search algorithm for multi-hop task offloading for IoV},
  author={Alseid, Mohammad and El-Moursy, Ali A and Alfawaz, Omar and others},
  journal={The Journal of Supercomputing},
  volume={79},
  pages={20769--20789},
  year={2023},
  publisher={Springer},
  doi={10.1007/s11227-023-05446-2}
}
```

**中文引用:**
> Alseid等人[x]提出了一种改进的麻雀搜索算法(MSSA)，通过引入Logistic混沌映射初始化、自适应惯性权重、均值终止准则和变异策略，显著提升了算法在MEC任务卸载场景中的性能。

---

## 验收标准

1. **功能完整**：四项改进策略全部实现
2. **代码规范**：完整注释、类型注解、异常处理
3. **可运行**：能与RDHO在相同实验设置下对比
4. **可复现**：固定随机种子后结果一致

---

请实现MSSA算法，创建文件 `src/algorithms/mssa.py`，然后将其添加到实验对比中。
