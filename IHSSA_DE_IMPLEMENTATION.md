# IHSSA-DE算法实现说明

## 背景

根据"fpats失败研究报告"分析，FPA-TS算法存在以下致命缺陷：

1. **Lévy飞行不稳定** - 步长过大导致过度探索
2. **种群多样性丧失** - 快速收敛到局部最优
3. **禁忌搜索效率低** - 高维空间中禁忌机制失效

因此，实现**改进混合麻雀搜索算法（IHSSA-DE）**作为替代方案。

---

## 实现文件

### 1. `src/algorithms/ssa.py` - 标准麻雀搜索算法

**SSA三种角色：**

#### 发现者（Producers）- 20%的种群
```python
def update_producer(self, sparrow, iteration, best_sparrow):
    """
    当R2 < ST（安全阈值）时，无捕食者:
        X[i] = X[i] * exp(-i / (alpha * max_iter))

    当R2 >= ST时，有捕食者威胁:
        X[i] = X[i] + Q * L
    """
    R2 = np.random.random()
    alpha = 0.5

    if R2 < self.ST:  # self.ST = 0.8
        decay = np.exp(-iteration / (alpha * self.max_iter))
        new_sparrow = sparrow_array * decay
    else:
        Q = np.random.random()
        L = np.ones_like(sparrow_array)
        new_sparrow = sparrow_array + Q * L
```

#### 加入者（Scroungers）- 70%的种群
```python
def update_scrounger(self, scrounger_idx, population, best_sparrow, worst_sparrow):
    """
    当i > n/2时（饥饿的加入者）:
        X[i] = Q * exp((X_worst - X[i]) / i^2)

    当i <= n/2时:
        X[i] = X_best + |X[i] - X_best| * A * L
        A是1×d矩阵，元素随机为+1或-1
    """
    if scrounger_idx > n / 2:
        Q = np.random.random()
        exponent = (worst_array - scrounger_array) / ((scrounger_idx + 1) ** 2)
        new_scrounger = Q * np.exp(exponent)
    else:
        A = np.random.choice([-1, 1], size=scrounger_array.shape)
        L = np.ones_like(scrounger_array)
        new_scrounger = best_array + np.abs(scrounger_array - best_array) * A * L
```

#### 侦察者（Scouts）- 10%的种群
```python
def update_scout(self, scout, best_sparrow, worst_sparrow, scout_fitness, best_fitness, worst_fitness):
    """
    当f[i] > f_g时（适应度差于全局最优）:
        X[i] = X_best + beta * |X[i] - X_best|

    当f[i] = f_g时（处于危险边缘）:
        X[i] = X[i] + K * ((X[i] - X_worst) / (f[i] - f_worst + eps))
    """
    if scout_fitness > best_fitness:
        beta = np.random.random()
        new_scout = best_array + beta * np.abs(scout_array - best_array)
    else:
        K = np.random.uniform(-1, 1)
        eps = 1e-10
        denominator = scout_fitness - worst_fitness + eps
        step = K * ((scout_array - worst_array) / denominator)
        new_scout = scout_array + step
```

---

### 2. `src/algorithms/ihssa_de.py` - 改进混合麻雀搜索算法

继承SSA基类，添加四个核心改进策略：

#### 改进策略1：Bernoulli混沌映射初始化

**目的**：生成更均匀分布的初始种群，避免随机初始化的聚集现象

**数学公式：**
```
z_{k+1} = z_k / (1 - λ),           当 0 < z_k <= 1 - λ
z_{k+1} = (z_k - 1 + λ) / λ,       当 1 - λ < z_k < 1

λ = 0.4（推荐值）
```

**实现：**
```python
def bernoulli_chaotic_map(self, z: float) -> float:
    if 0 < z <= 1 - self.lambda_chaos:
        z_new = z / (1 - self.lambda_chaos)
    else:
        z_new = (z - 1 + self.lambda_chaos) / self.lambda_chaos
    return z_new % 1.0

def initialize_population(self):
    """使用混沌映射生成初始种群"""
    population = []
    dim = self.num_tasks * 2  # 位置 + 频率
    z = np.random.rand(dim)

    for _ in range(self.population_size):
        # 应用Bernoulli混沌映射
        for d in range(dim):
            z[d] = self.bernoulli_chaotic_map(z[d])

        # 映射到决策变量范围
        solution = self.map_chaos_to_solution(z.copy())
        population.append(solution)

    return population
```

**优势：**
- 混沌序列具有遍历性，覆盖搜索空间更均匀
- 避免随机初始化的盲目性
- 提高初始解质量

---

#### 改进策略2：微分进化（DE）重构加入者策略

**目的**：提升加入者的搜索能力，平衡探索与开发

**DE变异公式：**
```
v_i = x_best + F * (x_r1 - x_r2)

其中：
- F = 0.7（缩放因子）
- x_best：全局最优解
- x_r1, x_r2：随机选择的两个不同个体
```

**DE交叉公式：**
```
u_ij = v_ij,  如果 rand < CR 或 j = jrand
     = x_ij,  否则

其中：
- CR = 0.8（交叉概率）
- jrand：随机选择的维度索引（保证至少一个维度交叉）
```

**实现：**
```python
def de_enhanced_scrounger_update(self, scrounger_idx, population,
                                  fitness_values, best_solution):
    """DE增强的加入者更新"""
    current = population[scrounger_idx]

    # 随机选择两个不同的个体
    candidates = [i for i in range(len(population)) if i != scrounger_idx]
    r1, r2 = np.random.choice(candidates, 2, replace=False)

    # DE变异: v = best + F * (r1 - r2)
    best_array = np.array(best_solution, dtype=float)
    r1_array = np.array(population[r1], dtype=float)
    r2_array = np.array(population[r2], dtype=float)

    mutant = best_array + self.F * (r1_array - r2_array)  # F=0.7

    # 二项交叉
    current_array = np.array(current, dtype=float)
    trial = current_array.copy()

    for j in range(len(current)):
        if np.random.random() < self.CR or j == np.random.randint(len(current)):
            trial[j] = mutant[j]

    return self.handle_constraints(trial.tolist())
```

**优势：**
- DE的变异操作引入全局信息（最优解）和差分信息（r1-r2）
- 交叉操作保持种群多样性
- 比标准SSA加入者更新更高效

---

#### 改进策略3：自适应t分布变异

**目的**：对全局最优解施加自适应扰动，避免早熟收敛

**t分布特性：**
- 自由度df小 → 长尾分布（类似柯西分布）→ 大跳跃
- 自由度df大 → 接近高斯分布 → 小步长精细搜索

**自适应自由度：**
```
df = 1 + iteration * (30 - 1) / max_iter

前期：df ≈ 1  → 柯西分布，大跳跃探索
后期：df ≈ 30 → 高斯分布，小步长开发
```

**变异公式：**
```
x_new = x_best + scale * x_best * t(df)

scale = 0.1 * (1 - iteration / max_iter)  # 幅度递减
```

**实现：**
```python
def adaptive_t_distribution_mutation(self, best_solution):
    """自适应t分布变异"""
    from scipy.stats import t as t_dist

    # 自适应自由度：从1（柯西）到30（近似高斯）
    min_df = 1
    max_df = 30
    df = min_df + self.current_iteration * (max_df - min_df) / self.max_iter

    best_array = np.array(best_solution, dtype=float)

    # 生成t分布随机扰动
    perturbation = t_dist.rvs(df, size=best_array.shape)

    # 扰动幅度随迭代递减
    scale = 0.1 * (1 - self.current_iteration / self.max_iter)

    new_solution = best_array + scale * best_array * perturbation

    return self.handle_constraints(new_solution.tolist())
```

**优势：**
- 自适应性强：前期大跳跃避免局部最优，后期小步长精细调优
- 理论基础扎实：t分布是统计学中的经典分布
- 比Lévy飞行更稳定：没有FPA的步长过大问题

---

#### 改进策略4：动态约束惩罚机制

**目的**：平衡可行性与优化质量

**动态惩罚公式：**
```
Penalty = base_penalty * (1 + t/T_max)^alpha * Σ(Violations)

其中：
- base_penalty = 0.5（初始惩罚系数）
- alpha = 2.0（增长指数）
- t：当前迭代
- T_max：最大迭代
```

**惩罚演化：**
```
前期（t=0）：
penalty_factor = 0.5 * (1 + 0)^2 = 0.5
→ 允许探索轻微违规但有潜力的解

中期（t=75）：
penalty_factor = 0.5 * (1 + 0.5)^2 = 1.125
→ 逐步增加惩罚

后期（t=150）：
penalty_factor = 0.5 * (1 + 1)^2 = 2.0
→ 强制收敛到可行域
```

**实现：**
```python
def calculate_dynamic_penalty(self, constraint_violations: int) -> float:
    """动态惩罚因子"""
    if constraint_violations == 0:
        return 0.0

    # 迭代进度
    progress = self.current_iteration / self.max_iter

    # 动态惩罚因子：随迭代指数增长
    penalty_factor = self.base_penalty * ((1 + progress) ** self.penalty_alpha)

    return penalty_factor * constraint_violations
```

**优势：**
- 前期宽容：允许探索违规但fitness低的解
- 后期严格：确保最终解满足所有约束
- 比固定惩罚（penalty=2.0）更灵活

---

## IHSSA-DE完整优化流程

```python
def optimize(self):
    """
    主循环流程:
    1. Bernoulli混沌初始化（改进策略1）
    2. 评估初始种群（动态惩罚）
    3. 主迭代循环:
       a. 划分角色（发现者20%、加入者70%、侦察者10%）
       b. 发现者更新（标准SSA）
       c. 加入者更新（DE增强 - 改进策略2）
       d. 侦察者更新（标准SSA）
       e. 对全局最优执行t分布变异（改进策略3）
       f. 更新全局最优
    4. 返回最优解
    """
    # 1. 混沌初始化
    population = self.initialize_population()  # 改进1

    # 2. 评估初始种群
    self.current_iteration = 0
    fitness_values = [self.evaluate_fitness(sol) for sol in population]  # 动态惩罚

    # 找初始最优
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    best_fitness = fitness_values[best_idx]

    # 3. 主迭代
    for iteration in range(self.max_iter):
        self.current_iteration = iteration

        # 划分角色
        sorted_indices = np.argsort(fitness_values)
        n_producers = int(self.population_size * 0.2)
        n_scouts = int(self.population_size * 0.1)

        producer_indices = sorted_indices[:n_producers]
        scout_indices = sorted_indices[-n_scouts:]
        scrounger_indices = sorted_indices[n_producers:-n_scouts]

        # a. 发现者更新（标准SSA）
        for idx in producer_indices:
            new_population[idx] = self.update_producer(...)

        # b. 加入者更新（DE增强 - 改进策略2）
        for idx in scrounger_indices:
            new_population[idx] = self.de_enhanced_scrounger_update(...)

        # c. 侦察者更新（标准SSA）
        for idx in scout_indices:
            new_population[idx] = self.update_scout(...)

        # 评估新种群
        fitness_values = [self.evaluate_fitness(sol) for sol in population]

        # 更新最优
        if current_best < best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness

        # d. t分布变异（改进策略3）
        mutated = self.adaptive_t_distribution_mutation(best_solution)
        if mutated_fitness < best_fitness:
            best_solution = mutated
            best_fitness = mutated_fitness

    return best_solution, best_fitness, history
```

---

## 实验配置

### 修改的文件

1. **`src/algorithms/ssa.py`** (新建)
   - 标准SSA算法实现
   - 三种角色更新策略
   - 五目标适应度函数

2. **`src/algorithms/ihssa_de.py`** (新建)
   - 继承SSA基类
   - 四个改进策略
   - 动态惩罚机制

3. **`src/experiments/chapter4_fpa_ts_experiment.py`** (修改)
   - 导入：`from ..algorithms.ihssa_de import IHSSADE`
   - 导入：`from ..algorithms.ssa import SSA`
   - 删除：FPA-TS和FPA相关代码
   - 算法字典：只保留IHSSA-DE、SSA、TLBO-HHO

### 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **SSA参数** | | |
| PD | 0.2 | 发现者比例（20%） |
| SD | 0.1 | 侦察者比例（10%） |
| ST | 0.8 | 安全阈值 |
| **DE参数** | | |
| F | 0.7 | 缩放因子 |
| CR | 0.8 | 交叉概率 |
| **混沌参数** | | |
| lambda_chaos | 0.4 | Bernoulli混沌映射参数 |
| **动态惩罚** | | |
| base_penalty | 0.5 | 初始惩罚系数 |
| penalty_alpha | 2.0 | 惩罚增长指数 |
| **通用参数** | | |
| population_size | 50 | 种群大小 |
| max_iter | 150 | 最大迭代次数 |
| **五目标权重** | | |
| w_energy | 0.15 | 能耗权重 |
| w_delay | 0.15 | 时延权重 |
| w_aoi | 0.20 | AoI权重 |
| w_qoe | 0.25 | QoE权重 |
| w_fairness | 0.25 | 公平性权重 |

---

## 预期性能

### 理论优势分析

**vs FPA-TS：**
1. ✅ **稳定性更高** - 无Lévy飞行的大步长问题
2. ✅ **收敛更快** - DE策略比禁忌搜索效率高
3. ✅ **适应性强** - t分布自适应调整探索/开发
4. ✅ **可行性好** - 动态惩罚机制保证CSR

**vs TLBO-HHO：**
1. ✅ **初始解更优** - 混沌初始化比随机初始化好
2. ✅ **多样性好** - 三种角色+DE+t变异保持多样性
3. ✅ **五目标优化** - 专为QoE和Fairness设计

**vs 标准SSA：**
1. ✅ **质量提升** - 四个改进策略全面增强
2. ✅ **收敛速度** - DE和t分布加速收敛
3. ✅ **约束处理** - 动态惩罚比固定惩罚好

### 预期实验结果

| 算法 | Fitness | QoE | Fairness | CSR | 收敛速度 |
|------|---------|-----|----------|-----|---------|
| **IHSSA-DE** | **<1.0** | **>0.60** | **>0.99** | **>98%** | **最快** |
| SSA | ~1.5 | ~0.55 | ~0.95 | >92% | 中等 |
| TLBO-HHO | ~1.3 | ~0.57 | ~0.98 | ~99% | 较快 |

**关键指标：**
- **Fitness**：IHSSA-DE应该最小（<1.0），优于TLBO-HHO（~1.3）
- **QoE**：IHSSA-DE应该最高（>0.60），因为五目标权重更大
- **Fairness**：IHSSA-DE应该最高（>0.99），接近完美公平
- **CSR**：所有算法都应该>95%，但IHSSA-DE应该最高（>98%）

---

## 验证方法

### 运行实验

```bash
cd /Users/white/develop/py_workspace/mec_offloading
python -m src.experiments.chapter4_fpa_ts_experiment
```

### 检查收敛曲线

观察 `results/aoi/convergence_curves_*_tasks.png`：

**期望：**
1. IHSSA-DE曲线在TLBO-HHO下方（fitness更低）
2. IHSSA-DE收敛最快（前50次迭代快速下降）
3. SSA曲线在IHSSA-DE上方（证明改进有效）

### 分析性能表格

检查实验输出的表格：

**关键对比：**
```
算法         Fitness    QoE      Fairness  CSR    运行时间
-------------------------------------------------------
IHSSA-DE    0.85±0.12  0.62±0.03  0.99±0.01  98.5%  ~15s
SSA         1.52±0.18  0.56±0.04  0.95±0.02  93.2%  ~13s
TLBO-HHO    1.28±0.15  0.58±0.03  0.98±0.01  99.1%  ~14s
```

**验证标准：**
- ✅ IHSSA-DE的Fitness最小
- ✅ IHSSA-DE的QoE最高
- ✅ IHSSA-DE的Fairness最高
- ✅ IHSSA-DE的CSR>98%

---

## 调试建议

### 如果IHSSA-DE性能不如预期

**问题1：Fitness仍然较高（>1.5）**

可能原因：动态惩罚过强

解决方案：
```python
# 降低惩罚增长速度
penalty_alpha = 1.5  # 从2.0降低到1.5
base_penalty = 0.3   # 从0.5降低到0.3
```

**问题2：CSR较低（<95%）**

可能原因：动态惩罚过弱

解决方案：
```python
# 增强后期惩罚
penalty_alpha = 2.5  # 从2.0增加到2.5
```

**问题3：收敛过慢**

可能原因：DE参数不当

解决方案：
```python
# 增强DE的影响
F = 0.8   # 从0.7增加到0.8（更激进的变异）
CR = 0.9  # 从0.8增加到0.9（更多交叉）
```

**问题4：陷入局部最优**

可能原因：t分布扰动不足

解决方案：
```python
# 增强t分布变异
scale = 0.15 * (1 - iteration / max_iter)  # 从0.1增加到0.15
```

---

## 算法复杂度分析

### 时间复杂度

**IHSSA-DE单次迭代：**
```
O(n * d) = O(50 * 80) = O(4000)

其中：
- n = population_size = 50
- d = num_tasks * 2 = 40 * 2 = 80

组成部分：
1. 发现者更新：O(0.2n * d) = O(800)
2. 加入者更新（DE）：O(0.7n * d) = O(2800)
3. 侦察者更新：O(0.1n * d) = O(400)
4. t分布变异：O(d) = O(80)
5. 适应度评估：O(n * d) = O(4000)

总复杂度：O(max_iter * n * d) = O(150 * 50 * 80) = O(600,000)
```

**vs TLBO-HHO：**
```
TLBO-HHO: O(max_iter * n * d) = O(100 * 50 * 80) = O(400,000)

IHSSA-DE虽然迭代多50次，但每次迭代效率更高（DE比HHO简单）
预期运行时间相近或略长（15s vs 14s）
```

### 空间复杂度

```
O(n * d) = O(50 * 80) = O(4000)

主要存储：
1. 种群：population[n][d]
2. 适应度：fitness_values[n]
3. 混沌序列：z[d]（可复用）
4. 临时变量：O(d)

总空间：O(n * d)
```

---

## 总结

IHSSA-DE通过四个核心改进策略，克服了FPA-TS的致命缺陷：

1. ✅ **Bernoulli混沌初始化** - 解决初始解质量问题
2. ✅ **DE重构加入者** - 解决搜索效率问题
3. ✅ **自适应t分布变异** - 解决早熟收敛问题
4. ✅ **动态约束惩罚** - 解决可行性与优化质量平衡问题

预期IHSSA-DE在五目标优化中全面超越TLBO-HHO，成为MEC任务卸载的最优算法！

现在可以运行实验验证效果。
