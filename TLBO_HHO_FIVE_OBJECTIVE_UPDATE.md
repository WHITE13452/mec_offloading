# TLBO-HHO五目标优化升级说明

## 更新概述

将TLBO-HHO算法从三目标优化（能耗、时延、AoI）升级为五目标优化（能耗、时延、AoI、QoE、公平性），使其与FPA-TS和FPA使用完全一致的适应度函数。

## 修改文件

### 1. `src/algorithms/tlbo_hho.py`

#### 新增导入
```python
from ..models.qoe_model import QoEModel
from ..models.fairness_model import FairnessModel
```

#### 更新__init__方法

**新增参数：**
- `qoe_model: Optional[QoEModel] = None` - QoE模型
- `fairness_model: Optional[FairnessModel] = None` - 公平性模型
- `w_qoe: float = 0.25` - QoE权重（默认0.25）
- `w_fairness: float = 0.25` - 公平性权重（默认0.25）

**权重默认值调整：**
- `w_energy: 0.4 → 0.15`
- `w_delay: 0.6 → 0.15`
- `w_aoi: 0.0 → 0.20`

**新增实例变量：**
```python
self.qoe_model = qoe_model
self.fairness_model = fairness_model
self.w_qoe = w_qoe
self.w_fairness = w_fairness
```

#### 新增evaluate_fitness方法

完全重写适应度函数，与FPA的实现保持一致：

```python
def evaluate_fitness(self, solution):
    """
    五目标适应度函数（与FPA保持一致）

    F(X) = w_E*E/E_norm + w_T*T/T_norm + w_AoI*AoI/AoI_norm
         + w_QoE*(1-QoE) + w_F*(1-Fairness) + Penalty
    """
```

**关键特性：**
1. **动态归一化因子**
   ```python
   self.energy_max = max(self.energy_max, total_energy, 1.0)
   self.delay_max = max(self.delay_max, total_delay, 1.0)
   if valid_aoi_tasks > 0:
       self.aoi_max = max(self.aoi_max, total_aoi / valid_aoi_tasks, 1.0)
   ```

2. **约束惩罚**
   ```python
   penalty = constraint_violations * 2.0  # 与FPA一致
   ```

3. **QoE和Fairness处理**
   ```python
   # 最大化目标转换为最小化
   self.w_qoe * (1.0 - qoe)
   self.w_fairness * (1.0 - fairness)
   ```

### 2. `src/experiments/chapter4_fpa_ts_experiment.py`

#### 更新TLBO-HHO实例化

**修改前（三目标）：**
```python
'TLBO-HHO': TLBOHHO(
    system, delay_model, energy_model, aoi_model,
    max_iter=max_iter, population_size=population_size,
    w_energy=weights['w_energy'] + weights['w_qoe']/3,  # 权重混合
    w_delay=weights['w_delay'] + weights['w_qoe']/3,
    w_aoi=weights['w_aoi'] + weights['w_qoe']/3,
    verbose=False
),
```

**修改后（五目标）：**
```python
'TLBO-HHO': TLBOHHO(
    system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
    max_iter=max_iter, population_size=population_size,
    **weights,  # 直接使用五目标权重
    verbose=False
),
```

#### 移除GA和GWO算法

删除了以下算法实例：
- `'GA': GA(...)`
- `'GWO': GWO(...)`

**原因：**
1. GA和GWO仅支持三目标，无法公平比较
2. 简化实验，专注于FPA-TS、FPA和TLBO-HHO的对比
3. 三个算法都使用相同的五目标适应度函数

## 验证标准

修复完成后，运行实验应该看到：

### 1. Fitness值范围一致
```bash
python -m src.experiments.chapter4_fpa_ts_experiment
```

期望结果：
- FPA-TS fitness: 0.001-0.1
- FPA fitness: 0.001-0.1
- TLBO-HHO fitness: 0.001-0.1

**所有算法在同一数量级！**

### 2. 收敛曲线可比较
- 三条曲线应该在同一Y轴范围内清晰显示
- 可以直观看出收敛速度差异

### 3. 性能指标对比
期望：
- **FPA-TS** 在QoE和Fairness上优于其他算法（因为有禁忌搜索增强）
- **TLBO-HHO** 可能在能耗和时延上表现更好（因为HHO的局部搜索能力）
- **FPA** 作为基准算法

## 适应度函数一致性检查

### FPA (src/algorithms/fpa.py:346-351)
```python
fitness = (self.w_energy * normalized_energy +
          self.w_delay * normalized_delay +
          self.w_aoi * normalized_aoi +
          self.w_qoe * (1.0 - qoe) +
          self.w_fairness * (1.0 - fairness) +
          penalty)
```

### TLBO-HHO (src/algorithms/tlbo_hho.py:187-192)
```python
fitness = (self.w_energy * normalized_energy +
          self.w_delay * normalized_delay +
          self.w_aoi * normalized_aoi +
          self.w_qoe * (1.0 - qoe) +
          self.w_fairness * (1.0 - fairness) +
          penalty)
```

✅ **完全一致！**

## 理论预期

### 权重分配（所有算法统一）
- 能耗权重：0.15 (15%)
- 时延权重：0.15 (15%)
- AoI权重：0.20 (20%)
- QoE权重：0.25 (25%)
- 公平性权重：0.25 (25%)

### Fitness计算示例

假设某个解：
- normalized_energy = 1.0
- normalized_delay = 1.0
- normalized_aoi = 1.0
- QoE = 0.85
- Fairness = 0.90
- constraint_violations = 0

计算过程：
```
fitness = 0.15*1.0 + 0.15*1.0 + 0.20*1.0
        + 0.25*(1-0.85) + 0.25*(1-0.90) + 0*2.0
        = 0.15 + 0.15 + 0.20 + 0.0375 + 0.025 + 0
        = 0.5625
```

### 优化方向

所有目标都是 **最小化**：
1. 能耗、时延、AoI：直接最小化 ✓
2. QoE、Fairness：通过(1-value)转换为最小化 ✓
   - QoE越高 → (1-QoE)越小 → fitness越小 ✓
   - Fairness越高 → (1-Fairness)越小 → fitness越小 ✓

## 实验对比重点

### 1. 算法优化能力
- **收敛速度**：哪个算法收敛最快？
- **最终质量**：哪个算法找到的解最好（fitness最小）？

### 2. 五目标平衡
- **能耗 vs QoE**：FPA-TS能否在保证QoE的同时降低能耗？
- **时延 vs 公平性**：TLBO-HHO的局部搜索是否有助于平衡这两个目标？

### 3. 约束满足
- **约束违反率**：哪个算法产生的解约束违反最少？
- **稳定性**：多次运行的标准差如何？

## 总结

通过这次升级：

1. ✅ **统一框架** - 三个算法使用相同的五目标适应度函数
2. ✅ **公平比较** - fitness值在同一数量级，可以直接对比
3. ✅ **代码一致性** - TLBO-HHO与FPA采用相同的归一化和惩罚策略
4. ✅ **实验简化** - 移除不相关的算法，专注于核心对比

现在可以进行有意义的五目标优化实验对比了！
