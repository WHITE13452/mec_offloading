# 适应度函数调试输出说明

## 添加的调试功能

为了验证FPA-TS和TLBO-HHO的适应度函数计算一致性，在两个算法的`evaluate_fitness`方法中添加了调试输出。

## 修改文件

### 1. `src/algorithms/fpa.py`

#### 新增实例变量（__init__方法）
```python
# Debug计数器
self._eval_count = 0
self._debug_first_eval = True
```

#### 调试输出（evaluate_fitness方法，第357-364行）
```python
# Debug输出（仅打印第一次评估）
if self._debug_first_eval:
    avg_aoi = total_aoi / valid_aoi_tasks if valid_aoi_tasks > 0 else 0.0
    print(f"[FPA DEBUG] Energy: {total_energy:.2f}, Delay: {total_delay:.4f}, "
          f"AoI: {avg_aoi:.4f}, QoE: {qoe:.4f}, Fairness: {fairness:.4f}, "
          f"Violations: {constraint_violations}, Penalty: {penalty:.4f}, "
          f"Fitness: {fitness:.6f}")
    self._debug_first_eval = False
```

**注意：** FPA-TS继承自FPA，会自动使用这个调试输出。

### 2. `src/algorithms/tlbo_hho.py`

#### 新增实例变量（__init__方法）
```python
# Debug计数器
self._eval_count = 0
self._debug_first_eval = True
```

#### 调试输出（evaluate_fitness方法，第198-205行）
```python
# Debug输出（仅打印第一次评估）
if self._debug_first_eval:
    avg_aoi = total_aoi / valid_aoi_tasks if valid_aoi_tasks > 0 else 0.0
    print(f"[TLBO-HHO DEBUG] Energy: {total_energy:.2f}, Delay: {total_delay:.4f}, "
          f"AoI: {avg_aoi:.4f}, QoE: {qoe:.4f}, Fairness: {fairness:.4f}, "
          f"Violations: {constraint_violations}, Penalty: {penalty:.4f}, "
          f"Fitness: {fitness:.6f}")
    self._debug_first_eval = False
```

## 调试输出格式

每个算法在第一次评估解时会打印以下信息：

```
[算法名 DEBUG] Energy: XX.XX, Delay: XX.XXXX, AoI: XX.XXXX, QoE: XX.XXXX, Fairness: XX.XXXX, Violations: X, Penalty: XX.XXXX, Fitness: XX.XXXXXX
```

### 字段说明

| 字段 | 含义 | 单位/范围 |
|------|------|-----------|
| Energy | 总能耗 | 焦耳(J) |
| Delay | 总时延 | 秒(s) |
| AoI | 平均信息年龄 | 秒(s) |
| QoE | 系统QoE | [0, 1] |
| Fairness | Jain公平性指数 | [0, 1] |
| Violations | 约束违反次数 | 整数 |
| Penalty | 惩罚项值 | = Violations × 2.0 |
| Fitness | 最终适应度值 | 浮点数（越小越好） |

## 使用方法

运行实验时，会自动输出调试信息：

```bash
cd /Users/white/develop/py_workspace/mec_offloading
python -m src.experiments.chapter4_fpa_ts_experiment
```

### 预期输出示例

```
============================================================
运行算法: FPA-TS
============================================================
  运行 1/5... [FPA DEBUG] Energy: 52.34, Delay: 148.2345, AoI: 1.8765, QoE: 0.7543, Fairness: 0.8234, Violations: 3, Penalty: 6.0000, Fitness: 6.545321
  完成! (用时: 12.34s)
  ...

============================================================
运行算法: FPA
============================================================
  运行 1/5... [FPA DEBUG] Energy: 51.87, Delay: 149.1234, AoI: 1.9123, QoE: 0.7612, Fairness: 0.8156, Violations: 2, Penalty: 4.0000, Fitness: 4.532109
  完成! (用时: 11.89s)
  ...

============================================================
运行算法: TLBO-HHO
============================================================
  运行 1/5... [TLBO-HHO DEBUG] Energy: 53.21, Delay: 147.8901, AoI: 1.8432, QoE: 0.7689, Fairness: 0.8312, Violations: 3, Penalty: 6.0000, Fitness: 6.234567
  完成! (用时: 13.21s)
  ...
```

## 验证要点

### 1. 数值范围检查

确认所有算法的各项指标在合理范围内：

- **Energy**: 通常在10-100 J范围（取决于任务数量）
- **Delay**: 通常在100-200 s范围
- **AoI**: 通常在1-5 s范围
- **QoE**: 必须在[0, 1]范围
- **Fairness**: 必须在[0, 1]范围
- **Violations**: 整数，通常0-10
- **Penalty**: = Violations × 2.0
- **Fitness**: 通常在0.5-10范围（五目标优化）

### 2. 一致性验证

检查三个算法的fitness计算逻辑：

```python
# 所有算法应该使用相同的公式
fitness = 0.15 * (Energy/Energy_max) +
          0.15 * (Delay/Delay_max) +
          0.20 * (AoI/AoI_max) +
          0.25 * (1 - QoE) +
          0.25 * (1 - Fairness) +
          Penalty
```

### 3. 数量级对比

关键检查点：
- ✅ 三个算法的fitness值应该在同一数量级（0.5-10）
- ✅ 不应该出现fitness值相差1000倍的情况
- ✅ 惩罚项不应该主导整个fitness值

### 4. 异常情况

如果出现以下情况，说明有问题：

| 异常现象 | 可能原因 |
|---------|---------|
| Fitness > 100 | 惩罚系数过大或归一化失败 |
| Fitness < 0 | QoE/Fairness处理错误（不应该直接减） |
| QoE > 1 或 < 0 | QoE模型计算错误 |
| Fairness > 1 或 < 0 | 公平性模型计算错误 |
| Penalty ≠ Violations × 2.0 | 惩罚系数不一致 |

## 调试输出的设计考虑

### 为什么只打印第一次评估？

1. **性能考虑**：每次迭代会评估多个解，打印所有评估会严重拖慢实验
2. **可读性**：只需验证计算逻辑是否正确，第一次评估足够
3. **对比方便**：三个算法的第一次评估都会打印，便于直接对比

### 如何临时启用更多调试输出？

如果需要查看更多评估过程，可以修改条件：

```python
# 打印前10次评估
if self._eval_count < 10:
    print(...)
    self._eval_count += 1

# 或者打印每次评估
# if True:  # 移除条件
```

## 清理调试输出

实验验证完成后，如果不再需要调试信息，可以：

1. **选项1：注释掉调试代码**
   ```python
   # if self._debug_first_eval:
   #     avg_aoi = ...
   #     print(...)
   #     self._debug_first_eval = False
   ```

2. **选项2：删除调试代码和变量**
   - 删除`__init__`中的`self._debug_first_eval = True`
   - 删除`evaluate_fitness`中的整个if块

3. **选项3：保留代码**
   - 调试代码不影响性能（只打印一次）
   - 未来调试时可能还需要
   - 建议保留

## 总结

通过添加调试输出，我们可以：

1. ✅ **验证计算一致性** - 确认三个算法使用相同的适应度函数
2. ✅ **快速定位问题** - 如果fitness值异常，立即看到是哪个指标出错
3. ✅ **监控数值范围** - 确保所有指标在合理范围内
4. ✅ **性能影响最小** - 只打印一次，不影响实验速度

现在可以运行实验，通过调试输出验证修复是否成功！
