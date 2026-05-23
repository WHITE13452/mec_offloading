# MEC任务卸载优化项目

## 项目背景

这是一个移动边缘计算(MEC)任务卸载优化的硕士论文研究项目。论文包含两部分核心工作：

* **第三章** ：TLBO-HHO算法 - 三目标优化（能耗、时延、AoI）✅ **已完成**
* **第四章** ：FPA-TS算法 - 五目标优化（能耗、时延、AoI、QoE、公平性）⏳ **需要实现**

你的任务是在现有第三章代码基础上，开发第四章FPA-TS算法及其实验代码。

## 代码结构

项目根目录：`~/develop/py_workspace/mec_offloading/`

```
mec_offloading/
├── src/
│   ├── __init__.py
│   ├── algorithms/                    # 算法模块
│   │   ├── __init__.py
│   │   ├── base_algorithm.py          # 算法基类
│   │   ├── tlbo.py                    # TLBO算法
│   │   ├── tlbo_hho.py                # TLBO-HHO混合算法 ✅
│   │   ├── tlbo_plus.py               # TLBO+算法
│   │   ├── ga.py                      # 遗传算法
│   │   ├── gwo.py                     # 灰狼优化算法
│   │   └── mo_tlbo.py                 # 多目标TLBO
│   ├── models/                        # 模型模块
│   │   ├── __init__.py
│   │   ├── system_model.py            # 系统模型（Task, Device, EdgeServer, CloudServer）
│   │   ├── delay_model.py             # 时延模型
│   │   ├── energy_model.py            # 能耗模型
│   │   └── aoi_model.py               # 信息年龄(AoI)模型
│   └── experiments/                   # 实验模块
│       ├── __init__.py
│       ├── realistic_system_setup.py  # 系统配置
│       ├── complete_aoi_experiment.py # 第三章实验 ✅
│       ├── complete_aoi_exp_balance.py # 平衡权重实验 ✅
│       └── part1_experiments.py
└── results/                           # 实验结果
```

## 代码规范

* **继承现有架构** ：所有新算法必须继承 `BaseAlgorithm` 基类
* **保持接口一致** ：`optimize()` 方法返回 `(best_solution, best_fitness, history)`
* **文档规范** ：所有类和方法必须有完整的docstring
* **类型注解** ：使用typing模块进行类型注解
* **可复现性** ：实验结果可复现，支持设置随机种子

## 关键数学公式

### QoE模型

```
时延满足度: S_delay = 1 / (1 + exp(β * (T - T_max)))

能耗接受度: S_energy = 1 - E/E_budget  (电池充足时)

任务完成率: S_completion = 1 if (T ≤ T_max and AoI ≤ AoI_max) else 0

综合QoE: QoE = w1*S_delay + w2*S_energy + w3*S_completion
```

### 公平性模型

```
Jain公平性指数: Fairness = (Σ QoE_i)² / (K * Σ QoE_i²)
```

### 五目标优化函数

```
min F(X) = w_E*E/E_norm + w_T*T/T_norm + w_AoI*AoI/AoI_norm 
         - w_QoE*QoE - w_F*Fairness + Penalty

权重设置: w_E=0.15, w_T=0.15, w_AoI=0.20, w_QoE=0.25, w_F=0.25
```
