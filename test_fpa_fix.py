"""
测试FPA和FPA-TS的适应度函数修复
验证fitness值是否在合理范围内（0.001-1）
"""
import numpy as np
from src.models.system_model import SystemModel
from src.models.delay_model import DelayModel
from src.models.energy_model import EnergyModel
from src.models.aoi_model import AoIModel
from src.models.qoe_model import QoEModel
from src.models.fairness_model import FairnessModel
from src.algorithms.fpa import FPA
from src.algorithms.fpa_ts import FPATS
from src.algorithms.tlbo_hho import TLBOHHO
from src.experiments.realistic_system_setup import create_realistic_edge_computing_system


def test_fitness_range():
    """测试适应度函数值的范围"""
    print("="*80)
    print("测试FPA/FPA-TS适应度函数修复")
    print("="*80)

    # 创建小规模系统进行快速测试
    system = create_realistic_edge_computing_system(
        num_devices=10,
        num_edge_servers=2,
        num_cloud_servers=1,
        num_tasks=20
    )

    # 为任务添加五目标参数
    for task in system.tasks:
        task.update_interval = np.random.uniform(0.1, 1.0)
        task.max_aoi = np.random.uniform(1.0, 3.0)
        task.delay_sensitivity = np.random.uniform(0.5, 1.0)
        task.energy_budget = task.data_size * task.computation_complexity * 1e-27 * 2

    # 创建模型
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    aoi_model = AoIModel(system, delay_model)
    qoe_model = QoEModel(system, delay_model, energy_model, aoi_model)
    fairness_model = FairnessModel(system, qoe_model)

    # 五目标权重
    weights = {
        'w_energy': 0.15,
        'w_delay': 0.15,
        'w_aoi': 0.20,
        'w_qoe': 0.25,
        'w_fairness': 0.25
    }

    # 创建算法实例
    algorithms = {
        'TLBO-HHO': TLBOHHO(
            system, delay_model, energy_model, aoi_model,
            max_iter=5, population_size=10,
            w_energy=0.33, w_delay=0.33, w_aoi=0.34,
            verbose=False
        ),
        'FPA': FPA(
            system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
            max_iter=5, population_size=10,
            **weights,
            verbose=False
        ),
        'FPA-TS': FPATS(
            system, delay_model, energy_model, aoi_model, qoe_model, fairness_model,
            max_iter=5, population_size=10,
            **weights,
            tabu_tenure=5, tabu_trigger_freq=2,
            verbose=False
        ),
    }

    print("\n测试各算法的适应度值范围：")
    print("-"*80)

    for name, algorithm in algorithms.items():
        print(f"\n{name}:")

        # 初始化种群
        population = algorithm.initialize_population()

        # 评估前10个解
        fitness_values = []
        for i, solution in enumerate(population[:10]):
            fitness = algorithm.evaluate_fitness(solution)
            fitness_values.append(fitness)
            if i < 3:  # 只打印前3个
                print(f"  解 {i+1}: fitness = {fitness:.6f}")

        # 统计
        valid_fitness = [f for f in fitness_values if np.isfinite(f)]
        if valid_fitness:
            print(f"  有效解数量: {len(valid_fitness)}/10")
            print(f"  Fitness范围: [{min(valid_fitness):.6f}, {max(valid_fitness):.6f}]")
            print(f"  平均Fitness: {np.mean(valid_fitness):.6f}")

            # 检查是否在合理范围内
            if min(valid_fitness) < 0.0001 or max(valid_fitness) > 100:
                print(f"  ⚠️  警告: Fitness值超出合理范围 (期望: 0.001-10)")
            else:
                print(f"  ✓ Fitness值在合理范围内")
        else:
            print(f"  ❌ 错误: 没有有效解")

    # 运行一轮完整优化
    print("\n" + "="*80)
    print("运行一轮完整优化测试（5次迭代）：")
    print("="*80)

    for name, algorithm in algorithms.items():
        print(f"\n{name}:")

        # 重置任务状态
        for task in system.tasks:
            task.execution_location = None
            task.execution_node_id = None
            task.allocated_resource = None

        # 运行优化
        best_solution, best_fitness, history = algorithm.optimize()

        if best_solution is not None:
            print(f"  初始Fitness: {history[0]:.6f}")
            print(f"  最终Fitness: {best_fitness:.6f}")
            print(f"  改进率: {(history[0] - best_fitness) / history[0] * 100:.2f}%")

            # 检查收敛历史
            if len(history) > 1:
                improvements = sum(1 for i in range(1, len(history)) if history[i] < history[i-1])
                print(f"  改进次数: {improvements}/{len(history)-1}")
        else:
            print(f"  ❌ 优化失败")

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == '__main__':
    test_fitness_range()
