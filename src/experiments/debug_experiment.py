# src/experiments/debug_experiment.py
import numpy as np
from src.experiments.realistic_system_setup import create_realistic_edge_computing_system
from src.models.delay_model import DelayModel
from src.models.energy_model import EnergyModel
from src.algorithms.tlbo import TLBO
from src.algorithms.ga import GA
from src.algorithms.gwo import GWO

def debug_single_evaluation(algorithm, system):
    """调试单次适应度评估"""
    print(f"\n调试 {algorithm.__class__.__name__} 算法...")
    
    # 初始化种群
    population = algorithm.initialize_population()
    print(f"种群大小: {len(population)}")
    print(f"解的维度: {len(population[0])} x {len(population[0][0])}")
    
    # 测试前几个解
    for i, solution in enumerate(population[:3]):
        print(f"\n测试解 {i+1}:")
        try:
            fitness = algorithm.evaluate_fitness(solution)
            print(f"  适应度: {fitness}")
            
            # 检查解的应用
            system.apply_solution(solution)
            for j, task in enumerate(system.tasks[:3]):
                print(f"  任务 {j}: 位置={task.execution_location}, 资源={task.allocated_resource:.2e}")
                
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            break

def main():
    # 创建系统
    system = create_realistic_edge_computing_system(
        num_devices=10, num_edge_servers=3, num_cloud_servers=2, num_tasks=20
    )
    
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    
    # 测试各个算法
    algorithms = [
        GWO(system, delay_model, energy_model, max_iter=5, population_size=10, verbose=True),
        TLBO(system, delay_model, energy_model, max_iter=5, population_size=10, verbose=True),
        GA(system, delay_model, energy_model, max_iter=5, population_size=10, verbose=True)
    ]
    
    for algorithm in algorithms:
        debug_single_evaluation(algorithm, system)

if __name__ == "__main__":
    main()

