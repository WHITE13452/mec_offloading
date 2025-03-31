# main.py
import argparse
import sys
import os

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='Mobile Edge Computing Task Offloading Optimization')
    parser.add_argument('--part', type=int, choices=[1, 2], default=1,
                        help='Which part of the research to run: 1 (without AoI) or 2 (with AoI)')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations for optimization algorithms')
    parser.add_argument('--pop_size', type=int, default=50,
                        help='Population size for optimization algorithms')
    parser.add_argument('--devices', type=int, default=5,
                        help='Number of devices in the system')
    parser.add_argument('--edge_servers', type=int, default=3,
                        help='Number of edge servers in the system')
    parser.add_argument('--cloud_servers', type=int, default=1,
                        help='Number of cloud servers in the system')
    parser.add_argument('--tasks', type=int, default=10,
                        help='Number of tasks in the system')
    parser.add_argument('--output_dir', type=str, default='data/results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 添加src目录到路径
    sys.path.append(os.path.abspath('src'))
    
    if args.part == 1:
        print("Running Part 1: Task Offloading and Resource Allocation without AoI")
        from src.experiments.part1_experiments import create_test_system, run_algorithm_comparison, plot_convergence_curves, analyze_task_allocation
        
        # 创建测试系统
        system = create_test_system(
            num_devices=args.devices,
            num_edge_servers=args.edge_servers,
            num_cloud_servers=args.cloud_servers,
            num_tasks=args.tasks
        )
        
        # 运行算法比较
        results = run_algorithm_comparison(
            system=system,
            max_iter=args.max_iter,
            population_size=args.pop_size
        )
        
        # 绘制收敛曲线
        plot_convergence_curves(results, save_path=os.path.join(args.output_dir, 'part1_convergence_curves.png'))
        
        # 分析最优解
        for name, data in results.items():
            best_run_idx = np.argmin(data['best_fitness_values'])
            best_solution = data['best_solutions'][best_run_idx]
            best_fitness = data['best_fitness_values'][best_run_idx]
            
            print(f"\nBest solution found by {name}:")
            print(f"Fitness value: {best_fitness:.6f}")
            
            allocation = analyze_task_allocation(system, best_solution)
            print(f"Task allocation: Device={allocation['device']}, Edge={allocation['edge']}, Cloud={allocation['cloud']}")
    
    else:  # args.part == 2
        print("Running Part 2: Task Offloading and Resource Allocation with AoI")
        from src.experiments.part2_experiments import (
            create_test_system_with_aoi, run_weighted_sum_optimization, run_multi_objective_optimization,
            plot_pareto_front, compare_weighted_sum_results, analyze_update_interval_distribution,
            analyze_task_allocation_distribution, analyze_aoi_impact
        )
        from src.models.delay_model import DelayModel
        from src.models.energy_model import EnergyModel
        from src.models.aoi_model import AoIModel
        
        # 创建测试系统
        system = create_test_system_with_aoi(
            num_devices=args.devices,
            num_edge_servers=args.edge_servers,
            num_cloud_servers=args.cloud_servers,
            num_tasks=args.tasks
        )
        
        # 创建模型
        delay_model = DelayModel(system)
        energy_model = EnergyModel(system)
        aoi_model = AoIModel(system, delay_model)
        
        # 运行基于加权和的单目标优化
        weighted_results = run_weighted_sum_optimization(
            system=system,
            max_iter=args.max_iter,
            population_size=args.pop_size
        )
        
        # 比较不同权重组合的结果
        compare_weighted_sum_results(
            weighted_results,
            save_path=os.path.join(args.output_dir, 'part2_weight_comparison.png')
        )
        
        # 运行多目标优化
        pareto_front, objectives = run_multi_objective_optimization(
            system=system,
            max_iter=args.max_iter,
            population_size=args.pop_size
        )
        
        # 绘制Pareto前沿
        plot_pareto_front(objectives, save_path=os.path.join(args.output_dir, 'part2_pareto_front.png'))
        
        # 分析更新间隔分布
        intervals = analyze_update_interval_distribution(
            pareto_front, system,
            save_path=os.path.join(args.output_dir, 'part2_update_interval_distribution.png')
        )
        
        # 分析任务分配分布
        allocation = analyze_task_allocation_distribution(
            pareto_front, system,
            save_path=os.path.join(args.output_dir, 'part2_task_allocation_distribution.png')
        )
        
        # 分析AoI对任务卸载决策的影响
        analyze_aoi_impact(
            system, delay_model, energy_model, aoi_model,
            save_path=os.path.join(args.output_dir, 'part2_aoi_impact.png')
        )
        
        print(f"\nFound {len(pareto_front)} non-dominated solutions in the Pareto front.")


if __name__ == "__main__":
    import numpy as np  # 导入numpy
    main()