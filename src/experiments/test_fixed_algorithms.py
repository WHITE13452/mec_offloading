# test_fixed_algorithms.py
import numpy as np
from src.experiments.realistic_system_setup import create_realistic_edge_computing_system
from src.models.delay_model import DelayModel
from src.models.energy_model import EnergyModel
from src.algorithms.fixed_tlbo import FixedTLBO
from src.algorithms.tlbo import TLBO
from src.algorithms.ga import GA
from src.algorithms.tlbo_hho import TLBOHHO

def test_fixed_algorithm():
    system = create_realistic_edge_computing_system(num_devices=10, num_edge_servers=3, num_cloud_servers=2, num_tasks=20)
    delay_model = DelayModel(system)
    energy_model = EnergyModel(system)
    
    algorithm = TLBOHHO(system, delay_model, energy_model, max_iter=50, population_size=20, verbose=True)
    
    best_solution, best_fitness, history = algorithm.optimize()
    
    print(f"\n修复后的结果:")
    print(f"最优适应度: {best_fitness}")
    print(f"收敛历史长度: {len(history)}")
    print(f"是否有NaN: {any(not np.isfinite(h) for h in history)}")

if __name__ == "__main__":
    test_fixed_algorithm()