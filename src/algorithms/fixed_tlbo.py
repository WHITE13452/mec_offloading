# src/algorithms/fixed_tlbo.py
import numpy as np
from .fixed_base_algorithm import FixedBaseAlgorithm

class FixedTLBO(FixedBaseAlgorithm):
    """修复后的TLBO算法"""
    
    def __init__(self, system_model, delay_model, energy_model, aoi_model=None,
                 max_iter=100, population_size=50, w_energy=0.4, w_delay=0.6, w_aoi=0.0, verbose=False):
        super().__init__(system_model, max_iter, population_size, verbose)
        
        self.delay_model = delay_model
        self.energy_model = energy_model
        self.aoi_model = aoi_model
        
        self.w_energy = w_energy
        self.w_delay = w_delay
        self.w_aoi = w_aoi
        
        self.history = []
    
    def evaluate_fitness(self, solution):
        """稳定的适应度评估"""
        try:
            # 确保解的格式正确
            if not solution or len(solution) != self.num_tasks:
                return float('inf')
            
            # 应用约束处理
            solution = self.handle_constraints(solution)
            
            # 将解应用到系统模型
            self.system_model.apply_solution(solution)
            
            total_energy = 0.0
            total_delay = 0.0
            total_aoi = 0.0
            
            # 计算所有任务的指标
            for task in self.system_model.tasks:
                try:
                    # 计算延迟
                    delay = self.delay_model.calculate_total_delay(task)
                    if not np.isfinite(delay) or delay < 0:
                        return float('inf')
                    total_delay += delay
                    
                    # 计算能耗
                    energy = self.energy_model.calculate_total_energy(task)
                    if not np.isfinite(energy) or energy < 0:
                        return float('inf')
                    total_energy += energy
                    
                    # 计算AoI
                    if self.consider_aoi and self.aoi_model and task.update_interval is not None:
                        aoi = self.aoi_model.calculate_average_aoi(task)
                        if not np.isfinite(aoi) or aoi < 0:
                            return float('inf')
                        total_aoi += aoi
                        
                except Exception:
                    return float('inf')
            
            # 检查总值
            if not (np.isfinite(total_energy) and np.isfinite(total_delay)):
                return float('inf')
            
            # 使用预设的归一化因子计算适应度
            fitness = (self.w_energy * total_energy / self.energy_max + 
                      self.w_delay * total_delay / self.delay_max)
            
            if self.consider_aoi and self.aoi_model:
                fitness += self.w_aoi * total_aoi / self.aoi_max
            
            return fitness if np.isfinite(fitness) else float('inf')
            
        except Exception:
            return float('inf')
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            solution = []
            for _ in range(self.num_tasks):
                loc_i = np.random.randint(self.loc_bounds[0], self.loc_bounds[1] + 1)
                f_i = np.random.uniform(self.freq_bounds[0], self.freq_bounds[1])
                
                if self.consider_aoi:
                    delta_i = np.random.uniform(self.update_interval_bounds[0], self.update_interval_bounds[1])
                    solution.append([loc_i, f_i, delta_i])
                else:
                    solution.append([loc_i, f_i])
            
            solution = self.handle_constraints(solution)
            population.append(solution)
        
        return population
    
    def teacher_phase(self, population, fitness_values):
        """修复的教师阶段"""
        new_population = []
        
        try:
            best_idx = np.argmin(fitness_values)
            teacher = population[best_idx]
            mean_solution = np.mean(population, axis=0)
            
            for i, student in enumerate(population):
                tf = np.random.randint(1, 3)
                r = np.random.rand(*np.array(student).shape)
                
                # 安全的数组运算
                try:
                    new_student = np.array(student) + r * (np.array(teacher) - tf * mean_solution)
                    new_student = self.handle_constraints(new_student.tolist())
                    
                    new_fitness = self.evaluate_fitness(new_student)
                    
                    if np.isfinite(new_fitness) and new_fitness < fitness_values[i]:
                        new_population.append(new_student)
                    else:
                        new_population.append(student)
                        
                except Exception:
                    new_population.append(student)
            
            return new_population
            
        except Exception:
            return population
    
    def learner_phase(self, population, fitness_values):
        """修复的学习者阶段"""
        new_population = []
        
        try:
            for i, student in enumerate(population):
                j = i
                while j == i:
                    j = np.random.randint(0, self.population_size)
                
                other_student = population[j]
                r = np.random.rand(*np.array(student).shape)
                
                try:
                    if fitness_values[i] < fitness_values[j]:
                        new_student = np.array(student) + r * (np.array(student) - np.array(other_student))
                    else:
                        new_student = np.array(student) + r * (np.array(other_student) - np.array(student))
                    
                    new_student = self.handle_constraints(new_student.tolist())
                    new_fitness = self.evaluate_fitness(new_student)
                    
                    if np.isfinite(new_fitness) and new_fitness < fitness_values[i]:
                        new_population.append(new_student)
                    else:
                        new_population.append(student)
                        
                except Exception:
                    new_population.append(student)
            
            return new_population
            
        except Exception:
            return population
    
    def optimize(self):
        """执行优化"""
        population = self.initialize_population()
        fitness_values = [self.evaluate_fitness(solution) for solution in population]
        
        # 移除无效解
        valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
        if not valid_indices:
            print("Warning: No valid solutions found!")
            return None, float('inf'), []
        
        best_idx = min(valid_indices, key=lambda i: fitness_values[i])
        self.best_solution = population[best_idx]
        self.best_fitness = fitness_values[best_idx]
        self.history = [self.best_fitness]
        
        for iter_idx in range(self.max_iter):
            try:
                population = self.teacher_phase(population, fitness_values)
                fitness_values = [self.evaluate_fitness(solution) for solution in population]
                
                population = self.learner_phase(population, fitness_values)
                fitness_values = [self.evaluate_fitness(solution) for solution in population]
                
                # 更新最佳解
                valid_indices = [i for i, f in enumerate(fitness_values) if np.isfinite(f)]
                if valid_indices:
                    best_idx = min(valid_indices, key=lambda i: fitness_values[i])
                    if fitness_values[best_idx] < self.best_fitness:
                        self.best_solution = population[best_idx]
                        self.best_fitness = fitness_values[best_idx]
                
                self.history.append(self.best_fitness)
                
                if self.verbose and (iter_idx + 1) % 10 == 0:
                    print(f"Iteration {iter_idx + 1}/{self.max_iter}, Best fitness: {self.best_fitness:.6f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error in iteration {iter_idx}: {e}")
                break
        
        return self.best_solution, self.best_fitness, self.history