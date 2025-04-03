# src/models/aoi_model.py 
from typing import Dict, Optional
from .system_model import SystemModel, Task
from .delay_model import DelayModel


class AoIModel:
    """计算信息年龄(AoI)的模型"""
    
    def __init__(self, system_model: SystemModel, delay_model: DelayModel):
        """
        初始化AoI模型
        
        Parameters:
        -----------
        system_model : SystemModel
            系统模型实例
        delay_model : DelayModel
            延迟模型实例，用于计算任务延迟
        """
        self.system_model = system_model
        self.delay_model = delay_model
    
    def calculate_average_aoi(self, task: Task) -> float:
        """
        计算给定任务的平均信息年龄(AoI)
        
        Parameters:
        -----------
        task : Task
            要计算AoI的任务
            
        Returns:
        --------
        float
            平均AoI（单位：秒）
        """
        if task.update_interval is None:
            raise ValueError(f"Task {task.task_id} does not have an update interval defined.")
        
        # 确保任务的延迟已计算
        if task.delay is None:
            task.delay = self.delay_model.calculate_total_delay(task)
        
        # 对于周期性任务，平均AoI = Δ/2 + T
        # 其中Δ是更新间隔，T是任务延迟
        average_aoi = task.update_interval / 2 + task.delay
        
        # 更新任务的AoI字段
        task.aoi = average_aoi
        
        return average_aoi
    
    def calculate_peak_aoi(self, task: Task) -> float:
        """
        计算给定任务的峰值信息年龄(AoI)
        
        Parameters:
        -----------
        task : Task
            要计算峰值AoI的任务
            
        Returns:
        --------
        float
            峰值AoI（单位：秒）
        """
        if task.update_interval is None:
            raise ValueError(f"Task {task.task_id} does not have an update interval defined.")
        
        # 确保任务的延迟已计算
        if task.delay is None:
            task.delay = self.delay_model.calculate_total_delay(task)
        
        # 峰值AoI = Δ + T
        # 其中Δ是更新间隔，T是任务延迟
        peak_aoi = task.update_interval + task.delay
        
        return peak_aoi
    
    def optimize_update_interval(self, task: Task, alpha: float = 0.5, beta: float = 0.5, 
                                min_interval: float = 0.1, max_interval: float = 10.0) -> float:
        """
        优化任务的更新间隔，平衡AoI和能耗
        
        Parameters:
        -----------
        task : Task
            要优化更新间隔的任务
        alpha : float
            AoI的权重系数
        beta : float
            能耗的权重系数
        min_interval : float
            最小允许的更新间隔
        max_interval : float
            最大允许的更新间隔
            
        Returns:
        --------
        float
            优化后的更新间隔
        """
        from scipy.optimize import minimize_scalar
        
        # 确保任务的延迟和能耗已计算
        if task.delay is None or task.energy is None:
            task.delay = self.delay_model.calculate_total_delay(task)
            # 假设能耗模型已经计算了任务的能耗
        
        # 定义目标函数：alpha * AoI + beta * (能耗/时间单位)
        def objective(interval):
            aoi = interval / 2 + task.delay
            energy_rate = task.energy / interval  # 单位时间内的能耗
            return alpha * aoi + beta * energy_rate
        
        # 使用单变量优化方法找到最优更新间隔
        result = minimize_scalar(objective, bounds=(min_interval, max_interval), method='bounded')
        
        if result.success:
            optimal_interval = result.x
            # 更新任务的更新间隔
            task.update_interval = optimal_interval
            # 重新计算AoI
            task.aoi = self.calculate_average_aoi(task)
            return optimal_interval
        else:
            raise RuntimeError(f"Failed to optimize update interval for task {task.task_id}: {result.message}")
    
    def calculate_aoi_for_all_tasks(self) -> Dict[int, float]:
        """
        计算系统中所有任务的平均AoI
        
        Returns:
        --------
        Dict[int, float]
            任务ID到平均AoI的映射
        """
        aoi_values = {}
        
        for task in self.system_model.tasks:
            if task.update_interval is not None:
                try:
                    aoi = self.calculate_average_aoi(task)
                    aoi_values[task.task_id] = aoi
                except ValueError as e:
                    print(f"Warning: {e}")
        
        return aoi_values
    
    def calculate_weighted_aoi(self, task_weights: Dict[int, float] = None) -> float:
        """
        计算加权平均AoI
        
        Parameters:
        -----------
        task_weights : Dict[int, float], optional
            任务权重字典，如果为None，则使用任务优先级
            
        Returns:
        --------
        float
            加权平均AoI
        """
        if task_weights is None:
            # 使用任务优先级作为权重
            task_weights = {task.task_id: task.priority for task in self.system_model.tasks if task.update_interval is not None}
        
        # 计算所有任务的AoI
        aoi_values = self.calculate_aoi_for_all_tasks()
        
        # 计算加权平均
        total_weight = sum(task_weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_aoi = sum(aoi_values.get(task_id, 0) * weight for task_id, weight in task_weights.items()) / total_weight
        
        return weighted_aoi