"""
QoE (Quality of Experience) 模型
综合考虑三个维度：时延满足度、能耗接受度、任务完成率
"""
import numpy as np
from typing import Optional
from .system_model import SystemModel, Task
from .delay_model import DelayModel
from .energy_model import EnergyModel
from .aoi_model import AoIModel


class QoEModel:
    """用户体验质量模型"""

    def __init__(self, system_model: SystemModel, delay_model: DelayModel,
                 energy_model: EnergyModel, aoi_model: Optional[AoIModel] = None,
                 w_delay_qoe: float = 0.4, w_energy_qoe: float = 0.3,
                 w_completion_qoe: float = 0.3, beta: float = 0.5, gamma: float = 0.3):
        """
        初始化QoE模型

        Parameters:
        -----------
        system_model : SystemModel
            系统模型
        delay_model : DelayModel
            时延模型
        energy_model : EnergyModel
            能耗模型
        aoi_model : AoIModel, optional
            AoI模型（可选）
        w_delay_qoe : float
            时延满足度权重 (default=0.4)
        w_energy_qoe : float
            能耗接受度权重 (default=0.3)
        w_completion_qoe : float
            任务完成率权重 (default=0.3)
        beta : float
            时延满足度sigmoid函数的陡峭度参数 (default=0.5)
        gamma : float
            电池感知的惩罚系数 (default=0.3)
        """
        self.system_model = system_model
        self.delay_model = delay_model
        self.energy_model = energy_model
        self.aoi_model = aoi_model

        self.w_delay_qoe = w_delay_qoe
        self.w_energy_qoe = w_energy_qoe
        self.w_completion_qoe = w_completion_qoe

        self.beta = beta  # sigmoid函数的陡峭度参数
        self.gamma = gamma  # 电池感知的惩罚系数

        # 归一化权重
        total_weight = w_delay_qoe + w_energy_qoe + w_completion_qoe
        if total_weight > 0:
            self.w_delay_qoe /= total_weight
            self.w_energy_qoe /= total_weight
            self.w_completion_qoe /= total_weight

    def calculate_delay_satisfaction(self, task: Task, actual_delay: float) -> float:
        """
        计算时延满足度 - 使用sigmoid函数

        公式: S_delay = 1 / (1 + exp(β * (T - T_max)))

        Parameters:
        -----------
        task : Task
            任务对象
        actual_delay : float
            实际时延

        Returns:
        --------
        float : 时延满足度 [0, 1]
        """
        try:
            # 使用任务特定的时延敏感度系数（如果有）
            if hasattr(task, 'delay_sensitivity'):
                beta = task.delay_sensitivity
            else:
                beta = self.beta

            # Sigmoid函数计算满足度
            # 当实际时延远小于最大时延时，满足度接近1
            # 当实际时延远大于最大时延时，满足度接近0
            exponent = beta * (actual_delay - task.max_delay)

            # 防止指数溢出
            exponent = np.clip(exponent, -20, 20)

            satisfaction = 1.0 / (1.0 + np.exp(exponent))

            return np.clip(satisfaction, 0.0, 1.0)

        except Exception:
            return 0.0

    def calculate_energy_acceptance(self, task: Task, actual_energy: float,
                                   device_battery_level: float = 0.8) -> float:
        """
        计算能耗接受度 - 电池感知的动态调整

        公式:
        - B >= 60%: S_energy = 1 - E/E_budget
        - 30% <= B < 60%: S_energy = (1 - E/E_budget) * (1 + γ*(0.6-B))
        - B < 30%: S_energy = (1 - E/E_budget) * (1 + 2γ*(0.3-B))

        Parameters:
        -----------
        task : Task
            任务对象
        actual_energy : float
            实际能耗
        device_battery_level : float
            设备剩余电量比例 [0, 1] (default=0.8)

        Returns:
        --------
        float : 能耗接受度 [0, 1]
        """
        try:
            # 获取能耗预算
            if hasattr(task, 'energy_budget') and task.energy_budget > 0:
                energy_budget = task.energy_budget
            else:
                # 如果没有预设能耗预算，使用一个合理的估算值
                # 基于任务大小和复杂度
                energy_budget = task.data_size * task.computation_complexity * 1e-27 * 2

            # 基础能耗接受度
            if energy_budget > 0:
                base_acceptance = max(0.0, 1.0 - actual_energy / energy_budget)
            else:
                base_acceptance = 0.5  # 默认值

            # 电池感知调整
            battery_level = np.clip(device_battery_level, 0.0, 1.0)

            if battery_level >= 0.6:
                # 电量充足，不调整
                acceptance = base_acceptance
            elif battery_level >= 0.3:
                # 电量中等，轻微惩罚
                penalty = 1.0 + self.gamma * (0.6 - battery_level)
                acceptance = base_acceptance * penalty
            else:
                # 电量低，较大惩罚
                penalty = 1.0 + 2 * self.gamma * (0.3 - battery_level)
                acceptance = base_acceptance * penalty

            return np.clip(acceptance, 0.0, 1.0)

        except Exception:
            return 0.0

    def calculate_completion_rate(self, task: Task, actual_delay: float,
                                  actual_aoi: Optional[float] = None) -> float:
        """
        计算任务完成率

        公式: S_completion = 1 if (T <= T_max and AoI <= AoI_max) else 0

        Parameters:
        -----------
        task : Task
            任务对象
        actual_delay : float
            实际时延
        actual_aoi : float, optional
            实际AoI值

        Returns:
        --------
        float : 完成率 {0, 1}
        """
        try:
            # 检查时延约束
            delay_satisfied = actual_delay <= task.max_delay

            # 检查AoI约束（如果考虑）
            if actual_aoi is not None and hasattr(task, 'max_aoi') and task.max_aoi is not None:
                aoi_satisfied = actual_aoi <= task.max_aoi
            else:
                aoi_satisfied = True  # 如果不考虑AoI，则默认满足

            # 只有两个条件都满足时才认为任务成功完成
            if delay_satisfied and aoi_satisfied:
                return 1.0
            else:
                return 0.0

        except Exception:
            return 0.0

    def calculate_qoe(self, task: Task, actual_delay: float, actual_energy: float,
                     actual_aoi: Optional[float] = None,
                     device_battery_level: float = 0.8) -> float:
        """
        计算综合QoE评分

        公式: QoE = w1*S_delay + w2*S_energy + w3*S_completion

        Parameters:
        -----------
        task : Task
            任务对象
        actual_delay : float
            实际时延
        actual_energy : float
            实际能耗
        actual_aoi : float, optional
            实际AoI值
        device_battery_level : float
            设备电池电量 [0, 1]

        Returns:
        --------
        float : QoE评分 [0, 1]
        """
        try:
            # 计算三个维度的满足度
            s_delay = self.calculate_delay_satisfaction(task, actual_delay)
            s_energy = self.calculate_energy_acceptance(task, actual_energy, device_battery_level)
            s_completion = self.calculate_completion_rate(task, actual_delay, actual_aoi)

            # 加权求和
            qoe = (self.w_delay_qoe * s_delay +
                   self.w_energy_qoe * s_energy +
                   self.w_completion_qoe * s_completion)

            return np.clip(qoe, 0.0, 1.0)

        except Exception:
            return 0.0

    def calculate_system_qoe(self, solution, device_battery_levels: Optional[dict] = None) -> float:
        """
        计算系统平均QoE

        公式: QoE_avg = (1/K) * Σ QoE_i

        Parameters:
        -----------
        solution : List[List[float]]
            卸载方案
        device_battery_levels : dict, optional
            设备电池电量字典 {device_id: battery_level}

        Returns:
        --------
        float : 系统平均QoE [0, 1]
        """
        try:
            # 应用解到系统
            self.system_model.apply_solution(solution)

            # 默认电池电量
            if device_battery_levels is None:
                device_battery_levels = {d.device_id: 0.8 for d in self.system_model.devices}

            total_qoe = 0.0
            valid_tasks = 0

            for task in self.system_model.tasks:
                # 计算时延
                delay = self.delay_model.calculate_total_delay(task)

                # 计算能耗
                energy = self.energy_model.calculate_total_energy(task)

                # 计算AoI（如果有）
                aoi = None
                if self.aoi_model is not None and task.update_interval is not None:
                    try:
                        aoi = self.aoi_model.calculate_average_aoi(task)
                    except Exception:
                        pass

                # 获取设备电池电量
                if task.source_device_id is not None and task.source_device_id in device_battery_levels:
                    battery_level = device_battery_levels[task.source_device_id]
                else:
                    battery_level = 0.8  # 默认值

                # 计算QoE
                qoe = self.calculate_qoe(task, delay, energy, aoi, battery_level)

                if np.isfinite(qoe):
                    total_qoe += qoe
                    valid_tasks += 1

            if valid_tasks > 0:
                return total_qoe / valid_tasks
            else:
                return 0.0

        except Exception:
            return 0.0
