"""
公平性模型 - 基于Jain公平性指数
衡量系统中各任务QoE分配的公平性
"""
import numpy as np
from typing import List, Optional
from .system_model import SystemModel
from .qoe_model import QoEModel


class FairnessModel:
    """公平性度量模型"""

    def __init__(self, system_model: SystemModel, qoe_model: QoEModel):
        """
        初始化公平性模型

        Parameters:
        -----------
        system_model : SystemModel
            系统模型
        qoe_model : QoEModel
            QoE模型
        """
        self.system_model = system_model
        self.qoe_model = qoe_model

    def calculate_jain_fairness_index(self, qoe_scores: List[float]) -> float:
        """
        计算Jain公平性指数

        公式: Fairness = (Σ QoE_i)² / (K * Σ QoE_i²)

        该指数衡量资源分配的公平性:
        - 值接近1表示完全公平（所有用户QoE相同）
        - 值接近0表示不公平（QoE分布极不均匀）

        Parameters:
        -----------
        qoe_scores : List[float]
            所有任务的QoE评分列表

        Returns:
        --------
        float : Jain公平性指数 [0, 1]，1表示完全公平
        """
        try:
            # 过滤掉无效值
            valid_scores = [score for score in qoe_scores if np.isfinite(score)]

            if not valid_scores or len(valid_scores) == 0:
                return 0.0

            # 转换为numpy数组以便计算
            scores_array = np.array(valid_scores)
            K = len(scores_array)

            if K == 0:
                return 0.0

            # 计算分子: (Σ QoE_i)²
            sum_qoe = np.sum(scores_array)
            numerator = sum_qoe ** 2

            # 计算分母: K * Σ QoE_i²
            sum_qoe_squared = np.sum(scores_array ** 2)
            denominator = K * sum_qoe_squared

            # 避免除零
            if denominator == 0 or not np.isfinite(denominator):
                return 0.0

            # 计算Jain指数
            fairness = numerator / denominator

            # 确保结果在[0, 1]范围内
            return np.clip(fairness, 0.0, 1.0)

        except Exception:
            return 0.0

    def calculate_fairness(self, solution, device_battery_levels: Optional[dict] = None) -> float:
        """
        基于卸载方案计算系统公平性

        Parameters:
        -----------
        solution : List[List[float]]
            卸载方案
        device_battery_levels : dict, optional
            设备电池电量字典 {device_id: battery_level}

        Returns:
        --------
        float : 公平性指数 [0, 1]
        """
        try:
            # 应用解到系统
            self.system_model.apply_solution(solution)

            # 默认电池电量
            if device_battery_levels is None:
                device_battery_levels = {d.device_id: 0.8 for d in self.system_model.devices}

            # 收集所有任务的QoE评分
            qoe_scores = []

            for task in self.system_model.tasks:
                # 计算时延
                delay = self.qoe_model.delay_model.calculate_total_delay(task)

                # 计算能耗
                energy = self.qoe_model.energy_model.calculate_total_energy(task)

                # 计算AoI（如果有）
                aoi = None
                if (self.qoe_model.aoi_model is not None and
                    task.update_interval is not None):
                    try:
                        aoi = self.qoe_model.aoi_model.calculate_average_aoi(task)
                    except Exception:
                        pass

                # 获取设备电池电量
                if (task.source_device_id is not None and
                    task.source_device_id in device_battery_levels):
                    battery_level = device_battery_levels[task.source_device_id]
                else:
                    battery_level = 0.8  # 默认值

                # 计算QoE
                qoe = self.qoe_model.calculate_qoe(task, delay, energy, aoi, battery_level)

                if np.isfinite(qoe):
                    qoe_scores.append(qoe)

            # 计算Jain公平性指数
            fairness = self.calculate_jain_fairness_index(qoe_scores)

            return fairness

        except Exception:
            return 0.0

    def calculate_variance_based_fairness(self, qoe_scores: List[float]) -> float:
        """
        基于方差的公平性度量（补充指标）

        使用变异系数的倒数作为公平性度量:
        Fairness = 1 / (1 + CV)
        其中 CV = std / mean (变异系数)

        Parameters:
        -----------
        qoe_scores : List[float]
            所有任务的QoE评分列表

        Returns:
        --------
        float : 公平性指数 [0, 1]
        """
        try:
            # 过滤掉无效值
            valid_scores = [score for score in qoe_scores if np.isfinite(score)]

            if not valid_scores or len(valid_scores) < 2:
                return 0.0

            scores_array = np.array(valid_scores)

            mean_qoe = np.mean(scores_array)
            std_qoe = np.std(scores_array)

            # 避免除零
            if mean_qoe == 0:
                return 0.0

            # 计算变异系数
            cv = std_qoe / mean_qoe

            # 基于变异系数的公平性
            fairness = 1.0 / (1.0 + cv)

            return np.clip(fairness, 0.0, 1.0)

        except Exception:
            return 0.0

    def calculate_detailed_fairness_metrics(self, solution,
                                          device_battery_levels: Optional[dict] = None) -> dict:
        """
        计算详细的公平性指标

        Returns:
        --------
        dict : 包含多个公平性指标的字典
        {
            'jain_index': Jain公平性指数,
            'variance_based': 基于方差的公平性,
            'min_qoe': 最小QoE,
            'max_qoe': 最大QoE,
            'mean_qoe': 平均QoE,
            'std_qoe': QoE标准差,
            'qoe_scores': 所有QoE评分列表
        }
        """
        try:
            # 应用解到系统
            self.system_model.apply_solution(solution)

            # 默认电池电量
            if device_battery_levels is None:
                device_battery_levels = {d.device_id: 0.8 for d in self.system_model.devices}

            # 收集所有任务的QoE评分
            qoe_scores = []

            for task in self.system_model.tasks:
                # 计算时延
                delay = self.qoe_model.delay_model.calculate_total_delay(task)

                # 计算能耗
                energy = self.qoe_model.energy_model.calculate_total_energy(task)

                # 计算AoI（如果有）
                aoi = None
                if (self.qoe_model.aoi_model is not None and
                    task.update_interval is not None):
                    try:
                        aoi = self.qoe_model.aoi_model.calculate_average_aoi(task)
                    except Exception:
                        pass

                # 获取设备电池电量
                if (task.source_device_id is not None and
                    task.source_device_id in device_battery_levels):
                    battery_level = device_battery_levels[task.source_device_id]
                else:
                    battery_level = 0.8

                # 计算QoE
                qoe = self.qoe_model.calculate_qoe(task, delay, energy, aoi, battery_level)

                if np.isfinite(qoe):
                    qoe_scores.append(qoe)

            # 计算各种公平性指标
            metrics = {
                'jain_index': self.calculate_jain_fairness_index(qoe_scores),
                'variance_based': self.calculate_variance_based_fairness(qoe_scores),
                'min_qoe': min(qoe_scores) if qoe_scores else 0.0,
                'max_qoe': max(qoe_scores) if qoe_scores else 0.0,
                'mean_qoe': np.mean(qoe_scores) if qoe_scores else 0.0,
                'std_qoe': np.std(qoe_scores) if qoe_scores else 0.0,
                'qoe_scores': qoe_scores
            }

            return metrics

        except Exception:
            return {
                'jain_index': 0.0,
                'variance_based': 0.0,
                'min_qoe': 0.0,
                'max_qoe': 0.0,
                'mean_qoe': 0.0,
                'std_qoe': 0.0,
                'qoe_scores': []
            }
