# -*- coding: utf-8 -*-
"""
core/evaluator.py - 多维气象风场评估模块 (增强版)
=================================================
设计原则:
  1. 保持与原代码 ModelEvaluator 类的接口兼容 (evaluate 方法)
  2. 新增气象专业指标: 风向圆周误差、安全 MAPE
  3. 返回值统一为结构化字典，供 main.py 记录和可视化使用
  4. 向量化计算，消灭 Python 慢循环
"""

import numpy as np
import traceback
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.helpers import print_info, print_success, print_warning, print_error

# ==================== 内部辅助函数 ====================

def _safe_mape(y_true, y_pred, epsilon=1e-8, calm_threshold=1.0):
    """
    安全的平均绝对百分比误差 (MAPE)。
    【核心改进】：原版代码在真实风速为 0 (静风) 时会触发除零崩溃。
    新版在分母中加入极小值 epsilon，确保数学上的绝对安全。
    【改进】增加静风样本过滤，排除风速 < calm_threshold 的样本后再计算 MAPE，
    避免静风条件下 MAPE 被 epsilon 人为低估。
    
    Args:
        y_true: 真实风速值
        y_pred: 预测风速值
        epsilon: 防止除零的极小值
        calm_threshold: 静风阈值，风速低于此值的样本被排除（默认 1.0 m/s）
    """
    # 【改进】过滤静风样本，仅在非静风条件下计算 MAPE
    valid = np.abs(y_true) >= calm_threshold
    if valid.sum() == 0:
        print_warning("无有效非静风样本（全部风速 < {:.1f} m/s），MAPE 返回 0.0".format(calm_threshold))
        return 0.0
    
    mape_val = np.mean(np.abs((y_true[valid] - y_pred[valid]) / (np.abs(y_true[valid]) + epsilon))) * 100.0
    return mape_val

def _circular_mae_wind_direction(dir_true, dir_pred):
    """
    风向圆周平均绝对误差 (Circular MAE for Wind Direction)。
    【核心亮点】：气象学专有指标。完美解决 359° 与 1° 误差被错算为 358° 的致命问题。
    公式：min(|true - pred|, 360 - |true - pred|)
    """
    # 确保风向在 [0, 360) 范围内
    dir_true = np.mod(dir_true, 360.0)
    dir_pred = np.mod(dir_pred, 360.0)
    
    # 计算绝对差值
    abs_diff = np.abs(dir_true - dir_pred)
    
    # 取圆周最短路径
    circular_diff = np.minimum(abs_diff, 360.0 - abs_diff)
    
    return np.mean(circular_diff)

def _circular_rmse_wind_direction(dir_true, dir_pred):
    """
    风向圆周均方根误差 (Circular RMSE for Wind Direction)。
    【新增】对大误差更敏感的圆周误差指标，补充 Circular MAE 的不足。
    公式：sqrt(mean(min(|true - pred|, 360 - |true - pred|)^2))
    """
    dir_true = np.mod(dir_true, 360.0)
    dir_pred = np.mod(dir_pred, 360.0)
    
    abs_diff = np.abs(dir_true - dir_pred)
    circular_diff = np.minimum(abs_diff, 360.0 - abs_diff)
    
    return np.sqrt(np.mean(circular_diff ** 2))

def _circular_stats_wind_direction(dir_true, dir_pred):
    """
    风向圆周误差分布统计。
    【新增】提供风向误差的中位数、95分位数等分布统计信息，
    弥补只看 MAE 无法反映长尾分布的不足。
    """
    dir_true = np.mod(dir_true, 360.0)
    dir_pred = np.mod(dir_pred, 360.0)
    
    abs_diff = np.abs(dir_true - dir_pred)
    circular_diff = np.minimum(abs_diff, 360.0 - abs_diff)
    
    return {
        "circular_mae": np.mean(circular_diff),
        "circular_rmse": np.sqrt(np.mean(circular_diff ** 2)),
        "circular_median": np.median(circular_diff),
        "circular_p95": np.percentile(circular_diff, 95),
        "circular_max": np.max(circular_diff),
    }

def _sanitize_inputs(y_true, y_pred):
    """
    输入数据清洗与维度对齐器。
    剔除 NaN 和 Inf，防止脏数据污染全局评估指标。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不匹配: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # 展平以便进行全局掩码操作
    orig_shape = y_true.shape
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 联合掩码：剔除任意一方为 NaN 或 Inf 的样本
    valid_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    invalid_count = np.sum(~valid_mask)
    
    if invalid_count > 0:
        print_warning(f"评估数据中发现 {invalid_count} 个无效值 (NaN/Inf)，已自动剔除。")
    
    return y_true_flat[valid_mask], y_pred_flat[valid_mask], orig_shape, invalid_count

# ==================== 核心对外接口 ====================

class ModelEvaluator:
    """
    多维气象风场评估器。
    【保留原点】：完整保留原代码的 MSE/RMSE/MAE/R² 基础指标计算逻辑。
    【增强点】：
      1. 新增风向圆周误差指标 (_circular_mae_wind_direction)
      2. 新增安全 MAPE (_safe_mape)，解决静风除零问题
      3. 输入数据清洗 (_sanitize_inputs)，防止脏数据导致崩溃
      4. 返回值统一为结构化字典，供 main.py 记录
      5. 【新增】分量级指标报告，按输出维度分别报告 MSE/RMSE/MAE/R²
      6. 【新增】Circular RMSE 和风向误差分布统计
    """
    
    def evaluate(self, y_true, y_pred, model_name="Model"):
        """
        多维度评估接口。
        
        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测结果
            model_name (str): 模型名称 (用于日志标识)
        
        Returns:
            metrics (dict): 结构化指标字典
                包含: MSE, RMSE, MAE, R², WS_MAPE, WD_Circular_MAE 等
        """
        # 1. 数据清洗与防御
        y_t_clean, y_p_clean, orig_shape, n_invalid = _sanitize_inputs(y_true, y_pred)
        
        if len(y_t_clean) == 0:
            print_warning("清洗后无有效数据，无法计算指标，返回空字典。")
            return {}
        
        # 2. 重塑回 2D 以区分风速和风向
        if len(orig_shape) > 1 and orig_shape[-1] >= 2:
            y_t_2d = y_t_clean.reshape(-1, orig_shape[-1])
            y_p_2d = y_p_clean.reshape(-1, orig_shape[-1])
        else:
            y_t_2d = y_t_clean.reshape(-1, 1)
            y_p_2d = y_p_clean.reshape(-1, 1)
        
        metrics = {
            "model_name": model_name,
            "valid_samples": len(y_t_clean),
            "invalid_samples": n_invalid,
        }
        
        # 3. 全局基础指标 (向量化计算，极速) - 保留原代码逻辑
        metrics["MSE"] = mean_squared_error(y_t_clean, y_p_clean)
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["MAE"] = mean_absolute_error(y_t_clean, y_p_clean)
        
        # R² 决定系数 (需捕获常数列导致的 ValueError)
        try:
            metrics["R^2"] = r2_score(y_t_clean, y_p_clean)
        except ValueError:
            metrics["R^2"] = float('nan')
        
        # 4. 【新增】分量级指标报告 - 按输出维度分别计算，避免全局指标掩盖分量差异
        per_component = {}
        dim_names = ["风速", "风向", "第3维度"]
        for col_idx in range(y_t_2d.shape[1]):
            col_name = f"dim{col_idx}"
            col_metrics = {}
            
            col_true = y_t_2d[:, col_idx]
            col_pred = y_p_2d[:, col_idx]
            
            col_metrics["MSE"] = mean_squared_error(col_true, col_pred)
            col_metrics["RMSE"] = np.sqrt(col_metrics["MSE"])
            col_metrics["MAE"] = mean_absolute_error(col_true, col_pred)
            try:
                col_metrics["R^2"] = r2_score(col_true, col_pred)
            except ValueError:
                col_metrics["R^2"] = float('nan')
            
            per_component[col_name] = col_metrics
            metrics[f"{col_name}_MSE"] = col_metrics["MSE"]
            metrics[f"{col_name}_RMSE"] = col_metrics["RMSE"]
            metrics[f"{col_name}_MAE"] = col_metrics["MAE"]
            metrics[f"{col_name}_R^2"] = col_metrics["R^2"]
        
        metrics["per_component"] = per_component
        
        # 5. 风速专项评估 (Wind Speed) - 新增气象专业指标
        # 假设第 0 列为风速
        if y_t_2d.shape[1] > 0:
            ws_true = y_t_2d[:, 0]
            ws_pred = y_p_2d[:, 0]
            metrics["WS_MAPE"] = _safe_mape(ws_true, ws_pred)
        
        # 6. 风向专项评估 (Wind Direction) - 核心气象指标
        # 假设第 1 列为风向 (如果存在)
        if y_t_2d.shape[1] > 1:
            wd_true = y_t_2d[:, 1]
            wd_pred = y_p_2d[:, 1]
            metrics["WD_Circular_MAE"] = _circular_mae_wind_direction(wd_true, wd_pred)
            # 【新增】Circular RMSE
            metrics["WD_Circular_RMSE"] = _circular_rmse_wind_direction(wd_true, wd_pred)
            # 【新增】风向误差分布统计
            wd_stats = _circular_stats_wind_direction(wd_true, wd_pred)
            metrics["WD_Circular_Median"] = wd_stats["circular_median"]
            metrics["WD_Circular_P95"] = wd_stats["circular_p95"]
            metrics["WD_Circular_Max"] = wd_stats["circular_max"]
        
        # 7. 控制台输出 - 保留原代码的 print_info 风格
        print_info(f"[{model_name}] MSE", f"{metrics.get('MSE', 0):.6f}")
        print_info(f"[{model_name}] RMSE", f"{metrics.get('RMSE', 0):.6f}")
        print_info(f"[{model_name}] MAE", f"{metrics.get('MAE', 0):.6f}")
        print_info(f"[{model_name}] R² Score", f"{metrics.get('R^2', 0):.4f}")
        
        # 新增指标输出
        if "WS_MAPE" in metrics:
            print_info(f"[{model_name}] WS MAPE", f"{metrics['WS_MAPE']:.2f}%")
        if "WD_Circular_MAE" in metrics:
            print_info(f"[{model_name}] WD Circular MAE", f"{metrics['WD_Circular_MAE']:.2f}°")
        if "WD_Circular_RMSE" in metrics:
            print_info(f"[{model_name}] WD Circular RMSE", f"{metrics['WD_Circular_RMSE']:.2f}°")
        if "WD_Circular_P95" in metrics:
            print_info(f"[{model_name}] WD Circular P95", f"{metrics['WD_Circular_P95']:.2f}°")
        
        # 【修复】分量级指标输出：print_info 必须传 title + msg 两个参数
        print_info(f"[{model_name}] === 分量级指标 ===", "")
        for col_idx in range(y_t_2d.shape[1]):
            col_name = dim_names[col_idx] if col_idx < len(dim_names) else f"维度{col_idx}"
            r2_val = metrics.get(f"dim{col_idx}_R^2", float('nan'))
            rmse_val = metrics.get(f"dim{col_idx}_RMSE", 0)
            mae_val = metrics.get(f"dim{col_idx}_MAE", 0)
            # 【修复】拆分为 title + msg 双参数，避免 missing 'msg' 崩溃
            print_info(f"  [{col_name}]", f"RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  R²={r2_val:.4f}")
        
        print_success(f"{model_name} 评估完成")
        
        return metrics


# ==================== 模块级便捷接口 ====================
# 【新增】以下两个函数供 menu.py 的 evl.evaluate_model() / evl.evaluate_all_models() 调用

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    单模型评估便捷接口。
    
    Args:
        y_true: 真实标签
        y_pred: 预测结果
        model_name: 模型名称
    
    Returns:
        metrics (dict): 结构化指标字典
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate(y_true, y_pred, model_name=model_name)


def evaluate_all_models(y_true, y_pred_dict, feature_names=None):
    """
    多模型批量评估接口。
    
    Args:
        y_true: 真实标签 (np.ndarray)
        y_pred_dict: 模型名 -> 预测数组的字典，如 {"Linear": y_pred_lin, "ResNet": y_pred_res}
        feature_names: 特征名列表 (预留，当前未使用)
    
    Returns:
        results (dict): 模型名 -> 指标字典的映射
    """
    results = {}
    evaluator = ModelEvaluator()
    
    for name, y_pred in y_pred_dict.items():
        print_info("评估模型", name)
        
        # 【新增】防御性校验：跳过空预测
        if y_pred is None or (hasattr(y_pred, 'size') and y_pred.size == 0):
            print_warning(f"{name} 预测数据为空，跳过评估")
            results[name] = {}
            continue
            
        try:
            metrics = evaluator.evaluate(y_true, y_pred, model_name=name)
            results[name] = metrics
        except Exception as e:
            # 【修复】补全 print_error 缺失的 msg 参数，并补充堆栈信息便于排查
            print_error(f"{name} 评估失败", f"{e}")
            print_info("堆栈", traceback.format_exc())
            results[name] = {}
    
    return results


# 【新增】兼容旧版 main.py 中可能直接调用的 print_info 包装
def print_info_safe(title, msg=""):
    """安全打印信息，确保始终传两个参数。"""
    print_info(title, msg)