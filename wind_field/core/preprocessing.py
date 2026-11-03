import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config.settings import HAMPEL_WINDOW, HAMPEL_THRESHOLD, SG_WINDOW, SG_POLYORDER
from utils.helpers import print_info, print_success, print_warning, print_step

# ================= 内部辅助函数：异常值检测与清洗 =================

def _detect_outliers(data, low=-999.0, high=999.0):
    """
    物理无效值与数学异常值检测器。
    【修改】在原有 NaN/Inf 检测基础上，增加 3 倍标准差硬截断，
    防止极端异常值逃逸到后续 Hampel 阶段。
    """
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"数据必须为数值类型，当前类型为: {data.dtype}")

    # 基础掩码：NaN, Inf, 以及超出物理合理范围的极值 (如雷达占位符 999/-999)
    mask = np.isnan(data) | np.isinf(data) | (data <= low) | (data >= high)

    # 【修复】逐列硬截断：超过 3 倍标准差的点标记为异常
    # 原代码用 np.isfinite(col_data).all() 判断，若列中存在任何 NaN/Inf 会跳过整列
    # 改为仅用有限值计算统计量，确保所有列都能参与 3σ 检测
    for col in range(data.shape[1]):
        col_data = data[:, col]
        finite_mask = np.isfinite(col_data)
        if np.any(finite_mask):
            mu = np.mean(col_data[finite_mask])
            sigma = np.std(col_data[finite_mask])
            if sigma > 1e-8:
                mask[:, col] |= (np.abs(col_data - mu) > 3.0 * sigma)

    return mask

def _clean_outliers_global(data, method="median"):
    """
    全局异常值清洗器。
    【修改】原 low/high 范围太宽松，配合 _detect_outliers 的 3σ 扩展后，
    能更有效地拦截极端离群点。
    """
    mask = _detect_outliers(data)
    n_outliers = int(np.sum(mask))

    if n_outliers == 0:
        return data, 0

    cleaned = data.copy()

    if method == "median":
        valid_mask = ~mask
        for col in range(data.shape[1]):
            col_mask = mask[:, col]
            if np.any(col_mask):
                valid_values = data[valid_mask[:, col], col]
                if len(valid_values) > 0:
                    col_median = np.median(valid_values)
                    cleaned[col_mask, col] = col_median
                else:
                    cleaned[col_mask, col] = 0.0
                    print_warning(f"第 {col} 列全为异常值，已强制填充为 0.0。")

    return cleaned, n_outliers


# ================= 内部辅助函数：Hampel 滤波 (核心性能优化) =================

def _fast_hampel_filter_1d(series, window=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD):
    """
    高性能一维 Hampel 滤波器。
    【核心修复】原代码 MAD 计算缺少 1.4826 正态分布缩放因子，
    且用 clip(1e-10) 强行截断导致阈值虚高，离群点检测失效。
    新版采用标准 MAD 公式：MAD_scaled = 1.4826 * median(|x - median|)。
    """
    if window % 2 == 0:
        window += 1

    if len(series) < window:
        return series

    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()

    diff = (series - rolling_median).abs()
    rolling_mad = diff.rolling(window=window, center=True, min_periods=1).median()

    # 【修复】用标准缩放因子替代粗暴的 clip(1e-10)
    mad_scale = 1.4826
    rolling_mad_scaled = rolling_mad * mad_scale
    rolling_mad_scaled = rolling_mad_scaled.clip(lower=1e-10)

    # 识别离群点
    outlier_mask = diff > (threshold * rolling_mad_scaled)

    filtered_series = series.copy()
    filtered_series[outlier_mask] = rolling_median[outlier_mask]

    return filtered_series

def _apply_hampel_filter(data, window=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD):
    """多维矩阵 Hampel 滤波包装器。"""
    df = pd.DataFrame(data)
    df_filtered = df.apply(lambda col: _fast_hampel_filter_1d(col, window, threshold))
    return df_filtered.values.astype(np.float32)


# ================= 内部辅助函数：Savitzky-Golay 滤波 (数学严谨性保留) =================

def _apply_savgol_filter(data, window=SG_WINDOW, polyorder=SG_POLYORDER):
    """Savitzky-Golay 滤波器。逻辑与原代码保持一致。"""
    n_samples = data.shape[0]

    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = window - 1

    if n_samples <= polyorder:
        print_warning(f"数据样本数({n_samples})过少，无法进行 S-G 滤波 (需 > {polyorder})，跳过平滑。")
        return data

    wl = window
    if wl > n_samples:
        wl = n_samples if n_samples % 2 != 0 else n_samples - 1
    if wl <= polyorder:
        wl = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 1
    if wl % 2 == 0:
        wl += 1

    try:
        smoothed_data = savgol_filter(data, window_length=wl, polyorder=polyorder, axis=0, mode='interp')
        return smoothed_data.astype(np.float32)
    except Exception as e:
        print_warning(f"S-G 滤波执行失败 ({e})，返回原始数据。")
        return data


# ================= 核心对外接口：预处理主流程 =================

def preprocess_pipeline(X_raw, y_raw, use_hampel=True, use_sg=True):
    """数据清洗与时序滤波主流程。流程保持不变。"""
    t_start = time.time()
    print_step("1.2", "执行数据清洗与时序滤波流程")
    report = {}

    print_info("阶段 1/3", "全局物理异常值清洗 (拦截雷达占位符 + 3σ硬截断)")
    X_clean, n_x = _clean_outliers_global(X_raw)
    y_clean, n_y = _clean_outliers_global(y_raw)
    report["X_cleaned"], report["y_cleaned"] = n_x, n_y
    print_info("清洗报告", f"拦截并替换了 X:{n_x} 个, y:{n_y} 个全局异常值。")

    if use_hampel:
        print_info("阶段 2/3", f"执行 Hampel 滤波 (窗口:{HAMPEL_WINDOW}, 阈值:{HAMPEL_THRESHOLD}σ)")
        X_clean = _apply_hampel_filter(X_clean, HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        y_clean = _apply_hampel_filter(y_clean, HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        report["hampel_applied"] = True
    else:
        print_info("阶段 2/3", "Hampel 滤波已禁用 (use_hampel=False)")
        report["hampel_applied"] = False

    if use_sg:
        print_info("阶段 3/3", f"执行 S-G 滤波 (窗口:{SG_WINDOW}, 阶数:{SG_POLYORDER})")
        X_clean = _apply_savgol_filter(X_clean, SG_WINDOW, SG_POLYORDER)
        y_clean = _apply_savgol_filter(y_clean, SG_WINDOW, SG_POLYORDER)
        report["sg_applied"] = True
    else:
        print_info("阶段 3/3", "S-G 滤波已禁用 (use_sg=False)")
        report["sg_applied"] = False

    elapsed = time.time() - t_start
    print_success(f"数据清洗与滤波流程完成 (总耗时: {elapsed:.3f}s)")
    report["elapsed_time"] = elapsed

    return X_clean, y_clean, report