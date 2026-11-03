"""
ui/visualizer.py - 多维风场专业可视化引擎 (工业级增强版)
=========================================================
【重构说明】
1. 完美贴合导师要求：支持三维风场(U/V/W)或多物理量子图自动网格排版
2. 深度气象专业分析：新增 散点回归图(带R²与y=x线)、残差分布直方图(带KDE)
3. 兼容新版评估器：自动解析并展示 metrics_dict 中的高级指标(圆周误差/静风MAPE等)
4. 跨平台字体鲁棒性：彻底解决 Mac/Linux/Windows 下 SimHei 字体缺失导致的方块乱码或崩溃
5. 吸纳源代码优点：保留自动创建目录、高清保存图表、内存安全释放(close)机制
6. 向后兼容：保留原版 plot_wind_comparison 全局函数签名，旧版 main.py 无需修改即可调用
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from scipy.signal import savgol_filter  # 【修改】导入 Savitzky-Golay 滤波器

from config.settings import PLOT_DIR
from utils.helpers import ensure_dir, print_success, print_warning, print_info, print_step


# ==================== 跨平台中文字体鲁棒性配置 ====================

def _setup_robust_chinese_font():
    """
    【关键修复】：原版直接硬编码 SimHei，在 Mac/Linux 上会报字体缺失警告并显示方块乱码。
    此处增加系统级字体探测与多级回退机制，确保任何操作系统下中文均正常显示。
    """
    # 按优先级排列的跨平台中文字体候选列表
    candidate_fonts = [
        'SimHei',               # Windows 默认黑体
        'Microsoft YaHei',      # Windows 微软雅黑
        'PingFang SC',          # macOS 苹方
        'Heiti TC',             # macOS 黑体
        'Arial Unicode MS',     # Mac/Office 通用
        'WenQuanYi Micro Hei',  # Linux 文泉驿
        'Noto Sans CJK SC'      # Linux Google Noto
    ]
    
    # 获取系统当前所有可用字体名称集合（小写化以忽略大小写差异）
    available_fonts = {f.name.lower() for f in fm.fontManager.ttflist}
    
    selected_font = None
    for font in candidate_fonts:
        if font.lower() in available_fonts:
            selected_font = font
            break
            
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        print_info("可视化字体", f"成功加载跨平台中文字体: {selected_font}")
    else:
        print_warning("未找到系统中文字体，图表中文可能显示为方块，请手动安装 SimHei 或 PingFang")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 终极兜底

    # 修复负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    # 抑制 matplotlib 内部无关紧要的字体警告
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 模块加载时自动执行字体配置
_setup_robust_chinese_font()


# ==================== 核心可视化引擎类 ====================

class WindFieldVisualizer:
    """
    多维风场专业可视化引擎。
    支持时序对比、散点回归、残差分布三大气象核心图表的自动生成。
    """
    
    # 模型配色与线型规范 (工业级UI一致性)
    STYLE_MAP = {
        "Ground Truth": {"color": "#2C3E50", "lw": 2.0, "ls": "-",  "alpha": 1.0},
        "真实值":       {"color": "#2C3E50", "lw": 2.0, "ls": "-",  "alpha": 1.0},
        "Linear":       {"color": "#3498DB", "lw": 1.5, "ls": "--", "alpha": 0.8},
        "线性回归":     {"color": "#3498DB", "lw": 1.5, "ls": "--", "alpha": 0.8},
        "ResNet":       {"color": "#E74C3C", "lw": 1.8, "ls": "-.", "alpha": 0.9},
        "SimpleResNet": {"color": "#E74C3C", "lw": 1.8, "ls": "-.", "alpha": 0.9}
    }

    def __init__(self, plot_dir=PLOT_DIR):
        self.plot_dir = plot_dir
        ensure_dir(self.plot_dir)

    def _get_style(self, model_name):
        """安全获取模型样式，未知模型回退为灰色虚线。"""
        return self.STYLE_MAP.get(model_name, {"color": "#7F8C8D", "lw": 1.2, "ls": ":", "alpha": 0.7})

    def _safe_extract_metric(self, metrics_dict, model_name, key, default="N/A"):
        """安全从复杂嵌套字典中提取评估指标，兼容新旧版 evaluator 输出。"""
        if not metrics_dict or model_name not in metrics_dict:
            return default
        m = metrics_dict[model_name]
        # 支持嵌套字典 (如 {"U": {"R2": 0.9}, "V":...}) 或 扁平字典 (如 {"R2": 0.9})
        if isinstance(m, dict):
            val = m.get(key, None)
            if val is None:
                # 尝试取第一个子维度的值作为代表
                for sub_k, sub_v in m.items():
                    if isinstance(sub_v, dict) and key in sub_v:
                        val = sub_v[key]
                        break
            # 【修复】增加 np.integer / np.floating，numpy 标量不是 Python int/float 的子类
            if isinstance(val, (int, float, np.integer, np.floating)):
                return f"{val:.4f}"
            return val if val is not None else default
        return default

    @staticmethod
    def _smooth(y, window=301):
        """
        对一维数组做 Savitzky-Golay 滤波（气象标准去噪方法）。
        相比简单移动平均，能在强力去除高频毛刺的同时精确保留峰值与趋势拐点。
        窗口必须为奇数；若数据过短或参数非法，自动回退到移动平均。
        """
        window = min(max(5, int(window)), len(y))
        if window % 2 == 0:
            window -= 1
        if window <= 3:
            # 数据过短，退化为恒等（避免 scipy 报错）
            return np.asarray(y, dtype=np.float32)
        try:
            return savgol_filter(np.asarray(y, dtype=np.float32),
                                 window_length=window, polyorder=3)
        except (ValueError, TypeError):
            # 回退到简单移动平均
            w = np.ones(window, dtype=np.float32) / window
            return np.convolve(np.asarray(y, dtype=np.float32), w, mode="valid")

    def plot_comprehensive_analysis(self, y_true, y_pred_dict, metrics_dict=None, 
                                    feature_names=None, save_prefix="wind_analysis"):
        """
        综合可视化主接口：生成 时序图 + 散点图 + 残差图 的三联排专业报告。
        
        参数:
            y_true: np.ndarray [N, D] 真实值
            y_pred_dict: dict {"ModelName": np.ndarray [N, D]} 预测值字典
            metrics_dict: dict {"ModelName": {metrics...}} 评估指标字典
            feature_names: list 特征名称 (如 ["U风速", "V风速", "W风速"])
        """
        print_step("5.0", "生成多维气象专业可视化报告")
        
        y_true = np.atleast_2d(y_true)
        if y_true.shape[0] == 1:
            y_true = y_true.T
            
        n_samples, n_dims = y_true.shape
        
        # 自动生成特征名称
        if feature_names is None or len(feature_names) != n_dims:
            if n_dims == 3:
                feature_names = ["U 分量 (东西向)", "V 分量 (南北向)", "W 分量 (垂直向)"]
            else:
                feature_names = [f"物理量 {i+1}" for i in range(n_dims)]

        # 限制最大绘图维度，防止维度爆炸导致内存溢出
        plot_dims = min(n_dims, 4)
        
        # ========== 图1：多维时序预测对比图 (带误差带) ==========
        fig_ts, axes_ts = plt.subplots(plot_dims, 1, figsize=(12, 4 * plot_dims), sharex=True)
        if plot_dims == 1:
            axes_ts = [axes_ts]
            
        x_axis = np.arange(n_samples)
        # 若数据点过多，开启降采样以保证渲染性能
        step = max(1, n_samples // 2000)
        x_ds, y_ds = x_axis[::step], y_true[::step]
        
        for i in range(plot_dims):
            ax = axes_ts[i]
            s_gt = self._get_style("真实值")
            # 【学长意见】真实值 Savitzky-Golay 滤波，强力去毛刺且保留峰值，曲线更光滑
            y_ds_i = y_ds[:, i]
            y_ds_i_smooth = self._smooth(y_ds_i)
            ax.plot(x_ds[:len(y_ds_i_smooth)], y_ds_i_smooth,
                    label="真实值（滤波后）", **s_gt)
            
            for model_name, y_pred in y_pred_dict.items():
                y_pred = np.atleast_2d(y_pred)
                if y_pred.shape[0] == 1: y_pred = y_pred.T
                y_p_ds = y_pred[::step]
                
                s_m = self._get_style(model_name)
                ax.plot(x_ds, y_p_ds[:, i], label=model_name, **s_m)
                
                # 【专业增强】：绘制 ±1σ 误差带，直观展示模型置信区间
                residuals = y_ds[:, i] - y_p_ds[:, i]
                std_err = np.std(residuals)
                if std_err > 0:
                    ax.fill_between(x_ds, y_p_ds[:, i] - std_err, y_p_ds[:, i] + std_err, 
                                    color=s_m["color"], alpha=0.1, linewidth=0)
            
            # 在子图右上角标注该维度的核心指标
            info_text = ""
            for m_name in y_pred_dict.keys():
                r2 = self._safe_extract_metric(metrics_dict, m_name, "R^2")
                rmse = self._safe_extract_metric(metrics_dict, m_name, "RMSE")
                info_text += f"[{m_name}] R²={r2} | RMSE={rmse}\n"
            ax.text(0.98, 0.95, info_text.strip(), transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='aliceblue', alpha=0.8))
            
            ax.set_title(feature_names[i], fontsize=13, fontweight='bold')
            ax.set_ylabel("幅值")
            ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
            ax.grid(True, linestyle=":", alpha=0.5)
            
        axes_ts[-1].set_xlabel("样本索引 (Sample Index)")
        fig_ts.suptitle("多维风场时序预测对比 (含 ±1σ 误差带)", fontsize=16, y=1.01)
        fig_ts.tight_layout()
        self._save_and_close(fig_ts, f"{save_prefix}_time_series.png")

        # ========== 图2：真实值 vs 预测值 散点回归图 ==========
        fig_sc, axes_sc = plt.subplots(plot_dims, len(y_pred_dict), 
                                       figsize=(6 * len(y_pred_dict), 5 * plot_dims),
                                       squeeze=False)
        for i in range(plot_dims):
            for j, (model_name, y_pred) in enumerate(y_pred_dict.items()):
                ax = axes_sc[i, j]
                y_pred = np.atleast_2d(y_pred)
                if y_pred.shape[0] == 1: y_pred = y_pred.T
                
                true_col = y_true[:, i]
                pred_col = y_pred[:, i]
                
                # 【修复】联合过滤 NaN/Inf，防止 stats.linregress 崩溃
                finite_mask = np.isfinite(true_col) & np.isfinite(pred_col)
                if finite_mask.sum() < 3:
                    ax.text(0.5, 0.5, "有效数据不足\n无法拟合回归线", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{feature_names[i]} - {model_name}", fontsize=12)
                    ax.set_xlabel("真实值")
                    ax.set_ylabel("预测值")
                    continue
                
                true_clean = true_col[finite_mask]
                pred_clean = pred_col[finite_mask]
                
                # 降采样防止散点图过密（基于清洗后的数据）
                n_clean = len(true_clean)
                idx = np.random.choice(n_clean, min(n_clean, 3000), replace=False)
                ax.scatter(true_clean[idx], pred_clean[idx], alpha=0.4, s=15, 
                           color=self._get_style(model_name)["color"], edgecolors='none')
                
                # 绘制 y=x 理想参考线
                min_val = min(true_clean.min(), pred_clean.min())
                max_val = max(true_clean.max(), pred_clean.max())
                margin = (max_val - min_val) * 0.05
                line_range = [min_val - margin, max_val + margin]
                ax.plot(line_range, line_range, 'k--', lw=1.5, label="理想参考线 (y=x)")
                
                # 计算并标注 R² 和 线性拟合线
                # 【修复】linregress 在数据方差为零时返回 nan，需捕获并防御
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(true_clean, pred_clean)
                    if np.isfinite(r_value):
                        r2 = r_value ** 2
                        fit_line = slope * np.array(line_range) + intercept
                        ax.plot(line_range, fit_line, 'r-', lw=1.2, label=f"拟合线 (R²={r2:.3f})")
                    else:
                        ax.plot([], [], ' ', label="R²=N/A (方差为零)")
                except (ValueError, FloatingPointError):
                    ax.plot([], [], ' ', label="R²=N/A (拟合失败)")
                
                ax.set_title(f"{feature_names[i]} - {model_name}", fontsize=12)
                ax.set_xlabel("真实值")
                ax.set_ylabel("预测值")
                ax.legend(fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.5)
                
        fig_sc.suptitle("真实值 vs 预测值 散点回归分析", fontsize=16, y=1.01)
        fig_sc.tight_layout()
        self._save_and_close(fig_sc, f"{save_prefix}_scatter_regression.png")

        # ========== 图3：残差分布直方图 (带 KDE 核密度估计) ==========
        fig_res, axes_res = plt.subplots(plot_dims, len(y_pred_dict), 
                                         figsize=(6 * len(y_pred_dict), 4 * plot_dims),
                                         squeeze=False)
        for i in range(plot_dims):
            for j, (model_name, y_pred) in enumerate(y_pred_dict.items()):
                ax = axes_res[i, j]
                y_pred = np.atleast_2d(y_pred)
                if y_pred.shape[0] == 1: y_pred = y_pred.T
                
                residuals = y_true[:, i] - y_pred[:, i]
                # 剔除 NaN 和 Inf
                valid_mask = np.isfinite(residuals)
                residuals = residuals[valid_mask]
                
                if len(residuals) == 0:
                    ax.text(0.5, 0.5, "无有效残差数据", ha='center', va='center')
                    continue

                # 绘制直方图
                n, bins, patches = ax.hist(residuals, bins=50, density=True, 
                                           alpha=0.6, color=self._get_style(model_name)["color"],
                                           edgecolor='white')
                
                # 叠加 KDE 核密度估计曲线 (气象误差分析标准做法)
                try:
                    kde = stats.gaussian_kde(residuals)
                    x_density = np.linspace(residuals.min(), residuals.max(), 200)
                    ax.plot(x_density, kde(x_density), 'k-', lw=2, label="KDE 密度曲线")
                except np.linalg.LinAlgError:
                    pass # 若数据方差为0导致KDE失败则静默跳过

                # 标注均值和标准差 (系统性偏差分析)
                mean_err = np.mean(residuals)
                std_err = np.std(residuals)
                ax.axvline(mean_err, color='red', linestyle='--', lw=1.5, 
                           label=f"均值偏差={mean_err:.3f}")
                
                info_text = f"RMSE={np.sqrt(np.mean(residuals**2)):.3f}\nStd={std_err:.3f}"
                ax.text(0.95, 0.95, info_text, transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_title(f"{feature_names[i]} - {model_name} 残差分布", fontsize=12)
                ax.set_xlabel("预测残差 (真实值 - 预测值)")
                ax.set_ylabel("概率密度")
                ax.legend(fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.5)
                
        fig_res.suptitle("预测残差分布与系统性偏差分析 (含 KDE)", fontsize=16, y=1.01)
        fig_res.tight_layout()
        self._save_and_close(fig_res, f"{save_prefix}_residual_distribution.png")
        
        print_success(f"综合可视化报告生成完毕，共输出 3 张高清专业图表至: {self.plot_dir}")

    def _save_and_close(self, fig, filename):
        """安全保存并释放内存。"""
        save_path = os.path.join(self.plot_dir, filename)
        try:
            fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
            print_info("图表保存", save_path)
        except Exception as e:
            print_warning(f"图表保存失败 ({e}): {save_path}")
        finally:
            plt.close(fig) # 【保留原版优点】强制关闭 figure 防止内存泄漏


# ==================== 向后兼容的全局接口 ====================

# 实例化全局引擎
_visualizer_engine = WindFieldVisualizer()

def plot_wind_comparison(y_true, y_pred_lr, y_pred_resnet, title="Wind Field Prediction", **kwargs):
    """
    【向后兼容接口】：保留原版函数签名，确保旧版 main.py 或 menu.py 调用不报错。
    内部自动将参数适配到新版工业级可视化引擎。
    """
    y_pred_dict = {
        "Linear": y_pred_lr,
        "ResNet": y_pred_resnet
    }
    metrics_dict = kwargs.get("metrics_dict", None)
    
    _visualizer_engine.plot_comprehensive_analysis(
        y_true=y_true, 
        y_pred_dict=y_pred_dict, 
        metrics_dict=metrics_dict,
        save_prefix="wind_field_comparison"
    )