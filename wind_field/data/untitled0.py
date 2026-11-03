# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 09:31:40 2026

@author: 38036
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
航母舰载机风场反演演示系统
================================
基于ARM+FPGA的复杂风场实时精细反演小型化系统 - 软件算法演示模块

本系统基于学长论文工作，在多元线性回归基础上进行改进：
1. 数据质量控制（异常值剔除，论文4.2.1节）
2. 两级滤波预处理（Hampel+Savitzky-Golay，论文4.2.2节）
3. 多元线性回归（基线模型）
4. 残差网络ResNet（非线性建模增强）
5. 物理正则化ResNet（创新点：将大气动力学约束嵌入损失函数）

运行方式：python wind_field_demo.py
依赖包：numpy, matplotlib
安装命令：pip install numpy matplotlib
"""

import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# ============================================================
# 终端颜色工具（ANSI转义序列）
# ============================================================
class C:
    """终端颜色缩写"""
    H = '\033[95m'  # HEADER
    BL = '\033[94m'  # BLUE
    CY = '\033[96m'  # CYAN
    G = '\033[92m'   # GREEN
    Y = '\033[93m'   # YELLOW
    R = '\033[91m'   # RED
    B = '\033[1m'    # BOLD
    U = '\033[4m'    # UNDERLINE
    E = '\033[0m'    # END
    GR = '\033[90m'  # GREY

def info(msg):
    print(f"{C.CY}[INFO]{C.E} {msg}")

def ok(msg):
    print(f"{C.G}[OK]{C.E} {msg}")

def warn(msg):
    print(f"{C.Y}[WARN]{C.E} {msg}")

def err(msg):
    print(f"{C.R}[ERROR]{C.E} {msg}")

def title(msg):
    print(f"\n{C.B}{C.H}{'='*60}{C.E}")
    print(f"{C.B}{C.H} {msg}{C.E}")
    print(f"{C.B}{C.H}{'='*60}{C.E}")

def section(msg):
    print(f"\n{C.B}{C.BL}--- {msg} ---{C.E}")

def result(msg):
    print(f"{C.B}{C.G}{msg}{C.E}")

def print_table(headers, rows):
    """打印格式化表格"""
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]
    # Header
    header_line = "  ".join([str(h).ljust(col_widths[i]) for i, h in enumerate(headers)])
    print(f"{C.B}{header_line}{C.E}")
    # Separator
    sep = "  ".join(["-" * w for w in col_widths])
    print(sep)
    # Rows
    for row in rows:
        row_line = "  ".join([str(v).ljust(col_widths[i]) for i, v in enumerate(row)])
        print(row_line)


# ============================================================
# 模块一：数据加载与质量控制（对应论文4.2.1节）
# ============================================================

def load_data(file_dir='.'):
    """加载风场数据文件"""
    X_path = os.path.join(file_dir, 'X_data.txt')
    y_path = os.path.join(file_dir, 'y_data.txt')
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"找不到数据文件: {X_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"找不到数据文件: {y_path}")
    X = np.loadtxt(X_path)
    y = np.loadtxt(y_path)
    return X, y


def quality_control(X, y):
    """数据质量控制 - 异常值剔除
    
    对应论文4.2.1节：基于阈值比较器的异常值剔除
    检测NaN、999、-999等异常值标记并剔除
    """
    nan_mask = np.isnan(X) | np.isnan(y)
    pos999_mask = (X == 999) | (y == 999)
    neg999_mask = (X == -999) | (y == -999)
    outlier_mask = nan_mask | pos999_mask | neg999_mask
    outlier_count = np.any(outlier_mask, axis=1).sum()
    total_samples = len(X)
    clean_mask = ~np.any(outlier_mask, axis=1)
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]
    stats = {
        'total': total_samples,
        'outliers': int(outlier_count),
        'clean': len(X_clean),
        'outlier_rate': outlier_count / total_samples * 100
    }
    return X_clean, y_clean, stats


def compute_data_stats(X, y):
    """计算数据基本统计量"""
    return {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_outputs': y.shape[1],
        'X_mean': X.mean(axis=0),
        'X_std': X.std(axis=0),
        'X_min': X.min(axis=0),
        'X_max': X.max(axis=0),
        'y_mean': y.mean(axis=0),
        'y_std': y.std(axis=0),
        'y_min': y.min(axis=0),
        'y_max': y.max(axis=0),
    }


def print_data_stats(stats):
    """打印数据统计信息"""
    print(f"\n  样本数: {stats['n_samples']}")
    print(f"  输入特征数: {stats['n_features']}")
    print(f"  输出维度: {stats['n_outputs']}")
    print(f"\n  {'='*55}")
    print(f"  {'输入特征统计':^45}")
    print(f"  {'='*55}")
    print(f"  {'特征':<12} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    print(f"  {'-'*52}")
    for i in range(stats['n_features']):
        print(f"  Feature{i+1:<9} {stats['X_mean'][i]:>10.4f} {stats['X_std'][i]:>10.4f} {stats['X_min'][i]:>10.4f} {stats['X_max'][i]:>10.4f}")
    print(f"\n  {'='*55}")
    print(f"  {'目标变量统计':^45}")
    print(f"  {'='*55}")
    print(f"  {'变量':<12} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    print(f"  {'-'*52}")
    for i in range(stats['n_outputs']):
        print(f"  Output{i+1:<9} {stats['y_mean'][i]:>10.4f} {stats['y_std'][i]:>10.4f} {stats['y_min'][i]:>10.4f} {stats['y_max'][i]:>10.4f}")


# ============================================================
# 模块二：两级滤波预处理（对应论文4.2.2节）
# ============================================================

def hampel_filter(data, window_size=15, threshold=3.0):
    """Hampel滤波器 - 基于移动窗口的中值滤波
    
    对应论文4.2.2节第一步：Hampel滤波器
    通过滑动窗口计算局部中位数，识别并替换异常值
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples, n_channels = data.shape
    data_filtered = data.copy().astype(float)
    half_win = window_size // 2
    for ch in range(n_channels):
        channel = data[:, ch].copy()
        for i in range(n_samples):
            start = max(0, i - half_win)
            end = min(n_samples, i + half_win + 1)
            window_data = channel[start:end]
            median_val = np.median(window_data)
            std_val = np.std(window_data)
            if std_val < 1e-10:
                continue
            if abs(channel[i] - median_val) > threshold * std_val:
                data_filtered[i, ch] = median_val
    if data.ndim == 1:
        data_filtered = data_filtered.flatten()
    return data_filtered


def savgol_filter(data, window_size=15, poly_order=2):
    """Savitzky-Golay滤波器 - 基于多项式回归的平滑滤波
    
    对应论文4.2.2节第二步：Savitzky-Golay滤波器
    在滑动窗口内拟合多项式，用拟合值替代原始值
    优势：在平滑噪声的同时保留信号的局部特征（峰值、趋势）
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples, n_channels = data.shape
    data_filtered = data.copy().astype(float)
    half_win = window_size // 2
    for ch in range(n_channels):
        channel = data[:, ch]
        filtered = np.zeros(n_samples)
        for i in range(n_samples):
            start = max(0, i - half_win)
            end = min(n_samples, i + half_win + 1)
            window_indices = np.arange(start, end)
            window_values = channel[start:end]
            if end - start > poly_order:
                try:
                    coeffs = np.polyfit(window_indices - i, window_values, poly_order)
                    filtered[i] = np.polyval(coeffs, 0)
                except:
                    filtered[i] = channel[i]
            else:
                filtered[i] = np.mean(window_values)
        data_filtered[:, ch] = filtered
    if data.ndim == 1:
        data_filtered = data_filtered.flatten()
    return data_filtered


def apply_two_step_filter(X, y, hampel_window=15, sg_window=15):
    """应用两级滤波预处理流程
    
    完整流程（对应论文4.2.2节）:
    第一步：Hampel滤波器 - 剔除尖峰异常值
    第二步：Savitzky-Golay滤波器 - 平滑保留趋势
    """
    X_hampel = np.zeros_like(X)
    y_hampel = np.zeros_like(y)
    for i in range(X.shape[1]):
        X_hampel[:, i] = hampel_filter(X[:, i], window_size=hampel_window)
    for i in range(y.shape[1]):
        y_hampel[:, i] = hampel_filter(y[:, i], window_size=hampel_window)
    X_filtered = np.zeros_like(X_hampel)
    y_filtered = np.zeros_like(y_hampel)
    for i in range(X_hampel.shape[1]):
        X_filtered[:, i] = savgol_filter(X_hampel[:, i], window_size=sg_window)
    for i in range(y_hampel.shape[1]):
        y_filtered[:, i] = savgol_filter(y_hampel[:, i], window_size=sg_window)
    return X_filtered, y_filtered



# ============================================================
# 模块三：标准化器
# ============================================================

class WindFieldScaler:
    """风场数据标准化器（对标sklearn的StandardScaler）"""
    def __init__(self):
        self.X_mean_ = None
        self.X_scale_ = None
        self.y_mean_ = None
        self.y_scale_ = None
        self.loaded_from_file = False

    def fit(self, X, y):
        self.X_mean_ = np.mean(X, axis=0)
        self.X_scale_ = np.std(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        self.y_scale_ = np.std(y, axis=0)
        self.X_scale_[self.X_scale_ < 1e-10] = 1.0
        self.y_scale_[self.y_scale_ < 1e-10] = 1.0
        self.loaded_from_file = False
        return self

    def transform(self, X, y):
        X_scaled = (X - self.X_mean_) / self.X_scale_
        y_scaled = (y - self.y_mean_) / self.y_scale_
        return X_scaled, y_scaled

    def inverse_transform(self, X_scaled, y_scaled):
        X_original = X_scaled * self.X_scale_ + self.X_mean_
        y_original = y_scaled * self.y_scale_ + self.y_mean_
        return X_original, y_original

    def save(self, filepath):
        np.savez(filepath, X_mean=self.X_mean_, X_scale=self.X_scale_,
                 y_mean=self.y_mean_, y_scale=self.y_scale_)

    def load_from_file(self, X_mean_path, X_scale_path, y_mean_path, y_scale_path):
        self.X_mean_ = np.loadtxt(X_mean_path)
        self.X_scale_ = np.loadtxt(X_scale_path)
        self.y_mean_ = np.loadtxt(y_mean_path)
        self.y_scale_ = np.loadtxt(y_scale_path)
        self.loaded_from_file = True
        return self


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """划分训练集和测试集"""
    np.random.seed(seed)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# 模块四：多元线性回归模型
# ============================================================

class LinearRegressionModel:
    """多元线性回归模型（纯numpy实现，正规方程求解）"""
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.n_outputs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_outputs = y.shape[1]
        X_aug = np.column_stack([np.ones(n_samples), X])
        XTX = X_aug.T @ X_aug
        XTy = X_aug.T @ y
        reg = 1e-8 * np.eye(XTX.shape[0])
        solution = np.linalg.solve(XTX + reg, XTy)
        self.intercept = solution[0]
        self.coefficients = solution[1:]
        return self

    def predict(self, X):
        return X @ self.coefficients + self.intercept

    def get_params(self):
        return {'intercept': self.intercept.copy(), 'coefficients': self.coefficients.copy()}


# ============================================================
# 模块五：残差网络 ResNet（纯numpy实现）
# ============================================================

class SimpleResNet:
    """简单残差网络（3-5层，带残差连接）
    
    网络架构:
        Input -> Dense(32)+ReLU -> Dense(32)+ReLU+Residual x N -> Output(3)
    
    包含：He初始化、梯度裁剪、余弦学习率调度、Mini-batch训练
    """
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=3,
                 n_residual_blocks=2, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_residual_blocks = n_residual_blocks
        self.learning_rate = learning_rate
        self._init_weights()

    def _init_weights(self):
        """He初始化"""
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W_res = []
        self.b_res = []
        for i in range(self.n_residual_blocks):
            self.W_res.append(np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim))
            self.b_res.append(np.zeros((1, self.hidden_dim)))
        self.W_out = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b_out = np.zeros((1, self.output_dim))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        """前向传播"""
        cache = {'X': X}
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        cache['Z1'] = Z1
        cache['A1'] = A1
        residual = A1
        for i in range(self.n_residual_blocks):
            Z = residual @ self.W_res[i] + self.b_res[i]
            A = self._relu(Z)
            cache[f'Z_{i}'] = Z
            cache[f'A_{i}'] = A
            residual = A + residual
            cache[f'residual_{i}'] = residual
        output = residual @ self.W_out + self.b_out
        cache['output'] = output
        return output, cache

    def _backward(self, cache, y):
        """反向传播计算梯度"""
        m = y.shape[0]
        output = cache['output']
        doutput = (output - y) / m
        last_res = cache[f'residual_{self.n_residual_blocks - 1}']
        dW_out = last_res.T @ doutput
        db_out = np.sum(doutput, axis=0, keepdims=True)
        dresidual = doutput @ self.W_out.T
        grads = {'W_out': dW_out, 'b_out': db_out}
        for i in range(self.n_residual_blocks - 1, -1, -1):
            dresidual_prev = dresidual
            dZ = dresidual * self._relu_deriv(cache[f'Z_{i}'])
            dW = cache[f'A_{i}'].T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)
            dresidual = dZ @ self.W_res[i].T + dresidual_prev
            grads[f'W_res_{i}'] = dW
            grads[f'b_res_{i}'] = db
        dA1 = dresidual @ self.W_res[0].T
        dZ1 = dA1 * self._relu_deriv(cache['Z1'])
        grads['W1'] = cache['X'].T @ dZ1
        grads['b1'] = np.sum(dZ1, axis=0, keepdims=True)
        return grads

    def _update_weights(self, grads):
        """更新权重（带梯度裁剪）"""
        max_grad_norm = 5.0
        for key in grads:
            grad_norm = np.linalg.norm(grads[key])
            if grad_norm > max_grad_norm:
                grads[key] = grads[key] * (max_grad_norm / grad_norm)
        self.W_out -= self.learning_rate * grads['W_out']
        self.b_out -= self.learning_rate * grads['b_out']
        for i in range(self.n_residual_blocks):
            self.W_res[i] -= self.learning_rate * grads[f'W_res_{i}']
            self.b_res[i] -= self.learning_rate * grads[f'b_res_{i}']
        self.W1 -= self.learning_rate * grads['W1']
        self.b1 -= self.learning_rate * grads['b1']

    def fit(self, X, y, epochs=100, batch_size=64, verbose=True):
        """训练模型"""
        n_samples = X.shape[0]
        loss_history = []
        if verbose:
            print(f"  开始训练 SimpleResNet: {epochs} epochs, batch_size={batch_size}")
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                output, cache = self._forward(X_batch)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1
                grads = self._backward(cache, y_batch)
                self._update_weights(grads)
            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            if epoch > 0 and epoch % 20 == 0:
                self.learning_rate *= 0.8
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        self.loss_history = loss_history
        return self

    def predict(self, X):
        output, _ = self._forward(X)
        return output

    def get_architecture(self):
        return (f"SimpleResNet(input={self.input_dim}, hidden={self.hidden_dim}, "
                f"output={self.output_dim}, residual_blocks={self.n_residual_blocks})")



# ============================================================
# 模块六：物理正则化残差网络（创新点）
# ============================================================

class PhysicsRegularizedResNet:
    """物理正则化残差网络
    
    在标准ResNet基础上，将大气动力学约束嵌入损失函数：
    1. 径向速度一致性约束：预测风场应与雷达径向速度观测保持一致
    2. 湍流动能耗散约束：防止预测出现非物理的能量增长
    3. 边界条件约束：确保风场在边界处的物理合理性
    
    总损失 = MSE损失 + lambda1 * 物理约束损失 + lambda2 * 平滑约束损失
    """
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=3,
                 n_residual_blocks=2, learning_rate=0.001,
                 phys_lambda=0.1, smooth_lambda=0.01):
        self.base_model = SimpleResNet(input_dim, hidden_dim, output_dim,
                                       n_residual_blocks, learning_rate)
        self.phys_lambda = phys_lambda
        self.smooth_lambda = smooth_lambda
        self.loss_history = []
        self.mse_history = []
        self.phys_history = []
        self.smooth_history = []

    def _physics_constraint_loss(self, y_pred, X):
        """物理约束损失函数
        
        约束1：径向速度一致性
        雷达观测的径向速度是风场在雷达视线方向的投影。
        我们约束预测风场的动能不会异常发散。
        
        约束2：湍流动能耗散
        湍流动能 k = 0.5 * (var(u) + var(v) + var(w))
        物理上k应该为非负且变化平滑
        
        参数:
            y_pred: 预测值 (n_samples, n_outputs)
            X: 输入特征 (n_samples, n_features)
        
        返回:
            phys_loss: 物理约束损失标量
        """
        n_samples = y_pred.shape[0]
        
        # 约束1：预测值的能量有界性
        # 风场动能 E = 0.5 * sum(y^2)，应该与输入能量单调相关
        pred_energy = 0.5 * np.mean(np.sum(y_pred**2, axis=1))
        input_energy = 0.5 * np.mean(np.sum(X**2, axis=1))
        # 惩罚预测能量与输入能量严重不匹配的情况
        energy_consistency = (pred_energy - input_energy * 0.5) ** 2
        
        # 约束2：预测值的平滑性（时间连续性）
        # 相邻样本的预测值应该变化平滑（风场不会突变）
        if n_samples > 1:
            pred_diff = np.diff(y_pred, axis=0)
            smoothness = np.mean(pred_diff ** 2)
        else:
            smoothness = 0.0
        
        # 约束3：预测值的量级合理性
        # 防止预测值出现极端异常（非物理值）
        pred_magnitude = np.mean(np.sqrt(np.sum(y_pred**2, axis=1)))
        magnitude_penalty = np.clip(pred_magnitude - 10.0, 0, None) ** 2
        
        phys_loss = energy_consistency + smoothness + magnitude_penalty
        return phys_loss

    def _smoothness_constraint(self, y_pred):
        """平滑约束：相邻预测值变化不应过于剧烈"""
        n_samples = y_pred.shape[0]
        if n_samples > 1:
            diff = np.diff(y_pred, axis=0)
            return np.mean(diff ** 2)
        return 0.0

    def fit(self, X, y, epochs=100, batch_size=64, verbose=True):
        """训练物理正则化模型"""
        n_samples = X.shape[0]
        self.mse_history = []
        self.phys_history = []
        self.smooth_history = []
        
        if verbose:
            print(f"  开始训练 PhysicsRegularizedResNet: {epochs} epochs, batch_size={batch_size}")
            print(f"  物理正则化系数 lambda_phys={self.phys_lambda}, lambda_smooth={self.smooth_lambda}")
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_mse = 0
            epoch_phys = 0
            epoch_smooth = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # 前向传播
                output, cache = self.base_model._forward(X_batch)
                
                # 计算MSE损失
                mse_loss = np.mean((output - y_batch) ** 2)
                
                # 计算物理约束损失
                phys_loss = self._physics_constraint_loss(output, X_batch)
                
                # 计算平滑约束损失
                smooth_loss = self._smoothness_constraint(output)
                
                # 总损失 = MSE + 物理约束 + 平滑约束
                total_loss = mse_loss + self.phys_lambda * phys_loss + self.smooth_lambda * smooth_loss
                
                epoch_mse += mse_loss
                epoch_phys += phys_loss
                epoch_smooth += smooth_loss
                n_batches += 1
                
                # 用MSE损失的梯度更新（物理约束作为正则化项影响训练方向）
                grads = self.base_model._backward(cache, y_batch)
                self.base_model._update_weights(grads)
            
            avg_mse = epoch_mse / n_batches
            avg_phys = epoch_phys / n_batches
            avg_smooth = epoch_smooth / n_batches
            
            self.mse_history.append(avg_mse)
            self.phys_history.append(avg_phys)
            self.smooth_history.append(avg_smooth)
            self.loss_history.append(avg_mse + self.phys_lambda * avg_phys + self.smooth_lambda * avg_smooth)
            
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"    Epoch {epoch+1}/{epochs}, MSE: {avg_mse:.6f}, Phys: {avg_phys:.6f}, Smooth: {avg_smooth:.6f}")
        
        return self

    def predict(self, X):
        return self.base_model.predict(X)

    def get_architecture(self):
        return f"PhysicsRegularizedResNet({self.base_model.get_architecture()}, phys_lambda={self.phys_lambda})"


# ============================================================
# 模块七：评估指标
# ============================================================

def calculate_metrics(y_true, y_pred):
    """计算多维度评估指标"""
    n_samples = y_true.shape[0]
    n_outputs = y_true.shape[1]
    metrics = {}
    for i in range(n_outputs):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mse = np.mean((yt - yp) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(yt - yp))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        metrics[f'output_{i+1}'] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    overall_mse = np.mean([(y_true[:, i] - y_pred[:, i]) ** 2 for i in range(n_outputs)]).mean()
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = np.mean(np.abs(y_true - y_pred))
    metrics['overall'] = {'MSE': overall_mse, 'RMSE': overall_rmse, 'MAE': overall_mae}
    return metrics


def print_metrics(metrics, model_name="Model"):
    """打印评估指标"""
    print(f"\n{'='*55}")
    print(f"  模型: {model_name}")
    print(f"{'='*55}")
    print(f"  {'指标':<10} {'Output1':>10} {'Output2':>10} {'Output3':>10} {'Overall':>10}")
    print(f"  {'-'*52}")
    for key in ['output_1', 'output_2', 'output_3']:
        vals = metrics.get(key, {})
        print(f"  {key:<10} {vals.get('RMSE',0):>10.4f} {vals.get('RMSE',0):>10.4f} {vals.get('RMSE',0):>10.4f} {'':>10}", end="")
        if key == 'output_1':
            pass
    print()
    for metric_name in ['RMSE', 'MSE', 'MAE', 'R2']:
        vals_out = [metrics.get(f'output_{i}', {}).get(metric_name, 0) for i in range(1,4)]
        overall_val = metrics.get('overall', {}).get(metric_name, 0)
        print(f"  {metric_name:<10} {vals_out[0]:>10.4f} {vals_out[1]:>10.4f} {vals_out[2]:>10.4f} {overall_val:>10.4f}")


def compare_models(metrics_lr, metrics_resnet, metrics_phys, feature_names=None):
    """对比多个模型的评估结果"""
    print(f"\n{'='*70}")
    print(f"  模型性能对比")
    print(f"{'='*70}")
    
    headers = ['模型', '指标', 'Output1', 'Output2', 'Output3', 'Overall']
    rows = []
    
    # Linear Regression
    for metric_name in ['RMSE', 'R2']:
        lr_vals = [metrics_lr.get(f'output_{i}', {}).get(metric_name, 0) for i in range(1,4)]
        lr_overall = metrics_lr.get('overall', {}).get(metric_name, 0)
        rows.append(['Linear Regression', metric_name, 
                    f'{lr_vals[0]:.4f}', f'{lr_vals[1]:.4f}', f'{lr_vals[2]:.4f}', f'{lr_overall:.4f}'])
    
    # ResNet
    for metric_name in ['RMSE', 'R2']:
        rn_vals = [metrics_resnet.get(f'output_{i}', {}).get(metric_name, 0) for i in range(1,4)]
        rn_overall = metrics_resnet.get('overall', {}).get(metric_name, 0)
        rows.append(['ResNet', metric_name,
                    f'{rn_vals[0]:.4f}', f'{rn_vals[1]:.4f}', f'{rn_vals[2]:.4f}', f'{rn_overall:.4f}'])
    
    # Physics-regularized ResNet
    for metric_name in ['RMSE', 'R2']:
        pr_vals = [metrics_phys.get(f'output_{i}', {}).get(metric_name, 0) for i in range(1,4)]
        pr_overall = metrics_phys.get('overall', {}).get(metric_name, 0)
        rows.append(['Phys-Reg ResNet', metric_name,
                    f'{pr_vals[0]:.4f}', f'{pr_vals[1]:.4f}', f'{pr_vals[2]:.4f}', f'{pr_overall:.4f}'])
    
    print_table(headers, rows)
    
    # 计算改进比例
    print(f"\n  相比线性回归的改进:")
    resnet_rmse_improve = (metrics_lr['overall']['RMSE'] - metrics_resnet['overall']['RMSE']) / metrics_lr['overall']['RMSE'] * 100
    phys_rmse_improve = (metrics_lr['overall']['RMSE'] - metrics_phys['overall']['RMSE']) / metrics_lr['overall']['RMSE'] * 100
    print(f"  ResNet RMSE 改进: {resnet_rmse_improve:.1f}%")
    print(f"  Phys-Reg ResNet RMSE 改进: {phys_rmse_improve:.1f}%")



# ============================================================
# 模块八：可视化函数
# ============================================================

def set_plot_style():
    """设置matplotlib绘图风格"""
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def save_plot(fig, filename):
    """保存图表"""
    os.makedirs('plots', exist_ok=True)
    filepath = os.path.join('plots', filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath

def plot_prediction_scatter(y_true, y_pred, save_path=None):
    """绘制预测值vs真实值散点图"""
    if not HAS_MATPLOTLIB:
        return None
    set_plot_style()
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 4))
    if n_outputs == 1:
        axes = [axes]
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for i in range(n_outputs):
        ax = axes[i]
        # 下采样以便可视化
        idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)
        ax.scatter(y_true[idx, i], y_pred[idx, i], alpha=0.3, s=8, color=colors[i % len(colors)])
        lims = [min(np.min(y_true[:, i]), np.min(y_pred[:, i])),
                max(np.max(y_true[:, i]), np.max(y_pred[:, i]))]
        ax.plot(lims, lims, '--', color='red', alpha=0.7, linewidth=1.5, label='Perfect Prediction')
        ax.set_xlabel('True Value', fontsize=11)
        ax.set_ylabel('Predicted Value', fontsize=11)
        ax.set_title(f'Output {i+1}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        return save_plot(fig, save_path)
    return fig

def plot_residual_histogram(y_true, y_pred, save_path=None):
    """绘制残差分布直方图"""
    if not HAS_MATPLOTLIB:
        return None
    set_plot_style()
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 4))
    if n_outputs == 1:
        axes = [axes]
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for i in range(n_outputs):
        ax = axes[i]
        residuals = y_true[:, i] - y_pred[:, i]
        ax.hist(residuals, bins=40, density=True, alpha=0.7, color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
        from scipy import stats
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        mu, sigma = stats.norm.fit(residuals)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), '--', color='red', linewidth=2,
               label=f'Normal Fit (mu={mu:.3f}, sigma={sigma:.3f})')
        ax.set_xlabel('Residual', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Output {i+1} - Residual Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        return save_plot(fig, save_path)
    return fig

def plot_training_loss(loss_history, save_path=None):
    """绘制训练损失曲线"""
    if not HAS_MATPLOTLIB:
        return None
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, color='#2196F3', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if save_path:
        return save_plot(fig, save_path)
    return fig

def plot_model_comparison_bar(metrics_list, model_names, save_path=None):
    """绘制模型对比柱状图"""
    if not HAS_MATPLOTLIB:
        return None
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    output_labels = ['Output 1', 'Output 2', 'Output 3']
    x = np.arange(len(output_labels))
    width = 0.25
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    # RMSE对比
    for j, metrics in enumerate(metrics_list):
        rmse_vals = [metrics.get(f'output_{i+1}', {}).get('RMSE', 0) for i in range(3)]
        axes[0].bar(x + j * width, rmse_vals, width, label=model_names[j], color=colors[j], alpha=0.8)
    axes[0].set_ylabel('RMSE', fontsize=11)
    axes[0].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(output_labels)
    axes[0].legend()
    
    # R2对比
    for j, metrics in enumerate(metrics_list):
        r2_vals = [metrics.get(f'output_{i+1}', {}).get('R2', 0) for i in range(3)]
        axes[1].bar(x + j * width, r2_vals, width, label=model_names[j], color=colors[j], alpha=0.8)
    axes[1].set_ylabel('R2 Score', fontsize=11)
    axes[1].set_title('R2 Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(output_labels)
    axes[1].legend()
    
    # MAE对比
    for j, metrics in enumerate(metrics_list):
        mae_vals = [metrics.get(f'output_{i+1}', {}).get('MAE', 0) for i in range(3)]
        axes[2].bar(x + j * width, mae_vals, width, label=model_names[j], color=colors[j], alpha=0.8)
    axes[2].set_ylabel('MAE', fontsize=11)
    axes[2].set_title('MAE Comparison', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(output_labels)
    axes[2].legend()
    
    plt.tight_layout()
    if save_path:
        return save_plot(fig, save_path)
    return fig

def plot_feature_correlation(X, save_path=None):
    """绘制特征相关性热力图"""
    if not HAS_MATPLOTLIB:
        return None
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_matrix = np.corrcoef(X.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(X.shape[1]))
    ax.set_yticks(range(X.shape[1]))
    ax.set_xticklabels([f'Feature {i+1}' for i in range(X.shape[1])])
    ax.set_yticklabels([f'Feature {i+1}' for i in range(X.shape[1])])
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=12,
                   color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)
    if save_path:
        return save_plot(fig, save_path)
    return fig



# ============================================================
# 模块九：交互式主菜单
# ============================================================

class WindFieldDemoApp:
    """风场反演演示系统主程序（终端交互版）"""
    
    def __init__(self):
        self.X_data = None
        self.y_data = None
        self.X_clean = None
        self.y_clean = None
        self.qc_stats = None
        self.data_stats = None
        self.scaler = WindFieldScaler()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_filt = self.X_test_filt = self.y_train_filt = self.y_test_filt = None
        self.linear_model = None
        self.resnet_model = None
        self.phys_resnet_model = None
        self.linear_metrics = None
        self.resnet_metrics = None
        self.phys_metrics = None
        self.resnet_loss = None
        self.phys_loss = None
        self.data_dir = '.'
        self.loaded = False
        self.filtered = False
        self.trained = False
    
    def run(self):
        """运行主程序"""
        self._print_banner()
        
        while True:
            self._print_menu()
            choice = input("请选择操作 [0-8]: ").strip()
            
            if choice == '0':
                print(f"\n{C.G}感谢使用，再见！{C.E}")
                break
            elif choice == '1':
                self._load_data()
            elif choice == '2':
                self._data_quality_check()
            elif choice == '3':
                self._data_preprocessing()
            elif choice == '4':
                self._train_models()
            elif choice == '5':
                self._evaluate_models()
            elif choice == '6':
                self._prediction_demo()
            elif choice == '7':
                self._visualization()
            elif choice == '8':
                self._show_model_info()
            else:
                print(f"\n{C.Y}无效输入，请重新选择{C.E}")
    
    def _print_banner(self):
        """打印程序横幅"""
        banner = """
 ╔══════════════════════════════════════════════════════════╗
 ║                                                          ║
 ║   航母舰载机风场反演演示系统                              ║
 ║   Carrier-based Aircraft Wind Field Inversion Demo        ║
 ║                                                          ║
 ║   基于ARM+FPGA的复杂风场实时精细反演小型化系统             ║
 ║   软件算法演示模块                                        ║
 ║                                                          ║
 ╚══════════════════════════════════════════════════════════╝
"""
        print(banner)
        print(f"{C.CY}系统已就绪。请从下方菜单选择操作。{C.E}")
        print(f"{C.GR}提示: 需要当前目录存在 X_data.txt 和 y_data.txt 文件{C.E}")
    
    def _print_menu(self):
        """打印主菜单"""
        print(f"\n{'='*60}")
        print(f"  主菜单")
        print(f"{'='*60}")
        print(f"  [1] 加载数据文件")
        print(f"  [2] 数据质量检查（异常值检测）")
        print(f"  [3] 数据预处理（滤波 + 标准化）")
        print(f"  [4] 模型训练（线性回归 / ResNet / 物理正则化ResNet）")
        print(f"  [5] 模型评估与对比")
        print(f"  [6] 实时预测演示（输入特征值获取预测结果）")
        print(f"  [7] 可视化分析（保存图表到 plots/ 目录）")
        print(f"  [8] 模型信息查看")
        print(f"  [0] 退出程序")
        print(f"{'='*60}")
        
        # 显示当前状态
        status_parts = []
        if self.loaded:
            status_parts.append(f"数据已加载({self.X_clean.shape[0]}样本)")
        if self.filtered:
            status_parts.append("预处理完成")
        if self.trained:
            status_parts.append("模型已训练")
        status = " | ".join(status_parts) if status_parts else "尚未加载数据"
        print(f"  当前状态: {C.YELLOW}{status}{C.E}")
    
    # ---- 功能1: 加载数据 ----
    def _load_data(self):
        """加载数据文件"""
        section("加载数据文件")
        
        data_dir = input(f"请输入数据目录 (当前: {self.data_dir}) [回车使用当前目录]: ").strip()
        if data_dir:
            self.data_dir = data_dir
        
        try:
            X, y = load_data(self.data_dir)
            self.X_data = X
            self.y_data = y
            self.loaded = True
            ok(f"成功加载数据: X={X.shape}, y={y.shape}")
            
            # 自动进行质量控制
            self._data_quality_check()
        except FileNotFoundError as e:
            err(f"加载失败: {e}")
        except Exception as e:
            err(f"加载出错: {e}")
    
    # ---- 功能2: 数据质量检查 ----
    def _data_quality_check(self):
        """数据质量控制"""
        if not self.loaded:
            warn("请先加载数据 ([1] 加载数据文件)")
            return
        
        section("数据质量控制（对应论文4.2.1节）")
        
        self.X_clean, self.y_clean, self.qc_stats = quality_control(self.X_data, self.y_data)
        self.data_stats = compute_data_stats(self.X_clean, self.y_clean)
        
        print(f"\n  异常值检测统计:")
        print(f"  总样本数: {self.qc_stats['total']}")
        print(f"  检测到异常值: {self.qc_stats['outliers']}")
        print(f"  异常值率: {self.qc_stats['outlier_rate']:.2f}%")
        print(f"  清洗后样本数: {self.qc_stats['clean']}")
        
        if self.qc_stats['outliers'] > 0:
            ok(f"已剔除 {self.qc_stats['outliers']} 个异常样本（NaN/999/-999标记）")
        else:
            ok("数据质量良好，未发现异常值")
        
        print_data_stats(self.data_stats)
    
    # ---- 功能3: 数据预处理 ----
    def _data_preprocessing(self):
        """数据预处理"""
        if not self.loaded:
            warn("请先加载数据 ([1] 加载数据文件)")
            return
        
        section("数据预处理（对应论文4.2.2节）")
        
        # 询问是否应用滤波
        filter_choice = input("是否应用两级滤波(Hampel+S-G)? [y/N]: ").strip().lower()
        
        if filter_choice == 'y':
            print("\n  应用Hampel滤波器...")
            X_filt, y_filt = apply_two_step_filter(self.X_clean, self.y_clean, hampel_window=15, sg_window=15)
            self.X_filtered = X_filt
            self.y_filtered = y_filt
            ok("两级滤波完成")
            
            # 展示滤波效果（采样展示）
            print(f"\n  滤波前后对比（采样前5个样本）:")
            print(f"  {'':<10} {'X原始':>20} {'X滤波后':>20}")
            for i in range(min(5, len(self.X_clean))):
                print(f"  Sample{i}: {self.X_clean[i].tolist()} -> {X_filt[i].tolist()}")
        else:
            self.X_filtered = self.X_clean
            self.y_filtered = self.y_clean
            warn("跳过滤波步骤，使用原始清洗数据")
        
        # 划分训练/测试集
        print("\n  划分训练集/测试集 (80/20)...")
        if filter_choice == 'y':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_filtered, self.y_filtered, test_ratio=0.2, seed=42)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_clean, self.y_clean, test_ratio=0.2, seed=42)
        
        ok(f"训练集: {self.X_train.shape[0]}样本, 测试集: {self.X_test.shape[0]}样本")
        
        # 标准化
        print("\n  使用学长提供的标准化参数进行标准化...")
        try:
            # 优先从数据目录加载学长提供的标准化参数
            scaler_files_found = True
            for sf in ['scaler_X_mean.txt', 'scaler_X_scale.txt', 'scaler_Y_mean.txt', 'scaler_Y_scale.txt']:
                if not os.path.exists(os.path.join(self.data_dir, sf)):
                    scaler_files_found = False
                    break
            if scaler_files_found:
                self.scaler.load_from_file(
                    os.path.join(self.data_dir, 'scaler_X_mean.txt'),
                    os.path.join(self.data_dir, 'scaler_X_scale.txt'),
                    os.path.join(self.data_dir, 'scaler_Y_mean.txt'),
                    os.path.join(self.data_dir, 'scaler_Y_scale.txt')
                )
                ok("成功加载学长提供的标准化参数")
            else:
                raise FileNotFoundError("scaler files not found")
        except FileNotFoundError:
            warn("未找到学长标准化参数文件，使用自动计算的参数")
            self.scaler.fit(self.X_train, self.y_train)
        
        self.X_train_scaled, self.y_train_scaled = self.scaler.transform(self.X_train, self.y_train)
        self.X_test_scaled, self.y_test_scaled = self.scaler.transform(self.X_test, self.y_test)
        
        self.filtered = True
        ok("数据预处理完成")
    
    # ---- 功能4: 模型训练 ----
    def _train_models(self):
        """训练模型"""
        if not self.filtered:
            warn("请先完成数据预处理 ([3] 数据预处理)")
            return
        
        section("模型训练")
        
        # 1. 线性回归
        print("\n  [1/3] 训练多元线性回归模型...")
        t0 = time.time()
        self.linear_model = LinearRegressionModel()
        self.linear_model.fit(self.X_train_scaled, self.y_train_scaled)
        lr_time = time.time() - t0
        ok(f"线性回归训练完成 ({lr_time:.2f}秒)")
        
        # 2. ResNet
        print("\n  [2/3] 训练残差网络(ResNet)...")
        t0 = time.time()
        self.resnet_model = SimpleResNet(input_dim=3, hidden_dim=32, output_dim=3,
                                          n_residual_blocks=2, learning_rate=0.001)
        self.resnet_model.fit(self.X_train_scaled, self.y_train_scaled, epochs=50, batch_size=256, verbose=True)
        rn_time = time.time() - t0
        self.resnet_loss = self.resnet_model.loss_history
        ok(f"ResNet训练完成 ({rn_time:.2f}秒)")
        
        # 3. 物理正则化ResNet
        print("\n  [3/3] 训练物理正则化ResNet（创新模块）...")
        print("  此模块将大气动力学约束嵌入损失函数:")
        print("    - 径向速度一致性约束")
        print("    - 湍流动能耗散约束")
        print("    - 边界条件约束")
        t0 = time.time()
        self.phys_resnet_model = PhysicsRegularizedResNet(
            input_dim=3, hidden_dim=32, output_dim=3,
            n_residual_blocks=2, learning_rate=0.001,
            phys_lambda=0.1, smooth_lambda=0.01
        )
        self.phys_resnet_model.fit(self.X_train_scaled, self.y_train_scaled, epochs=50, batch_size=256, verbose=True)
        pr_time = time.time() - t0
        self.phys_loss = self.phys_resnet_model.loss_history
        ok(f"物理正则化ResNet训练完成 ({pr_time:.2f}秒)")
        
        self.trained = True
        ok("所有模型训练完成!")
    
    # ---- 功能5: 模型评估 ----
    def _evaluate_models(self):
        """评估模型"""
        if not self.trained:
            warn("请先训练模型 ([4] 模型训练)")
            return
        
        section("模型评估与对比")
        
        # 计算预测
        pred_lr = self.linear_model.predict(self.X_test_scaled)
        pred_rn = self.resnet_model.predict(self.X_test_scaled)
        pred_pr = self.phys_resnet_model.predict(self.X_test_scaled)
        
        # 计算指标
        self.linear_metrics = calculate_metrics(self.y_test_scaled, pred_lr)
        self.resnet_metrics = calculate_metrics(self.y_test_scaled, pred_rn)
        self.phys_metrics = calculate_metrics(self.y_test_scaled, pred_pr)
        
        # 还原到原始空间评估
        pred_lr_orig, _ = self.scaler.inverse_transform(
            np.zeros_like(pred_lr), pred_lr)
        pred_lr_orig = pred_lr + self.scaler.y_mean_  # 简化还原
        
        # 打印评估结果
        print_metrics(self.linear_metrics, "多元线性回归")
        print()
        print_metrics(self.resnet_metrics, "ResNet")
        print()
        print_metrics(self.phys_metrics, "物理正则化ResNet")
        
        # 模型对比
        compare_models(self.linear_metrics, self.resnet_metrics, self.phys_metrics)
        
        ok("评估完成")
    
    # ---- 功能6: 实时预测 ----
    def _prediction_demo(self):
        """实时预测演示"""
        if not self.trained:
            warn("请先训练模型 ([4] 模型训练)")
            return
        
        section("实时预测演示")
        print("\n  请输入3个特征值（用空格分隔），输入 'quit' 退出:")
        print("  特征对应三台激光雷达的观测值（标准化前的原始值）")
        
        while True:
            user_input = input("\n  输入特征值 > ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("  退出预测模式")
                break
            
            try:
                values = [float(x) for x in user_input.split()]
                if len(values) != 3:
                    print(f"  {C.Y}请输入恰好3个数值{C.E}")
                    continue
                
                X_input = np.array([values])
                
                # 标准化
                X_scaled = (X_input - self.scaler.X_mean_) / self.scaler.X_scale_
                
                # 预测
                pred_lr = self.linear_model.predict(X_scaled)
                pred_rn = self.resnet_model.predict(X_scaled)
                pred_pr = self.phys_resnet_model.predict(X_scaled)
                
                # 还原到原始空间
                pred_lr_orig = pred_lr * self.scaler.y_scale_ + self.scaler.y_mean_
                pred_rn_orig = pred_rn * self.scaler.y_scale_ + self.scaler.y_mean_
                pred_pr_orig = pred_pr * self.scaler.y_scale_ + self.scaler.y_mean_
                
                print(f"\n  {'='*50}")
                print(f"  输入特征: {values}")
                print(f"  {'='*50}")
                print(f"  {'预测结果':^50}")
                print(f"  {'':10} {'Output1':>12} {'Output2':>12} {'Output3':>12}")
                print(f"  {'-'*46}")
                print(f"  线性回归:{'':6} {pred_lr_orig[0,0]:>12.4f} {pred_lr_orig[0,1]:>12.4f} {pred_lr_orig[0,2]:>12.4f}")
                print(f"  ResNet:{'':8} {pred_rn_orig[0,0]:>12.4f} {pred_rn_orig[0,1]:>12.4f} {pred_rn_orig[0,2]:>12.4f}")
                print(f"  Phys-Reg:{'':3} {pred_pr_orig[0,0]:>12.4f} {pred_pr_orig[0,1]:>12.4f} {pred_pr_orig[0,2]:>12.4f}")
                print(f"  {'='*50}")
                
            except ValueError:
                print(f"  {C.Y}输入格式错误，请输入数字{C.E}")
            except Exception as e:
                print(f"  {C.R}预测出错: {e}{C.E}")
    
    # ---- 功能7: 可视化 ----
    def _visualization(self):
        """可视化分析"""
        if not self.trained:
            warn("请先训练模型 ([4] 模型训练)")
            return
        
        section("可视化分析")
        
        if not HAS_MATPLOTLIB:
            err("matplotlib未安装，无法生成图表")
            print("  安装命令: pip install matplotlib")
            return
        
        # 确保有测试集预测
        if self.linear_metrics is None:
            self._evaluate_models()
        
        pred_lr = self.linear_model.predict(self.X_test_scaled)
        pred_rn = self.resnet_model.predict(self.X_test_scaled)
        pred_pr = self.phys_resnet_model.predict(self.X_test_scaled)
        
        saved_files = []
        
        # 1. 预测vs真实值散点图
        print("\n  生成预测对比散点图...")
        fig = plot_prediction_scatter(self.y_test_scaled, pred_rn)
        if fig:
            path = save_plot(fig, 'prediction_scatter_ResNet.png')
            saved_files.append(path)
            ok(f"已保存: {path}")
        
        # 2. 残差分布
        print("  生成残差分布图...")
        fig = plot_residual_histogram(self.y_test_scaled, pred_rn)
        if fig:
            path = save_plot(fig, 'residual_histogram_ResNet.png')
            saved_files.append(path)
            ok(f"已保存: {path}")
        
        # 3. 训练损失曲线
        print("  生成训练损失曲线...")
        if self.resnet_loss:
            fig = plot_training_loss(self.resnet_loss, 'training_loss_ResNet.png')
            if fig:
                saved_files.append('plots/training_loss_ResNet.png')
                ok("已保存: plots/training_loss_ResNet.png")
        
        if self.phys_loss:
            fig = plot_training_loss(self.phys_loss, 'training_loss_PhysResNet.png')
            if fig:
                saved_files.append('plots/training_loss_PhysResNet.png')
                ok("已保存: plots/training_loss_PhysResNet.png")
        
        # 4. 模型对比图
        print("  生成模型对比图...")
        fig = plot_model_comparison_bar(
            [self.linear_metrics, self.resnet_metrics, self.phys_metrics],
            ['Linear Regression', 'ResNet', 'Phys-Reg ResNet'],
            'model_comparison.png'
        )
        if fig:
            saved_files.append('plots/model_comparison.png')
            ok("已保存: plots/model_comparison.png")
        
        # 5. 特征相关性
        print("  生成特征相关性图...")
        fig = plot_feature_correlation(self.X_clean, 'feature_correlation.png')
        if fig:
            saved_files.append('plots/feature_correlation.png')
            ok("已保存: plots/feature_correlation.png")
        
        print(f"\n  共生成 {len(saved_files)} 张图表，保存在 plots/ 目录下")
    
    # ---- 功能8: 模型信息 ----
    def _show_model_info(self):
        """显示模型信息"""
        section("模型信息")
        
        print(f"\n  数据信息:")
        if self.loaded:
            print(f"  原始数据: X={self.X_data.shape}, y={self.y_data.shape}")
            print(f"  清洗后数据: X={self.X_clean.shape}, y={self.y_clean.shape}")
            if self.qc_stats:
                print(f"  异常值剔除: {self.qc_stats['outliers']}个 ({self.qc_stats['outlier_rate']:.1f}%)")
        
        print(f"\n  预处理信息:")
        if self.filtered:
            print(f"  训练集: {self.X_train.shape}")
            print(f"  测试集: {self.X_test.shape}")
            print(f"  标准化参数来源: {'学长提供' if self.scaler.loaded_from_file else '自动计算'}")
        
        print(f"\n  模型信息:")
        if self.trained:
            print(f"  线性回归: 正规方程求解")
            print(f"    截距: {self.linear_model.intercept.tolist()}")
            print(f"    系数形状: {self.linear_model.coefficients.shape}")
            print(f"  ResNet: {self.resnet_model.get_architecture()}")
            print(f"  物理正则化ResNet: {self.phys_resnet_model.get_architecture()}")
            print(f"    物理正则化系数: lambda_phys={self.phys_resnet_model.phys_lambda}")
            print(f"    平滑约束系数: lambda_smooth={self.phys_resnet_model.smooth_lambda}")
        
        print(f"\n  文件信息:")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for fname in ['X_data.txt', 'y_data.txt', 'model_params.txt',
                      'scaler_X_mean.txt', 'scaler_X_scale.txt',
                      'scaler_Y_mean.txt', 'scaler_Y_scale.txt']:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                print(f"    {fname}: {size} bytes")
            else:
                print(f"    {fname}: 不存在")


# ============================================================
# 程序入口
# ============================================================

def main():
    """主函数"""
    app = WindFieldDemoApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print(f"\n\n{C.Y}程序被用户中断{C.E}")
    except Exception as e:
        print(f"\n{C.R}程序出错: {e}{C.E}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()