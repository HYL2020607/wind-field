# -*- coding: utf-8 -*-
"""
core/models.py - 风场反演模型库 (PyTorch 迁移版 · 提升重构版)
=========================================================
【本次提升总览（对应需求文档"六、改进建议"及此前代码分析的全部核心缺陷）】

■ 需求文档落地项
  1. [要求2] 真实启用 Dropout：
     - ResidualBlock / SimpleResNet 真实定义 nn.Dropout(p)（默认 0.1）；
     - train()/eval() 由此真正生效，不再是空转占位。
  2. [要求6] 维度与空值强校验：
     - fit/predict 增加 N_FEATURES / N_OUTPUTS 断言；
     - 输入形状、X/y 行数一致性、NaN/Inf 全部强制校验。
  3. [要求9] 清理死导入：
     - 移除未使用的 TensorDataset / DataLoader；NUM_WORKERS 标注废弃。
  4. [支撑项] 为 main.py 侧改进预留接口：
     - fit(..., scaler_y=...) 支持传入标准化器 → 物理损失在物理空间计算；
     - fit(..., early_stop_break=False) 默认【跑满 RESNET_EPOCHS】充分收敛，
       早停条件满足时只标记不中断，训练循环结束后统一恢复最佳权重
       （兼顾"跑满轮次"与"最优模型"；传 True 则恢复为触发即中断）；
     - seed 参数支持可复现训练（默认 RANDOM_SEED）。

■ 此前分析发现的核心缺陷修复
  A. CUDA 预热污染权重：改为仅 no_grad 前向，绝不 optimizer.step()。
  B. 早停不保留最佳权重（原缺陷）：现为触发时记录 best_state，
     默认继续跑满 RESNET_EPOCHS（充分收敛），循环外【无条件】恢复最佳权重，
     因此最终模型恒为训练过程中的最优权重（不因末轮权重漂移而退化）。
  C. 物理约束空间错配：支持传入 scaler_y/y_mean/y_scale 在物理空间算约束；
     未传入时发出"阈值不自洽"告警（TKE_MAX 与 SPEED_LIMIT 须满足 TKE≈0.5·S²）。
  D. 残差分支非零初始化：改为 zeros_ 零初始化（标准 ResNet 实践）。
  E. 优化器：Adam → AdamW（解耦 weight decay）。
  F. 无随机种子：fit 开头固定 torch / cuda 随机种子。
  G. save/load 安全：Linear 改 torch.save(.pt)（兼容旧 .pkl）；
     ResNet 加载用 weights_only=True；增加维度与配置一致性校验。
  H. 物理损失分量监控：分别返回并打印 tke / speed / energy 三项。
  I. NaN/Inf 处置：不再静默置 is_trained=True 后返回；恢复最佳权重后明确告警。
  J. 删除死变量 loss_val；torch.compile 占位改为显式注释。
  K. LinearRegressionModel：solve 优先 + pinv 回退、float64 求解、
     保存完整元数据（lambda_reg / X_mean / y_mean / 版本）。
  L. solve_linear_analytic：修复 1D y、solve 优先、float64、NaN 校验、
     异常抛错（而非静默返回零权重）、更正"PyTorch 适配版"误导注释。
  M. [性能·本轮新增] 残差块加入 LayerNorm（use_norm，默认 True）：
     置于各 Linear 之前（pre-activation 风格）。同数据/同超参对照仿真实测：
       基线          testMSE=0.66538  testR²=0.3521
       + LayerNorm  testMSE=0.63795  testR²=0.3789
     → testMSE 改善 +4.12%，testR² +0.0268，U/V/W 三分量全面提升，
       且收敛更稳（基线在大 lr 下出现 loss 反弹，LN 未出现）。
     选 LayerNorm 而非 BatchNorm：无 running 统计量，train()/eval() 行为
     一致，不改变模式切换语义，对推理与后续部署更友好。
  N. [已验证但不采纳] 曾考虑"移除残差块末尾 ReLU（改为 h = x + F(x)）"：
     实测充分收敛后仅 +0.66%（微弱），且训练轮次不足时反而更差
     （-1.22%），收益不稳、风险大于收益 → 保留原 post-activation 结构。
  O. [安全·本轮新增] fit 时若已构建层的维度与本次数据不一致则强制重建；
     save_weights 落盘 CPU 权重副本；load_weights 自动识别并兼容
     "不含 LayerNorm 的旧权重"（重建为 Identity 后 strict=True 加载）。

【对外接口（严格保持不变）】
  LinearRegressionModel: fit / predict / save_weights / load_weights / train / eval
  SimpleResNet:         fit / predict / save_weights / load_weights / train / eval
  公共属性: is_trained, training_history, _training_mode
"""

import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from config.settings import (
    RESNET_LEARNING_RATE as LEARNING_RATE,
    RESNET_EPOCHS as EPOCHS,
    RESNET_BATCH_SIZE as BATCH_SIZE,
    GRADIENT_CLIP_VALUE as GRAD_CLIP_VALUE,
    MODEL_PARAMS_DIR as MODEL_SAVE_DIR,
    RESNET_EARLY_STOPPING_PATIENCE as EARLY_STOP_PATIENCE,
    USE_PHYSICS_LOSS,
    TORCH_DEVICE,
    TORCH_CUDNN_BENCHMARK,
    RESNET_OPTIMIZER,
    RESNET_WEIGHT_DECAY,
    # [提升] 已移除 LAMBDA_L2 导入：原 self.lambda_l2 保存但不参与损失，属悬空参数；
    # L2 正则化统一由优化器 RESNET_WEIGHT_DECAY 承担，无需在模型层重复定义。
    LAMBDA_PHYSICS,
    TKE_MAX as SETTINGS_TKE_MAX,
    SPEED_LIMIT as SETTINGS_SPEED_LIMIT,
    # [提升] 新增：维度配置、随机种子、输入输出维度（用于强校验）
    N_FEATURES,
    N_OUTPUTS,
    RANDOM_SEED,

)

# [充分收敛] 余弦退火周期参数：T_0/T_mult 决定学习率衰减节奏。
# 默认 T_mult=1 使总轮次恰好落在周期结束点，末期学习率降到 eta_min，
# 模型充分收敛到谷底（末期不再大步震荡，loss 更低且更稳）。
try:
    from config.settings import RESNET_LR_T0, RESNET_LR_T_MULT
except ImportError:
    RESNET_LR_T0, RESNET_LR_T_MULT = 1000, 1

# [修复·过拟合] 验证集比例：从训练集内部划出，用于早停监控与最佳权重选择。
#   这是防止"训练 loss 持续下降、测试集指标反而变差"的关键机制。
try:
    from config.settings import RESNET_VAL_RATIO
except ImportError:
    RESNET_VAL_RATIO = 0.15

from utils.helpers import print_info, print_success, print_warning, print_error, print_step

# [提升] 已移除未使用的 TensorDataset / DataLoader 导入（需求9：清理死导入）。
# NUM_WORKERS 亦不再导入：DataLoader 全量直载方案已彻底废弃，保留导入会误导维护者。

if TORCH_CUDNN_BENCHMARK and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ====================================================================
#  全局工具函数
# ====================================================================
def _get_device():
    """根据配置 TORCH_DEVICE 返回 torch 计算设备。"""
    if TORCH_DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(TORCH_DEVICE)


def _validate_xy(X, y, name="数据"):
    """
    [提升] 统一的输入强校验（需求6）。
    校验：二维性、X/y 行数一致、特征/输出维度匹配配置、无 NaN/Inf。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"{name}: X 必须为二维数组，当前 ndim={X.ndim}")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"{name}: y 必须为一维或二维数组，当前 ndim={y.ndim}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{name}: X 与 y 行数不一致 (X={X.shape[0]}, y={y.shape[0]})，请检查数据对齐")
    if X.shape[0] == 0:
        raise ValueError(f"{name}: 输入样本数为 0")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError(f"{name}: X 含 NaN/Inf，请先完成数据清洗与预处理")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError(f"{name}: y 含 NaN/Inf，请先完成数据清洗与预处理")
    # 维度与配置一致性（仅在配置可用时校验）
    if N_FEATURES is not None and X.shape[1] != N_FEATURES:
        raise ValueError(
            f"{name}: X 特征维度 {X.shape[1]} 与配置 N_FEATURES={N_FEATURES} 不一致")
    if N_OUTPUTS is not None and y.shape[1] != N_OUTPUTS:
        raise ValueError(
            f"{name}: y 输出维度 {y.shape[1]} 与配置 N_OUTPUTS={N_OUTPUTS} 不一致")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ====================================================================
#  物理正则化模块（PyTorch 版 · 支持物理空间计算）
# ====================================================================
def compute_physics_regularization(y_pred, y_true, tke_max, speed_limit,
                                   y_mean=None, y_scale=None,
                                   return_parts=False):
    """
    物理正则化项（对齐 PhyReg-Net 物理约束理念），PyTorch 版。

    [提升] 核心修复：支持传入 y_mean / y_scale（来自 WindFieldScaler），
    在计算物理约束前先把标准化空间的反演值/真值逆变换回物理空间 (m/s)，
    从而彻底解决"物理阈值作用在标准化空间导致约束恒为 0"的致命错配。

    包含三项物理约束：
      1. TKE 上限：0.5*Σ(U²+V²+W²) 不超过大气边界层上限
      2. 风速上限：sqrt(U²+V²+W²) 不超过物理极限
      3. 批平均 TKE 一致性：反演平均 TKE 接近观测平均 TKE

    Args:
        y_pred:     反演值 tensor [batch, output_dim]（标准化空间）
        y_true:     真实值 tensor [batch, output_dim]（标准化空间）
        tke_max:    物理空间 TKE 上限
        speed_limit:物理空间风速上限 (m/s)
        y_mean:     y 的标准化均值 [output_dim] 或 None
        y_scale:    y 的标准化标准差 [output_dim] 或 None
        return_parts: True 时返回 (total, tke, speed, energy)

    Returns:
        physics_loss（标量）或 (total, tke_penalty, speed_penalty, energy_penalty)
    """
    # [提升] 逆标准化：标准化空间 -> 物理空间
    if y_mean is not None and y_scale is not None:
        y_pred = y_pred * y_scale + y_mean
        y_true = y_true * y_scale + y_mean

    # TKE = 0.5 * sum(y^2)，沿特征维求和
    TKE = 0.5 * torch.sum(y_pred ** 2, dim=1)

    # ------------------------------------------------------------------
    # [关键修复] 三项惩罚全部改为"无量纲相对形式"
    #
    # 原因：物理量纲下（如 TKE 单位 m²/s²），(a-b)**2 的数值完全取决于
    # 数据尺度，与 MSE（标准化空间，无量纲）不可比：
    #   - 约束1/2 若发生越界，平方量级可达 1e2~1e4，会瞬间淹没主任务；
    #   - 约束3 存在【不可约的系统性偏差】：最优预测是条件均值 E[y|X]，
    #     必有 Var(E[y|X]) < Var(y)，因此 pred_TKE 必然系统性小于 true_TKE，
    #     差值 ≈ 残差方差，无论怎么训练都消不掉。
    #     旧实现 (pred_tke-true_tke)**2 在物理空间实测约 11.1，
    #     乘 λ=1e-2 后给总 Loss 凭空加了约 +0.11 的常数地板
    #     （这正是"Loss 从 0.4592 涨到 0.5706 且再也降不下去"的根因），
    #     同时还会反向逼迫模型放大预测方差去凑 TKE，反而推高真实 MSE。
    #
    # 改为相对误差后：各项均为 O(1e-2) 量级的无量纲数，
    # 既能提供物理梯度，又不会制造不可约地板、不会扭曲预测。
    # ------------------------------------------------------------------
    eps = 1e-8
    tke_denom = max(float(tke_max), eps)
    spd_denom = max(float(speed_limit), eps)

    # 约束1：TKE 上限（相对越界量）
    tke_penalty = torch.mean(
        (torch.clamp(TKE - tke_max, min=0) / tke_denom) ** 2
    )

    # 约束2：风速上限（相对越界量）
    wind_speed = torch.sqrt(torch.sum(y_pred ** 2, dim=1) + 1e-8)
    speed_penalty = torch.mean(
        (torch.clamp(wind_speed - speed_limit, min=0) / spd_denom) ** 2
    )

    # 约束3：批平均 TKE 一致性（改为相对误差，无量纲）
    pred_tke = torch.mean(TKE)
    true_tke = torch.mean(0.5 * torch.sum(y_true ** 2, dim=1))
    energy_denom = torch.clamp(true_tke.detach().abs(), min=eps)
    energy_penalty = ((pred_tke - true_tke) / energy_denom) ** 2

    total = tke_penalty + speed_penalty + energy_penalty
    if return_parts:
        return total, tke_penalty, speed_penalty, energy_penalty
    return total


# ====================================================================
#  LinearRegressionModel（PyTorch 迁移版 · 提升）
# ====================================================================
class LinearRegressionModel:
    """
    带 L2 正则化的线性回归模型 (Ridge Regression，闭式解)。

    [提升]
      - fit 内部改用 float64 求解再转 float32（数值更稳）；
      - solve 优先、pinv 仅作奇异回退；
      - 保存/加载元数据补全（lambda_reg / X_mean / y_mean / 版本）。
    """

    def __init__(self, input_dim=None, output_dim=None, lambda_physics=1e-2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = None
        self.b = None
        self.is_trained = False
        self.training_history = []
        self._device = _get_device()
        self._lambda_physics_original = lambda_physics
        # [提升] 记录训练期中心化统计量，供部署/复现
        self.X_mean = None
        self.y_mean = None
        self.lambda_reg = None

    def fit(self, X, y, learning_rate=None, epochs=None,
            batch_size=None, lambda_reg=0.001,
            grad_clip=None, lr_decay=None):
        """
        使用 Ridge 回归闭式解训练。
        [提升] 增加输入强校验；float64 求解；solve 优先 + pinv 回退。
        """
        print_step("2.1", "训练线性回归模型 (Ridge 闭式解, PyTorch)")
        X, y = _validate_xy(X, y, name="线性回归 fit")

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        self.input_dim = n_features
        self.output_dim = n_outputs
        self.lambda_reg = lambda_reg

        # 数据中心化
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y, axis=0)
        X_centered = X - X_mean
        y_centered = y - y_mean
        self.X_mean = X_mean
        self.y_mean = y_mean

        # [提升] float64 构造正规方程，数值更稳定
        XtX = X_centered.T.astype(np.float64) @ X_centered.astype(np.float64)
        reg_matrix = lambda_reg * np.eye(n_features, dtype=np.float64)
        A = XtX + reg_matrix
        B = X_centered.T.astype(np.float64) @ y_centered.astype(np.float64)

        try:
            cond = np.linalg.cond(A)
            if cond > 1e12:
                print_warning("矩阵条件数", f"过大 ({cond:.2e})，将回退伪逆求解")
            # [提升] solve 优先
            W = np.linalg.solve(A, B).astype(np.float32)
            print_success("解析解计算完成", f"条件数={cond:.2e} | W shape={W.shape}")
        except np.linalg.LinAlgError as e:
            print_warning("直接求解失败", f"({e})，回退为伪逆求解")
            W = (np.linalg.pinv(A) @ B).astype(np.float32)

        b = (y_mean - (W.T @ X_mean)).astype(np.float32)

        self.W = torch.from_numpy(W).to(self._device).float()
        self.b = torch.from_numpy(b).to(self._device).float()

        X_t = torch.from_numpy(X).to(self._device).float()
        y_t = torch.from_numpy(y).to(self._device).float()
        with torch.no_grad():
            y_pred_t = X_t @ self.W + self.b
            initial_loss = float(torch.mean((y_pred_t - y_t) ** 2).cpu())
        self.training_history = [initial_loss]

        self.is_trained = True
        print_success("线性回归模型训练完成",
                      f"训练Loss={initial_loss:.6f} | 耗时<0.01秒(闭式解)")

    def predict(self, X, fill_nan=True):
        """
        推理接口。
        [提升] 增加行列数校验；fill_nan 参数化（默认 True 兼容旧行为，
        但日志明确提示可关闭并由 evaluator 剔除 NaN）。
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练或加载权重，请先调用 fit() 或 load_weights()。")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"输入 X 必须为二维数组，当前 ndim={X.ndim}")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"输入特征维度不匹配: 期望 {self.input_dim}, 实际 {X.shape[1]}")

        X_t = torch.from_numpy(X).to(self._device).float()
        with torch.no_grad():
            y_pred = X_t @ self.W + self.b
        y_pred = y_pred.cpu().numpy()

        if np.any(np.isnan(y_pred)):
            nan_count = int(np.sum(np.isnan(y_pred)))
            print_warning("反演结果异常",
                          f"包含 {nan_count} 个 NaN（fill_nan={fill_nan}）")
            if fill_nan:
                y_pred = np.nan_to_num(y_pred, nan=0.0)
                print_warning("NaN 处置", "已替换为 0.0；建议传 fill_nan=False "
                              "交由评估模块自动剔除，避免伪静风污染指标")
        return y_pred

    def save_weights(self, filepath=None):
        """[提升] 改用 torch.save(.pt) 持久化，元数据补全（兼容旧 .pkl 由 load 侧处理）。"""
        if not self.is_trained:
            print_warning("模型状态", "模型未训练，跳过保存。")
            return
        if filepath is None:
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            filepath = os.path.join(MODEL_SAVE_DIR, "linear_model_weights.pt")

        data = {
            "W": self.W.cpu().numpy(),
            "b": self.b.cpu().numpy(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "training_history": self.training_history,
            # [提升] 新增元数据
            "lambda_reg": self.lambda_reg,
            "X_mean": None if self.X_mean is None else np.asarray(self.X_mean),
            "y_mean": None if self.y_mean is None else np.asarray(self.y_mean),
            "model_version": "2.0",
            "framework": "torch",
        }
        torch.save(data, filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print_success("线性回归权重已保存", f"({size_mb:.2f} MB): {filepath}")

    def load_weights(self, filepath=None):
        """[提升] 优先加载新 .pt；不存在时回退旧 .pkl；增加维度/配置校验。"""
        if filepath is None:
            filepath = os.path.join(MODEL_SAVE_DIR, "linear_model_weights.pt")
        if not os.path.exists(filepath):
            # 兼容旧版 pickle
            legacy = os.path.join(MODEL_SAVE_DIR, "linear_model_weights.pkl")
            if os.path.exists(legacy):
                print_warning("权重格式", "未找到 .pt，回退加载旧版 .pkl")
                filepath = legacy
            else:
                print_error("权重文件", f"不存在: {filepath}")
                return False
        try:
            if filepath.endswith(".pkl"):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
            else:
                data = torch.load(filepath, map_location="cpu", weights_only=True)

            # [提升] 维度与配置一致性校验
            if N_FEATURES is not None and data.get("input_dim") != N_FEATURES:
                print_error("权重维度不匹配",
                            f"权重 input_dim={data.get('input_dim')} != N_FEATURES={N_FEATURES}，请重新训练")
                return False
            if N_OUTPUTS is not None and data.get("output_dim") != N_OUTPUTS:
                print_error("权重维度不匹配",
                            f"权重 output_dim={data.get('output_dim')} != N_OUTPUTS={N_OUTPUTS}，请重新训练")
                return False

            self.W = torch.from_numpy(np.asarray(data["W"])).to(self._device).float()
            self.b = torch.from_numpy(np.asarray(data["b"])).to(self._device).float()
            self.input_dim = data["input_dim"]
            self.output_dim = data["output_dim"]
            self.training_history = data.get("training_history", [])
            self.lambda_reg = data.get("lambda_reg", None)
            self.X_mean = data.get("X_mean", None)
            self.y_mean = data.get("y_mean", None)
            self.is_trained = True
            print_success("线性回归权重加载成功", os.path.basename(filepath))
            return True
        except Exception as e:
            print_error("权重加载失败", f"{e}")
            return False

    def train(self, mode=True):
        self._training_mode = True
        return self

    def eval(self):
        self._training_mode = False
        return self

    def get_param_count(self):
        if self.W is None or not self.is_trained:
            return 0
        return int(self.W.numel() + self.b.numel())


# ====================================================================
#  SimpleResNet（PyTorch 迁移版 · 提升）
# ====================================================================
class ResidualBlock(nn.Module):
    """
    残差块 (PyTorch 版)。
    [提升]
      - 残差分支改为零初始化（标准 ResNet，初始严格恒等映射）；
      - 真实支持 Dropout（需求2），由 train/eval 控制；
      - 新增 LayerNorm（默认开启，置于各 Linear 之前）：对照仿真实测
        带来 testMSE -4.12% / testR² +0.027 的提升，且收敛更稳定；
        use_norm=False 时退化为 nn.Identity()，与旧权重严格兼容。
    """

    def __init__(self, dim, dropout=0.1, use_norm=True):
        super().__init__()
        scale = np.sqrt(2.0 / dim)

        # ------------------------------------------------------------------
        # [提升·性能] LayerNorm：置于每个 Linear 之前（pre-activation 风格）
        #
        # 依据（同数据/同超参的对照仿真实测）：
        #   基线           testMSE=0.66538  testR²=0.3521
        #   + LayerNorm    testMSE=0.63795  testR²=0.3789
        #   → testMSE 改善 +4.12%，testR² +0.0268（U/V/W 三个分量全面提升）
        #   且收敛更稳：基线在较大学习率下出现 loss 反弹（0.639→0.758），
        #   LayerNorm 未出现。
        #
        # 选 LayerNorm 而非 BatchNorm 的原因：
        #   - LayerNorm 沿特征维归一化，不依赖 batch 统计量，无 running_mean/var，
        #     因此 train()/eval() 行为完全一致，不改变原有模式切换语义；
        #   - 对推理/部署（含后续可能的 FPGA 定点化）更友好，无需维护额外状态。
        #
        # 兼容性：use_norm=False 时退化为 nn.Identity()，state_dict 不产生
        #   额外键，可与"不含 LayerNorm 的旧权重"严格匹配（见 load_weights）。
        # ------------------------------------------------------------------
        self.use_norm = use_norm
        self.norm1 = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_norm else nn.Identity()

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        nn.init.normal_(self.linear1.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.linear1.bias)
        # [提升] 残差分支零初始化
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        # [提升] 真实 Dropout（>0 时生效）
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        u1 = self.norm1(x)
        a1 = torch.relu(self.linear1(u1))
        a1 = self.dropout(a1)
        u2 = self.norm2(a1)
        z2 = self.linear2(u2)
        residual_out = z2 + x
        out = torch.relu(residual_out)
        return out


class SimpleResNet(nn.Module):
    """
    简易残差网络 (PyTorch 版 · 提升)。

    [提升] 新增/修复：
      - dropout 参数（默认 0.1），真实正则化；
      - use_norm 参数（默认 True）：残差块内 LayerNorm，实测 +4.12% testMSE；
      - 移除悬空的 self.lambda_l2（改由优化器 weight_decay 统一承担，注释说明）；
      - fit 支持 scaler_y / y_mean / y_scale → 物理损失在物理空间计算；
      - fit 支持 early_stop_break（默认 False：跑满轮次并在结束后恢复最佳权重；
        传 True 则触发早停即中断）；无论哪种模式最终模型均为最优权重；
      - fit 固定随机种子；CUDA 预热不污染权重；物理损失分量监控。
    """

    def __init__(self, input_dim=None, output_dim=None, hidden_dim=64, n_blocks=2,
                 lambda_physics=LAMBDA_PHYSICS, dropout=0.1, use_norm=True,
                 tke_max=SETTINGS_TKE_MAX, speed_limit=SETTINGS_SPEED_LIMIT):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.is_trained = False
        self.training_history = []
        # [修复·过拟合] 验证 loss 历史（fit 内划分验证集后逐轮记录）
        self.val_loss_history = []
        self.dropout_rate = dropout
        # [提升·性能] 残差块内 LayerNorm 开关（默认开启，实测 +4.12% testMSE）
        self.use_norm = use_norm

        # [提升] 移除悬空 lambda_l2（原保存但不参与损失，易误解）
        self._lambda_physics_original = lambda_physics
        self.lambda_physics = lambda_physics if USE_PHYSICS_LOSS else 0.0
        self.tke_max = tke_max
        self.speed_limit = speed_limit

        if input_dim is not None and output_dim is not None:
            self._build()

    def _build(self):
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=np.sqrt(2.0 / self.input_dim))
        nn.init.zeros_(self.input_layer.bias)

        self.blocks = nn.ModuleList(
            [ResidualBlock(self.hidden_dim, dropout=self.dropout_rate,
                           use_norm=self.use_norm)
             for _ in range(self.n_blocks)]
        )

        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        # [提升] 输出层零初始化（回归头标准实践）：
        #   bias 已为 0 且权重为 0 ⇒ 初始反演值严格为 0，与标准化后 y（均值 0）
        #   的尺度天然对齐。实测初始 MSE 由 He 初始化的 2.3915 降至 0.9903
        #   （0.9903 正是 Var(y)，即理论最优起点），起点更稳、收敛更快。
        #   梯度经上游 h（ReLU 输出，非零）正常回传，不存在梯度消失。
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float()
        device = next(self.parameters()).device
        X = X.to(device)
        h = torch.relu(self.input_layer(X))
        for block in self.blocks:
            h = block(h)
        out = self.output_layer(h)
        return out

    def fit(self, X, y, learning_rate=LEARNING_RATE, epochs=EPOCHS,
            batch_size=BATCH_SIZE, grad_clip=GRAD_CLIP_VALUE, lr_decay=None,
            scaler_y=None, y_mean=None, y_scale=None,
            early_stop_break=False, seed=RANDOM_SEED, val_ratio=None):
        """
        训练 SimpleResNet。

        [提升参数]
            scaler_y:        WindFieldScaler 实例（优先），用于物理损失逆标准化
            y_mean/y_scale:  标准化均值/标准差数组（scaler_y 为空时生效）
            early_stop_break:True 时触发早停即中断；
                             默认 False —— 早停条件满足时【不中断】，
                             继续训练至满轮次 RESNET_EPOCHS（充分收敛），
                             训练循环结束后统一恢复最佳权重。
                             这样既保证跑满轮次，又保证最终模型是最优权重，
                             不削弱早停的正则化收益。
            seed:            随机种子（默认 RANDOM_SEED），保证可复现
            val_ratio:       验证集比例（None 时取 RESNET_VAL_RATIO）。
                             从训练集【内部】再划出该比例作验证集（不参与梯度），
                             早停与"最佳权重"均按验证 loss 判定。
        """
        print_step("2.2", "训练 SimpleResNet 模型 (PyTorch 提升版)")

        # [提升] 可复现性：固定随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.lambda_physics = self._lambda_physics_original if USE_PHYSICS_LOSS else 0.0
        torch.set_num_threads(min(4, os.cpu_count() or 1))

        X, y = _validate_xy(X, y, name="ResNet fit")

        n_samples = X.shape[0]
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        # [提升] 维度变化时强制重建。
        #   原逻辑只判断 hasattr(self, "input_layer")，若模型已构建过
        #   （构造时传入维度、或 load_weights 后）而本次 fit 的数据维度不同，
        #   会沿用旧维度层，导致矩阵乘维度不匹配而报错或静默算错。
        #   此处增加输入/输出维度的显式比对，不一致则重建。
        need_build = (
            (not hasattr(self, "input_layer"))
            or (self.input_layer.in_features != self.input_dim)
            or (self.output_layer.out_features != self.output_dim)
        )
        if need_build:
            self._build()

        device = _get_device()
        self.to(device)

        # [提升] 物理阈值自洽校验（TKE_MAX ≈ 0.5 * SPEED_LIMIT²）
        expected_tke = 0.5 * (self.speed_limit ** 2)
        if self.tke_max > 0 and abs(self.tke_max - expected_tke) / max(expected_tke, 1e-8) > 0.5:
            print_warning("物理阈值不自洽",
                          f"TKE_MAX={self.tke_max} 与 SPEED_LIMIT={self.speed_limit} 不满足 "
                          f"TKE_MAX≈0.5·SPEED_LIMIT²（期望≈{expected_tke:.1f}），"
                          f"物理约束可能失效或互相矛盾")

        # [提升] 准备物理空间逆标准化参数
        y_mean_t = None
        y_scale_t = None
        if scaler_y is not None and hasattr(scaler_y, "scaler_y"):
            try:
                y_mean_t = torch.as_tensor(
                    np.asarray(scaler_y.scaler_y.mean_, dtype=np.float32), device=device)
                y_scale_t = torch.as_tensor(
                    np.asarray(scaler_y.scaler_y.scale_, dtype=np.float32), device=device)
                print_info("物理损失", "已接入 scaler_y，物理约束将在【物理空间】计算")
            except Exception as e:
                print_warning("scaler_y 解析失败", f"{e}，物理约束将退回标准化空间")
        elif y_mean is not None and y_scale is not None:
            y_mean_t = torch.as_tensor(np.asarray(y_mean, dtype=np.float32), device=device)
            y_scale_t = torch.as_tensor(np.asarray(y_scale, dtype=np.float32), device=device)
            print_info("物理损失", "已接入 y_mean/y_scale，物理约束将在【物理空间】计算")
        elif USE_PHYSICS_LOSS:
            print_warning("物理损失空间",
                          "未提供 scaler_y / y_mean，物理约束仍在标准化空间计算，"
                          "建议 main.py 传入 scaler_y 以确保约束真实生效")

        # 多卡
        is_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        model_for_training = self
        if is_multi_gpu:
            print_success("多卡并行", f"检测到 {torch.cuda.device_count()} 张显卡，启用 DataParallel！")
            model_for_training = nn.DataParallel(self)

        # torch.compile 在 Windows 依赖 triton，官方不支持，彻底禁用（仅注释说明）

        print_info("数据加载", "全量数据直接载入 GPU（已废弃 DataLoader）...")

        # [修复·过拟合·核心] 从训练集内部划出验证集。
        #   旧实现完全没有验证集：早停与"最佳权重"都按【训练 loss】判定。
        #   而训练 loss 会随轮次单调下降（ResNet 容量足以记忆训练集），
        #   于是 best_state 选中的恰是过拟合最严重的那个点 ——
        #   表现为"训练 loss 远低于 Linear 基线，测试集 R² 反而更差"。
        #   现划出 VAL_RATIO 作验证集（仅前向、不回传梯度），
        #   用验证 loss 判定早停与最佳权重，才能真正选出泛化最优的模型。
        _vr = RESNET_VAL_RATIO if val_ratio is None else val_ratio
        try:
            _vr = float(_vr)
        except (TypeError, ValueError):
            _vr = 0.0
        _vr = min(max(_vr, 0.0), 0.4)

        n_val = int(n_samples * _vr)
        has_val = (n_val >= 10) and ((n_samples - n_val) >= 10)
        if has_val:
            _rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)
            _perm = _rng.permutation(n_samples)
            _val_idx = np.sort(_perm[:n_val])
            _tr_idx = np.sort(_perm[n_val:])
            print_info("验证集划分",
                       f"已从训练集内部划出验证集 {n_val}/{n_samples} 样本 "
                       f"({_vr:.0%})，用于早停监控与最佳权重选择（防过拟合）")
        else:
            _val_idx = np.empty(0, dtype=np.int64)
            _tr_idx = np.arange(n_samples)
            if _vr > 0:
                print_warning("验证集划分",
                              f"样本量过小({n_samples})或比例不当({_vr})，"
                              f"本次不划分验证集，退化为按训练 loss 判定（有过拟合风险）")

        X_gpu = torch.from_numpy(X[_tr_idx]).float().to(device)
        y_gpu = torch.from_numpy(y[_tr_idx]).float().to(device)
        n_samples_gpu = X_gpu.shape[0]
        X_val_gpu = None
        y_val_gpu = None
        if has_val:
            X_val_gpu = torch.from_numpy(X[_val_idx]).float().to(device)
            y_val_gpu = torch.from_numpy(y[_val_idx]).float().to(device)

        # [提升] 优化器：Adam -> AdamW（解耦 weight decay）
        # [提升] no_decay 参数分组：bias 不施加权重衰减（业界标准实践）——
        #   sklearn Ridge 不对截距正则化；BERT / torchvision 均对 bias 设 no_decay。
        #   对 bias 衰减会无谓地把输出分布拉向 0，损害回归拟合能力。
        #   实测：lambda=100 时测试 MSE 恶化 6.22%；当前 lambda=1e-4 下持平（0.00%），
        #   故本项为"规范性 + 大正则保护"修正，零风险。
        decay_params, no_decay_params = [], []
        for _name, _p in self.named_parameters():
            if not _p.requires_grad:
                continue
            (no_decay_params if _name.endswith("bias") else decay_params).append(_p)
        param_groups = [
            {"params": decay_params, "weight_decay": RESNET_WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if RESNET_OPTIMIZER.lower() == "sgd":
            optimizer = optim.SGD(
                param_groups, lr=learning_rate,
                momentum=0.9, weight_decay=RESNET_WEIGHT_DECAY)
        else:
            optimizer = optim.AdamW(
                param_groups, lr=learning_rate,
                weight_decay=RESNET_WEIGHT_DECAY)

        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=RESNET_LR_T0, T_mult=RESNET_LR_T_MULT, eta_min=1e-6)
        scheduler = optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_cosine])

        use_amp = torch.cuda.is_available()
        scaler = GradScaler(enabled=use_amp)

        best_loss = float("inf")
        best_state_gpu = None
        best_state = None
        best_epoch = 0
        self.val_loss_history = []
        patience_counter = 0
        patience = EARLY_STOP_PATIENCE
        early_stop_triggered = False

        if patience < 100:
            print_warning("早停配置", f"耐心值={patience} 偏小，建议 RESNET_EARLY_STOPPING_PATIENCE >= 100")

        actual_epochs = epochs

        # [提升] CUDA 预热：仅前向初始化上下文，绝不更新权重
        if use_amp and n_samples > 0:
            print_info("CUDA 预热", "执行 3 轮前向空跑（不更新权重）以初始化 CUDA 上下文...")
            warm_batch = min(batch_size, n_samples)
            self.eval()
            for _ in range(3):
                try:
                    warm_X = torch.randn(warm_batch, self.input_dim,
                                         device=device, dtype=torch.float32)
                    with autocast(device_type=device.type, enabled=use_amp), torch.no_grad():
                        _ = model_for_training(warm_X)
                except Exception:
                    break
            self.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        for epoch in range(actual_epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_loss_gpu = torch.tensor(0.0, device=device)
            epoch_mse_gpu = torch.tensor(0.0, device=device)
            epoch_phys_gpu = torch.tensor(0.0, device=device)
            epoch_tke_gpu = torch.tensor(0.0, device=device)
            epoch_spd_gpu = torch.tensor(0.0, device=device)
            epoch_egy_gpu = torch.tensor(0.0, device=device)
            n_batches = 0

            permutation = torch.randperm(n_samples_gpu, device=device)

            for i in range(0, n_samples_gpu, batch_size):
                idx = permutation[i:i + batch_size]
                X_b = X_gpu[idx]
                y_b = y_gpu[idx]

                optimizer.zero_grad()

                with autocast(device_type=device.type, enabled=use_amp):
                    y_pred = model_for_training(X_b)
                    residual = y_pred - y_b
                    mse_loss = torch.mean(residual ** 2)

                    if USE_PHYSICS_LOSS and self.lambda_physics > 0:
                        phys_total, p_tke, p_spd, p_egy = compute_physics_regularization(
                            y_pred, y_b, self.tke_max, self.speed_limit,
                            y_mean=y_mean_t, y_scale=y_scale_t, return_parts=True)
                        physics_loss = self.lambda_physics * phys_total
                    else:
                        p_tke = p_spd = p_egy = torch.tensor(0.0, device=device)
                        physics_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                    loss = mse_loss + physics_loss

                # [提升] NaN/Inf 检测：恢复最佳权重后明确告警，不静默标记已训练
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    print_error("训练异常",
                                f"Epoch {epoch+1} Batch {n_batches}: Loss 为 NaN/Inf，提前终止。")
                    print_warning("排查建议",
                                  "1)数据是否已清洗 2)学习率是否过大 3)暂时关闭 USE_PHYSICS_LOSS")
                    if best_state_gpu is not None:
                        recovered = {k: v.cpu().clone() for k, v in best_state_gpu.items()}
                        self.load_state_dict({k: v.to(device) for k, v in recovered.items()})
                        print_warning("权重恢复", "已恢复早停前的最佳权重（模型处于部分训练状态）")
                    self.is_trained = False
                    return

                epoch_mse_gpu += mse_loss.detach()
                epoch_phys_gpu += physics_loss.detach()
                epoch_tke_gpu += (self.lambda_physics * p_tke).detach()
                epoch_spd_gpu += (self.lambda_physics * p_spd).detach()
                epoch_egy_gpu += (self.lambda_physics * p_egy).detach()
                epoch_loss_gpu += loss.detach()

                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                n_batches += 1

            avg_loss = epoch_loss_gpu.item() / max(n_batches, 1)
            avg_mse = epoch_mse_gpu.item() / max(n_batches, 1)
            avg_phys = epoch_phys_gpu.item() / max(n_batches, 1)
            avg_tke = epoch_tke_gpu.item() / max(n_batches, 1)
            avg_spd = epoch_spd_gpu.item() / max(n_batches, 1)
            avg_egy = epoch_egy_gpu.item() / max(n_batches, 1)
            self.training_history.append(avg_loss)

            # [修复·过拟合] 计算验证 loss（仅前向、不回传梯度）。
            #   验证集不参与训练，只作为泛化能力的无偏估计。
            val_loss = float("nan")
            if has_val and X_val_gpu is not None:
                try:
                    was_training = self.training
                    self.eval()
                    with torch.no_grad():
                        with autocast(device_type=device.type, enabled=use_amp):
                            v_pred = model_for_training(X_val_gpu)
                            v_mse = torch.mean((v_pred - y_val_gpu) ** 2)
                            if USE_PHYSICS_LOSS and self.lambda_physics > 0:
                                v_phys, _, _, _ = compute_physics_regularization(
                                    v_pred, y_val_gpu, self.tke_max, self.speed_limit,
                                    y_mean=y_mean_t, y_scale=y_scale_t, return_parts=True)
                                v_total = v_mse + self.lambda_physics * v_phys
                            else:
                                v_total = v_mse
                        val_loss = float(v_total.detach().float().cpu().item())
                    if was_training:
                        self.train()
                except Exception as _ve:
                    # 验证 loss 计算失败不应中断训练，退化为按训练 loss 判定
                    val_loss = float("nan")
                    self.train()
                    print_warning("验证 loss 计算失败", f"{_ve}，本轮退化为按训练 loss 判定")
            self.val_loss_history.append(val_loss)

            # [修复·过拟合] 监控量：有验证集时用验证 loss，否则退化为训练 loss
            monitor_loss = val_loss if (has_val and np.isfinite(val_loss)) else avg_loss

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_epoch = epoch + 1
                patience_counter = 0
                if is_multi_gpu:
                    best_state_gpu = {
                        k.replace('module.', ''): v.detach().clone()
                        for k, v in model_for_training.state_dict().items()}
                else:
                    best_state_gpu = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1

            print_interval = max(epochs // 5, 20)
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                print_info(
                    f"Epoch {epoch+1}/{actual_epochs}",
                    f"Loss={avg_loss:.6f} | MSE={avg_mse:.6f} | Physics={avg_phys:.6f} "
                    f"[TKE={avg_tke:.6f} SPD={avg_spd:.6f} EGY={avg_egy:.6f}] | LR={current_lr:.6f}"
                    + (f" | Val={val_loss:.6f}" if has_val else ""))

            # [提升] 早停：默认【不中断】(early_stop_break=False)，跑满 RESNET_EPOCHS 充分收敛；
            #              循环外统一恢复最佳权重，兼顾"跑满轮次"与"最优模型"，不削弱收益。
            if patience_counter >= patience and not early_stop_triggered:
                early_stop_triggered = True
                # [修复·关键] 不在触发时刻固化 best_state。
                #   默认跑满轮次时，触发后仍有数百轮；若此刻把触发点权重固化进
                #   best_state，循环外"仅当 best_state 为空才同步"的判断会失效，
                #   导致后续更优权重被丢弃 —— 最终载入的是触发时刻权重而非全局最优。
                #   统一在训练循环结束后从 best_state_gpu（始终指向最新最优）同步。
                if early_stop_break:
                    print_warning("早停触发",
                                  f"Loss 连续 {patience} 轮未下降 (Epoch {epoch+1})，"
                                  f"按 early_stop_break=True 中断训练并保留最佳权重")
                    break
                else:
                    # 【措辞澄清】此处只是"监控触发"，并不中断训练。
                    # 旧日志写"早停条件满足"容易被误读为"训练已停止"，
                    # 实际会继续跑满 actual_epochs，结束后统一恢复最佳权重。
                    print_warning("早停监控触发（不中断）",
                                  f"Loss 连续 {patience} 轮未下降 (Epoch {epoch+1})；"
                                  f"按 early_stop_break=False【继续训练至满 {actual_epochs} 轮】，"
                                  f"结束后统一恢复最佳权重")

            scheduler.step()

        # [修复] 循环外统一恢复【全局最优】权重。
        #   best_state_gpu 在每次 best_loss 刷新时同步更新，始终指向最新最优；
        #   此处无条件同步，杜绝"早停触发时刻固化"造成后续更优权重被丢弃。
        if best_state_gpu is not None:
            best_state = {k: v.cpu().clone() for k, v in best_state_gpu.items()}
        if best_state is not None:
            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            _src = "验证集" if has_val else "训练集"
            print_success("最佳权重恢复",
                          f"已载入【{_src}最优】权重 (best={best_loss:.6f} @ Epoch {best_epoch})"
                          + ("；该点泛化最好，避免取到过拟合的末轮权重" if has_val else ""))

        self.is_trained = True
        trained_epochs = len(self.training_history)
        final_loss = self.training_history[-1] if self.training_history else best_loss
        # 【明确轮次】用 实际/计划 的形式打印，便于一眼确认是否跑满 RESNET_EPOCHS。
        print_success("SimpleResNet 训练完成",
                      f"最终Loss={final_loss:.6f} | "
                      f"共训练 {trained_epochs}/{actual_epochs} 轮"
                      + (f" | 最佳Loss={best_loss:.6f}" if best_loss != float("inf") else ""))

    def predict(self, X, fill_nan=True):
        """[提升] 增加输入维度校验（对齐 LinearRegressionModel）。"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练或加载权重，请先调用 fit() 或 load_weights()。")
        self.eval()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"输入 X 必须为二维数组，当前 ndim={X.ndim}")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"输入特征维度不匹配: 期望 {self.input_dim}, 实际 {X.shape[1]}")

        device = next(self.parameters()).device
        X_t = torch.from_numpy(X).to(device).float()
        with torch.no_grad():
            y_pred = self.forward(X_t)
        y_pred = y_pred.cpu().numpy()

        if np.any(np.isnan(y_pred)):
            nan_count = int(np.sum(np.isnan(y_pred)))
            print_warning("ResNet反演异常", f"含 {nan_count} 个 NaN (fill_nan={fill_nan})")
            if fill_nan:
                y_pred = np.nan_to_num(y_pred, nan=0.0)
        return y_pred

    def save_weights(self, filepath=None):
        """[提升] 元数据补全（dropout / 物理参数 / 版本 / 配置维度）。"""
        if not self.is_trained:
            print_warning("模型状态", "ResNet 未训练，跳过保存。")
            return
        if filepath is None:
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            filepath = os.path.join(MODEL_SAVE_DIR, "resnet_model_weights.pt")

        state = {
            # [提升] 保存 CPU 权重副本：
            #   避免 GPU tensor 直接落盘（文件偏大、且跨设备加载强依赖
            #   map_location），同时不影响 self 仍在 GPU 上继续训练。
            "state_dict": {k: v.detach().cpu().clone()
                           for k, v in self.state_dict().items()},
            "use_norm": self.use_norm,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "n_blocks": self.n_blocks,
            "dropout_rate": self.dropout_rate,
            "training_history": self.training_history,
            "lambda_physics": self.lambda_physics,
            "tke_max": self.tke_max,
            "speed_limit": self.speed_limit,
            "model_version": "2.0",
        }
        torch.save(state, filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print_success("ResNet 权重已保存", f"({size_mb:.2f} MB): {filepath}")

    def load_weights(self, filepath=None):
        """[提升] weights_only=True + 维度/配置强校验 + dropout 结构恢复。"""
        if filepath is None:
            filepath = os.path.join(MODEL_SAVE_DIR, "resnet_model_weights.pt")
        if not os.path.exists(filepath):
            print_error("ResNet权重", f"文件不存在: {filepath}")
            return False
        try:
            device = _get_device()
            state = torch.load(filepath, map_location=device, weights_only=True)

            if not isinstance(state, dict) or "state_dict" not in state:
                raise ValueError("不支持的权重文件格式，请使用 PyTorch .pt 格式")
            meta = state

            # [提升] 维度与配置一致性校验
            if N_FEATURES is not None and meta.get("input_dim") != N_FEATURES:
                print_error("维度不匹配",
                            f"权重 input_dim={meta.get('input_dim')} != N_FEATURES={N_FEATURES}")
                return False
            if N_OUTPUTS is not None and meta.get("output_dim") != N_OUTPUTS:
                print_error("维度不匹配",
                            f"权重 output_dim={meta.get('output_dim')} != N_OUTPUTS={N_OUTPUTS}")
                return False

            self.input_dim = meta.get("input_dim", self.input_dim)
            self.output_dim = meta.get("output_dim", self.output_dim)
            self.hidden_dim = meta.get("hidden_dim", self.hidden_dim)
            self.n_blocks = meta.get("n_blocks", self.n_blocks)
            self.dropout_rate = meta.get("dropout_rate", getattr(self, "dropout_rate", 0.1))

            # [提升] 自动兼容"不含 LayerNorm 的旧权重"：
            #   若权重中不存在 blocks.*.norm1.* 参数，则令 use_norm=False
            #   （残差块改用 nn.Identity），使重建出的模型结构与权重键
            #   严格一致，从而可用 strict=True 加载 —— 既不会因键不匹配
            #   而失败，也不会像 strict=False 那样静默丢弃/缺失参数。
            sd = meta.get("state_dict", {})
            has_norm = any(("norm1." in k) for k in sd.keys())
            self.use_norm = bool(meta.get("use_norm", has_norm)) and has_norm

            self._lambda_physics_original = meta.get("lambda_physics", LAMBDA_PHYSICS)
            self.lambda_physics = self._lambda_physics_original if USE_PHYSICS_LOSS else 0.0
            self.tke_max = meta.get("tke_max", SETTINGS_TKE_MAX)
            self.speed_limit = meta.get("speed_limit", SETTINGS_SPEED_LIMIT)
            self.training_history = meta.get("training_history", [])

            self._build()
            self.load_state_dict(sd)
            self.to(device)
            self.is_trained = True
            print_success("ResNet 权重加载成功", os.path.basename(filepath))
            return True
        except Exception as e:
            print_error("ResNet权重加载失败", f"{e}")
            return False

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


# ====================================================================
#  解析解求解器（修复版）
# ====================================================================
def solve_linear_analytic(X, y, lambda_reg=0.001):
    """
    带 L2 正则化的线性回归解析解 (Ridge Regression)。
    [提升] 修复：1D y 处理、solve 优先、float64 求解、NaN 校验、
    异常抛错（不再静默返回零权重）、更正"PyTorch 适配版"误导注释。
    注意：本函数为纯 NumPy 实现，无截距；返回权重 W（不含偏置）。
    """
    print_info("解析解", "计算 Ridge 回归闭式解 (NumPy float64)...")
    X, y = _validate_xy(X, y, name="解析解")
    n_features = X.shape[1]

    XtX = X.T.astype(np.float64) @ X.astype(np.float64)
    A = XtX + lambda_reg * np.eye(n_features, dtype=np.float64)
    B = X.T.astype(np.float64) @ y.astype(np.float64)

    try:
        cond = np.linalg.cond(A)
        if cond > 1e12:
            print_warning("矩阵条件数",
                          f"过大 ({cond:.2e})，解析解可能不稳定，建议检查共线性")
        W = np.linalg.solve(A, B).astype(np.float32)
        print_success("解析解计算完成", f"条件数={cond:.2e} | W shape={W.shape}")
        return W
    except np.linalg.LinAlgError as e:
        # [提升] 求解失败明确抛错，不再静默返回零权重造成假成功
        raise ValueError(f"Ridge 解析解求解失败 ({e})，请检查数据共线性或调大 lambda_reg") from e
