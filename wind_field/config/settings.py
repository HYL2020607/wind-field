# -*- coding: utf-8 -*-
"""
config/settings.py - 全局配置中心
================================
【说明】
本文件是所有超参数、路径、运行时开关的唯一来源，供 core/models.py、
main.py、data/ 等模块统一 import。

【PyTorch 迁移后新增项】
• TORCH_DEVICE：计算设备（"auto" 自动选 cuda / mps / cpu）

• NUM_WORKERS：DataLoader 并行加载线程数（已废弃，保留仅作兼容）

• TORCH_CUDNN_BENCHMARK：卷积/线性层自动择优 kernel

• RESNET_OPTIMIZER：优化器选择（adam / sgd）

• RESNET_WEIGHT_DECAY：Adam 的 L2 权重衰减系数

• RESNET_EPOCHS：已上调至 1000，保证充分收敛

"""

import os

# ==================== 基础路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

X_DATA_FILE = os.path.join(DATA_DIR, "X_data.txt")
Y_DATA_FILE = os.path.join(DATA_DIR, "y_data.txt")

# 标准化参数文件
SCALER_X_MEAN_FILE = os.path.join(DATA_DIR, "scaler_X_mean.txt")
SCALER_X_SCALE_FILE = os.path.join(DATA_DIR, "scaler_X_scale.txt")
SCALER_Y_MEAN_FILE = os.path.join(DATA_DIR, "scaler_Y_mean.txt")
SCALER_Y_SCALE_FILE = os.path.join(DATA_DIR, "scaler_Y_scale.txt")

# 模型保存与图表输出路径
MODEL_PARAMS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

# ==================== 计算设备与并行配置 ====================
TORCH_DEVICE = "auto"

# 【优化】DataLoader 已废弃，数据全量直载 GPU 显存，本项保留仅作兼容。
# （原 Windows + Spyder 下多进程崩溃问题已通过去 DataLoader 化彻底解决）
NUM_WORKERS = 4

TORCH_CUDNN_BENCHMARK = True

# ==================== 核心超参数配置 ====================
N_FEATURES = 3
N_OUTPUTS = 3
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# --- ResNet 模型架构 ---
RESNET_HIDDEN_UNITS = 256
RESNET_NUM_LAYERS = 3

RESNET_LEARNING_RATE = 0.0005

RESNET_EPOCHS = 1000

# 【提速】512 在 AMP 混合精度下足以喂满 GPU；若 OutOfMemory 再降到 256/128
RESNET_BATCH_SIZE = 512

GRADIENT_CLIP_VALUE = 5.0

RESNET_OPTIMIZER = "adam"

RESNET_WEIGHT_DECAY = 1e-4

# --- 数据预处理滤波器参数 ---
HAMPEL_WINDOW = 11
HAMPEL_THRESHOLD = 3.0
SG_WINDOW = 21
SG_POLYORDER = 3

# 【修复】上调至 150，配合"实打实跑 1000 轮"，避免因早停提前打断收敛
RESNET_EARLY_STOPPING_PATIENCE = 150

# 【保留】物理约束必须开启，这是 PhyReg-Net 的核心理念
USE_PHYSICS_LOSS = True

# 物理正则化超参数（对齐 PhyReg-Net 论文物理约束理念）
LAMBDA_L2 = 1e-4
# 【修复】1e-2 -> 1e-3：降低物理惩罚权重，避免初期物理梯度压制 MSE 主任务。
# 学习率由 LinearLR(前5轮热身) + CosineAnnealingWarmRestarts 管理，
# 不再出现调度器将学习率连续压到瘫痪的问题。物理约束真实参与反向传播。
LAMBDA_PHYSICS = 1e-3
TKE_MAX = 5.0
SPEED_LIMIT = 50.0

# 模型权重文件路径
# 【修改】恢复 LINEAR_MODEL_WEIGHTS_FILE / RESNET_MODEL_WEIGHTS_FILE 常量定义。
# 原因：main.py 仍依赖此常量导入（尽管 models.py 内部实际拼装路径为 .pkl/.pt）。
LINEAR_MODEL_WEIGHTS_FILE = os.path.join(MODEL_PARAMS_DIR, "linear_regression_weights.npy")
RESNET_MODEL_WEIGHTS_FILE = os.path.join(MODEL_PARAMS_DIR, "resnet_model_weights.npy")

# ==================== 显示与高级控制 ====================
TERMINAL_WIDTH = 72
SEPARATOR_CHAR = "="
SUB_SEPARATOR_CHAR = "-"

# 【关键开关】设为 True 强制根据当前数据重新计算标准化参数
# ⚠️ 生产/演示提醒：开发调试阶段保持 True 没问题；
#    向导师演示或生产部署前，建议改为 False，
#    确保使用稳定的 scaler 参数，避免因数据微调导致模型表现波动。
FORCE_RESCALER = True

# 【推理模式开关】
INFERENCE_ONLY_MODE = False

# ==================== 路径初始化 ====================
os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)