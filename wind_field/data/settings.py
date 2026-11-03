# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 09:56:04 2026

@author: 38036
"""

"""
全局配置文件
统一管理路径、常量、显示参数等
"""
import os

# ==================== 路径配置 ====================
# 获取当前脚本所在的根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 输入数据文件路径
X_DATA_FILE = os.path.join(DATA_DIR, "X_data.txt")
Y_DATA_FILE = os.path.join(DATA_DIR, "y_data.txt")

# 标准化参数文件（学长提供的 scaler 参数）
SCALER_X_MEAN_FILE = os.path.join(DATA_DIR, "scaler_X_mean.txt")
SCALER_X_SCALE_FILE = os.path.join(DATA_DIR, "scaler_X_scale.txt")
SCALER_Y_MEAN_FILE = os.path.join(DATA_DIR, "scaler_Y_mean.txt")
SCALER_Y_SCALE_FILE = os.path.join(DATA_DIR, "scaler_Y_scale.txt")

# 输出目录
MODEL_PARAMS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

# ==================== 数据配置 ====================
N_FEATURES = 3      # 输入特征维度
N_OUTPUTS = 3       # 输出目标维度
TRAIN_RATIO = 0.8   # 训练集比例
RANDOM_SEED = 42    # 随机种子，保证结果可复现

# ==================== 模型超参数 ====================
# ResNet 网络配置
RESNET_HIDDEN_UNITS = 32
RESNET_NUM_LAYERS = 3
RESNET_LEARNING_RATE = 0.005
RESNET_EPOCHS = 150
RESNET_BATCH_SIZE = 256
GRADIENT_CLIP_VALUE = 5.0

# 物理正则化权重（对应学长论文中的物理约束）
PHYS_REG_WEIGHT_CONSISTENCY = 0.1
PHYS_REG_WEIGHT_DISSIPATION = 0.05
PHYS_REG_WEIGHT_BOUNDARY = 0.02

# ==================== 滤波参数 ====================
HAMPEL_WINDOW = 11      # Hampel滤波器窗口
HAMPEL_THRESHOLD = 3.0  # Hampel阈值
SG_WINDOW = 21          # SG滤波器窗口
SG_POLYORDER = 3        # SG多项式阶数

# ==================== 显示配置 ====================
TERMINAL_WIDTH = 72     # 终端显示宽度
SEPARATOR_CHAR = "="
SUB_SEPARATOR_CHAR = "-"
ENABLE_COLOR = False    # 设为False以兼容Spyder/IDLE，设为True在CMD下显示颜色


class Colors:
    """终端颜色代码定义"""
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def colorize(text, color_name):
    """根据配置决定是否给文本上色"""
    if not ENABLE_COLOR:
        return str(text)
    color_code = getattr(Colors, color_name.upper(), "")
    return f"{color_code}{text}{Colors.RESET}"