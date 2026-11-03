# -*- coding: utf-8 -*-
"""
风场预测系统-主控程序(重构版)
系统架构(按导师要求重构):
第一部分:多源数据的读取与预处理
第二部分:三维风场高精度反演(模型推理)
第三部分:反演结果评估
设计原则:
1.所有配置从 config.settings统一导入,禁止在 main.py中重复定义
2.数据加载使用 core.data_loader，预处理使用 core.preprocessing
3.模型定义使用 core.models,评估使用 core.evaluator
4.成品系统模式:仅执行推理(加载已有模型权重)，不进行训练
5.开发调试模式:可执行完整流程(训练+推理+评估)
"""

# ====================== 1.统一配置导入 ======================
from config.settings import (
    N_FEATURES, N_OUTPUTS, TRAIN_RATIO, RANDOM_SEED,
    RESNET_HIDDEN_UNITS, RESNET_NUM_LAYERS, RESNET_LEARNING_RATE,
    RESNET_EPOCHS, RESNET_BATCH_SIZE, GRADIENT_CLIP_VALUE,
    HAMPEL_WINDOW, HAMPEL_THRESHOLD, SG_WINDOW, SG_POLYORDER,
    X_DATA_FILE, Y_DATA_FILE,
    MODEL_PARAMS_DIR, PLOT_DIR,
    LINEAR_MODEL_WEIGHTS_FILE, RESNET_MODEL_WEIGHTS_FILE,
    TERMINAL_WIDTH, SEPARATOR_CHAR, SUB_SEPARATOR_CHAR,
    FORCE_RESCALER, INFERENCE_ONLY_MODE
)

# ===================== 2.核心模块导入 ======================
from core.data_loader import load_data, load_scaler_params, WindFieldScaler
from core.preprocessing import preprocess_pipeline
from core.models import LinearRegressionModel, SimpleResNet
from core.evaluator import ModelEvaluator

# ===================== 3.UI模块导入 =====================
from ui.display import print_menu, print_model_results, clear_screen
from ui.visualizer import plot_wind_comparison
from utils.helpers import (
    print_info, print_success, print_warning, print_error,
    print_step, ensure_dir, print_separator, print_sub_separator
)

# ===================== 4.第三方库导入 ======================
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFilter

# ===================== 5.全局状态管理 ======================
data_store = {
    "X_raw": None, "y_raw": None,
    "X_clean": None, "y_clean": None,
    "X_scaled": None, "y_scaled": None,
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "X_pred": None, "y_pred": None,
    "y_pred_linear": None, "y_pred_resnet": None,
    "scaler_X": None, "scaler_y": None,
    "linear_history": None, "resnet_history": None,
}

models = {"linear": None, "resnet": None}

scaler_X = None
scaler_y = None
evaluator = ModelEvaluator()

_canvas_ref = {"canvas": None, "fig": None}

PLOT_PREVIEW_POINTS = 10000
PLOT_SMOOTH_WINDOW = 200


def _smooth(y, window=None):
    """对一维数组做滑动平均滤波(边缘用有效长度,不丢弃端点)。"""
    if window is None:
        window = PLOT_SMOOTH_WINDOW
    window = min(max(1, int(window)), len(y))
    if window <= 1:
        return y
    w = np.ones(window, dtype=np.float32) / window
    return np.convolve(y, w, mode="valid")


def _preview_indices(n, max_points=PLOT_PREVIEW_POINTS):
    n = max(1, int(n))
    max_points = max(1000, int(max_points))
    if n <= max_points:
        return np.arange(n, dtype=np.int32)
    indices = np.unique(np.linspace(0, n - 1, max_points, dtype=np.int32))
    return indices


# ===================== 5.5 数据增强与正则化工具 ======================

DATA_AUGMENT_ENABLED = True
DATA_AUGMENT_NOISE_STD = 0.02
DATA_AUGMENT_SHIFT_RANGE = 2
DATA_AUGMENT_SCALE_RANGE = 0.05
DATA_AUGMENT_MIXUP_ALPHA = 0.2

DROPOUT_ENABLED = True
DROPOUT_RATE = 0.1


def augment_training_data(X, y, seed=None):
    """训练集数据增强函数。"""
    if not DATA_AUGMENT_ENABLED:
        return X, y
    if seed is not None:
        np.random.seed(seed)

    X_aug = X.copy().astype(np.float32)
    y_aug = y.copy().astype(np.float32)

    # 高斯噪声
    noise_x = np.random.normal(0, DATA_AUGMENT_NOISE_STD, X_aug.shape)
    X_aug = X_aug + noise_x
    noise_y = np.random.normal(0, DATA_AUGMENT_NOISE_STD, y_aug.shape)
    y_aug = y_aug + noise_y

    # 时间偏移
    if X_aug.shape[0] > DATA_AUGMENT_SHIFT_RANGE * 2:
        shift = np.random.randint(-DATA_AUGMENT_SHIFT_RANGE, DATA_AUGMENT_SHIFT_RANGE)
        if shift != 0:
            X_aug = np.roll(X_aug, shift, axis=0)
            y_aug = np.roll(y_aug, shift, axis=0)

    # 幅度缩放
    scale_x = 1.0 + np.random.uniform(-DATA_AUGMENT_SCALE_RANGE,
                                       DATA_AUGMENT_SCALE_RANGE, size=X_aug.shape[1])
    X_aug = X_aug * scale_x
    scale_y = 1.0 + np.random.uniform(-DATA_AUGMENT_SCALE_RANGE,
                                       DATA_AUGMENT_SCALE_RANGE, size=y_aug.shape[1])
    y_aug = y_aug * scale_y

    # Mixup
    if X_aug.shape[0] >= 2:
        X_orig = X_aug.copy()
        y_orig = y_aug.copy()
        mixup_indices = np.random.permutation(X_aug.shape[0])[:X_aug.shape[0] // 2]
        for idx in mixup_indices:
            other_idx = np.random.randint(0, X_aug.shape[0])
            while other_idx == idx:
                other_idx = np.random.randint(0, X_aug.shape[0])
            lam = np.random.beta(DATA_AUGMENT_MIXUP_ALPHA, DATA_AUGMENT_MIXUP_ALPHA)
            X_aug[idx] = lam * X_aug[idx] + (1 - lam) * X_orig[other_idx]
            y_aug[idx] = lam * y_aug[idx] + (1 - lam) * y_orig[other_idx]

    return X_aug, y_aug


def inject_dropout_to_resnet(model, dropout_rate=0.1):
    """已废弃，PyTorch原生支持Dropout。"""
    if not DROPOUT_ENABLED:
        print_info("Dropout", "已禁用")
        return False
    print_info("Dropout", "PyTorch原生已支持Dropout控制，monkey-patch跳过注入")
    return False


def inject_dropout_to_resnet_v2(model, dropout_rate=0.1):
    """备选方案，已废弃。"""
    if not DROPOUT_ENABLED:
        return False
    print_info("Dropout(v2)", "PyTorch原生已支持Dropout控制，备选方案已废弃，跳过注入")
    return False


# ===================== 5.6 dtype安全包装 =====================

def _safe_linear_fit(linear_model, X, y):
    """安全包装LinearRegressionModel.fit()，确保传入数据为连续内存float32。"""
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    y32 = np.ascontiguousarray(y, dtype=np.float32)
    linear_model.fit(X32, y32)
    if hasattr(linear_model, 'weights_') and linear_model.weights_ is not None:
        w = linear_model.weights_
        if hasattr(w, 'dtype') and w.dtype != np.float32:
            linear_model.weights_ = w.astype(np.float32)
    return linear_model


# ===================== 6. 多源数据的读取与预处理 =====================

def step_load_and_preprocess():
    """步骤1:多源数据的读取与预处理。"""
    print_step("1.1", "多源数据读取")
    try:
        X, y = load_data(X_DATA_FILE, Y_DATA_FILE)
        data_store["X_raw"] = X
        data_store["y_raw"] = y
        print_success(f"数据读取完成:X={X.shape}, y={y.shape}")
    except FileNotFoundError as e:
        print_error(f"数据文件读取失败:{e}")
        raise
    except Exception as e:
        print_error(f"数据读取异常:{e}")
        raise

    print_step("1.2", "数据预处理(异常清洗+Hampel滤波+S-G平滑)")
    try:
        X_clean, y_clean, report = preprocess_pipeline(
            data_store["X_raw"], data_store["y_raw"],
            use_hampel=True, use_sg=True
        )
        data_store["X_clean"] = X_clean
        data_store["y_clean"] = y_clean
        print_success(f"预处理完成:清洗异常值 X={report.get('X_cleaned', 0)}个，y={report.get('y_cleaned', 0)}个")

        if X_clean.size == 0 or y_clean.size == 0:
            raise ValueError("预处理后数据为空，请检查原始数据或清洗逻辑")
        if np.isnan(X_clean).any() or np.isinf(X_clean).any():
            raise ValueError("清洗后X数据包含NaN或无穷大，请检查数据源")
        if np.isnan(y_clean).any() or np.isinf(y_clean).any():
            raise ValueError("清洗后y数据包含 NaN或无穷大,请检查数据源")
    except Exception as e:
        print_error(f"预处理失败:{e}")
        raise

    print_step("1.3", "训练/测试集分割")
    try:
        m = X_clean.shape[0]
        split_idx = int(m * TRAIN_RATIO)
        data_store["X_train"] = X_clean[:split_idx]
        data_store["X_test"] = X_clean[split_idx:]
        data_store["y_train"] = y_clean[:split_idx]
        data_store["y_test"] = y_clean[split_idx:]
        print_success(f"数据集分割完成:训练集={data_store['X_train'].shape[0]}条，测试集={data_store['X_test'].shape[0]}条")
    except Exception as e:
        print_error(f"数据集分割失败:{e}")
        raise

    print_step("1.4", "数据标准化(Z-score)")
    try:
        global scaler_X, scaler_y
        if FORCE_RESCALER:
            scaler = WindFieldScaler()
            X_train_scaled, y_train_scaled = scaler.fit_transform(data_store["X_train"], data_store["y_train"])
            X_test_scaled, y_test_scaled = scaler.transform(data_store["X_test"], data_store["y_test"])
            data_store["scaler_X"] = scaler
            data_store["scaler_y"] = scaler
            print_info("标准化", "已根据当前训练数据重新计算标准化参数")
        else:
            scaler_loaded = load_scaler_params()
            if scaler_loaded is not None:
                X_train_scaled, y_train_scaled = scaler_loaded.transform(data_store["X_train"], data_store["y_train"])
                X_test_scaled, y_test_scaled = scaler_loaded.transform(data_store["X_test"], data_store["y_test"])
                data_store["scaler_X"] = scaler_loaded
                data_store["scaler_y"] = scaler_loaded
                print_info("标准化", "已加载已有标准化参数")
            else:
                scaler = WindFieldScaler()
                X_train_scaled, y_train_scaled = scaler.fit_transform(data_store["X_train"], data_store["y_train"])
                X_test_scaled, y_test_scaled = scaler.transform(data_store["X_test"], data_store["y_test"])
                data_store["scaler_X"] = scaler
                data_store["scaler_y"] = scaler
                print_warning("未找到已有标准化参数文件，已根据当前训练数据重新计算")

        data_store["X_train"] = X_train_scaled
        data_store["X_test"] = X_test_scaled
        data_store["y_train"] = y_train_scaled
        data_store["y_test"] = y_test_scaled
        print_success("数据标准化完成")
    except Exception as e:
        print_error(f"标准化失败:{e}")
        raise

    print_step("1.5", "数据类型统一(float64→ float32)")
    try:
        for key in ["X_train", "X_test", "y_train", "y_test"]:
            if data_store[key] is not None:
                data_store[key] = data_store[key].astype(np.float32)
        print_success("数据类型已统一为 float32")
    except Exception as e:
        print_error(f"数据类型转换失败:{e}")
        raise

    print_success("[步骤1完成]多源数据的读取与预处理全部完成")
    return data_store


# ===================== 7. 三维风场高精度反演(第二部分) =====================

def step_inference(train_models=False):
    """步骤2:三维风场高精度反演(模型推理)。"""
    print_step("2", "三维风场高精度反演(模型推理)")

    if data_store["X_test"] is None:
        print_warning("尚未执行数据预处理,自动执行步骤 1...")
        step_load_and_preprocess()

    print_info("子步骤2.1", "模型准备")

    if train_models or not INFERENCE_ONLY_MODE:
        print_info("运行模式", "[训练模式]将训练模型并保存权重")

        from config.settings import USE_PHYSICS_LOSS as _USE_PHYSICS_LOSS
        if _USE_PHYSICS_LOSS:
            print_warning("[注意]物理正则化(USE_PHYSICS_LOSS)已启用，确保数据已清洗干净")
        else:
            print_info("物理正则化", "已禁用(纯MSE)，可在config/settings.py中开启")

        if np.isnan(data_store["X_train"]).any() or np.isinf(data_store["X_train"]).any():
            raise ValueError("训练集X包含NaN或无穷大，模型训练将失败")
        if np.isnan(data_store["y_train"]).any() or np.isinf(data_store["y_train"]).any():
            raise ValueError("训练集y包含NaN或无穷大，模型训练将失败")

        # 线性回归
        print_info("训练", "线性回归模型...")
        linear_model = LinearRegressionModel()
        _safe_linear_fit(linear_model, data_store["X_train"], data_store["y_train"])
        linear_model.save_weights()
        data_store["linear_history"] = linear_model.training_history
        models["linear"] = linear_model
        print_success("线性回归模型训练并保存完成")

        # ResNet
        print_info("训练", "ResNet模型...")
        resnet_model = SimpleResNet(
            hidden_dim=RESNET_HIDDEN_UNITS,
            n_blocks=RESNET_NUM_LAYERS
        )

        augmentation_applied = False
        if DATA_AUGMENT_ENABLED:
            print_info("数据增强", "正在对训练集进行数据增强...")
            try:
                X_train_aug, y_train_aug = augment_training_data(
                    data_store["X_train"], data_store["y_train"], seed=42
                )
                X_train_combined = np.concatenate([data_store["X_train"], X_train_aug], axis=0)
                y_train_combined = np.concatenate([data_store["y_train"], y_train_aug], axis=0)
                perm = np.random.permutation(X_train_combined.shape[0])
                X_train_combined = X_train_combined[perm]
                y_train_combined = y_train_combined[perm]
                original_X_train = data_store["X_train"]
                original_y_train = data_store["y_train"]
                data_store["X_train"] = X_train_combined
                data_store["y_train"] = y_train_combined
                augmentation_applied = True
                print_success(f"数据增强完成:增强后 X={X_train_combined.shape}, y={y_train_combined.shape}")
            except Exception as e:
                print_warning(f"数据增强失败:{e}，将使用原始训练数据")

        if DROPOUT_ENABLED:
            print_info("Dropout", "正在为ResNet注入Dropout...")
            try:
                inject_dropout_to_resnet(resnet_model, dropout_rate=DROPOUT_RATE)
            except Exception as e:
                print_warning(f"Dropout注入失败:{e}")

        try:
            resnet_model.fit(
                data_store["X_train"], data_store["y_train"],
                learning_rate=RESNET_LEARNING_RATE,
                epochs=RESNET_EPOCHS,
                batch_size=RESNET_BATCH_SIZE,
                grad_clip=GRADIENT_CLIP_VALUE
            )
            resnet_model.save_weights()
            data_store["resnet_history"] = resnet_model.training_history
            models["resnet"] = resnet_model
        finally:
            if augmentation_applied:
                data_store["X_train"] = original_X_train
                data_store["y_train"] = original_y_train
                print_info("数据增强", "已恢复原始训练数据(训练完成)")

        print_success("ResNet模型训练并保存完成")

    else:
        # 推理模式
        print_info("运行模式", "推理模式加载已有模型权重")
        try:
            linear_model = LinearRegressionModel()
            if linear_model.load_weights():
                models["linear"] = linear_model
                print_success("线性回归模型权重加载完成")
            else:
                raise FileNotFoundError("线性回归模型权重未找到")
        except FileNotFoundError:
            print_warning("线性回归模型权重未找到，将临时训练一个")
            linear_model = LinearRegressionModel()
            _safe_linear_fit(linear_model, data_store["X_train"], data_store["y_train"])
            linear_model.save_weights()
            data_store["linear_history"] = linear_model.training_history
            models["linear"] = linear_model

        try:
            resnet_model = SimpleResNet(
                hidden_dim=RESNET_HIDDEN_UNITS,
                n_blocks=RESNET_NUM_LAYERS
            )
            if resnet_model.load_weights():
                data_store["resnet_history"] = resnet_model.training_history
                models["resnet"] = resnet_model
                print_success("ResNet模型权重加载完成")
            else:
                raise FileNotFoundError("ResNet模型权重未找到")
        except FileNotFoundError:
            print_warning("ResNet模型权重未找到，将临时训练一个")
            resnet_model = SimpleResNet(
                hidden_dim=RESNET_HIDDEN_UNITS,
                n_blocks=RESNET_NUM_LAYERS
            )
            if DATA_AUGMENT_ENABLED:
                try:
                    original_X_train_fb = data_store["X_train"]
                    original_y_train_fb = data_store["y_train"]
                    X_train_aug, y_train_aug = augment_training_data(
                        data_store["X_train"], data_store["y_train"], seed=42
                    )
                    X_train_combined = np.concatenate([data_store["X_train"], X_train_aug], axis=0)
                    y_train_combined = np.concatenate([data_store["y_train"], y_train_aug], axis=0)
                    perm = np.random.permutation(X_train_combined.shape[0])
                    data_store["X_train"] = X_train_combined[perm]
                    data_store["y_train"] = y_train_combined[perm]
                except Exception as e:
                    print_warning(f"数据增强失败:{e}")

            resnet_model.fit(
                data_store["X_train"], data_store["y_train"],
                learning_rate=RESNET_LEARNING_RATE,
                epochs=RESNET_EPOCHS,
                batch_size=RESNET_BATCH_SIZE,
                grad_clip=GRADIENT_CLIP_VALUE
            )
            resnet_model.save_weights()
            data_store["resnet_history"] = resnet_model.training_history
            models["resnet"] = resnet_model

            if DATA_AUGMENT_ENABLED and 'original_X_train_fb' in locals():
                data_store["X_train"] = original_X_train_fb
                data_store["y_train"] = original_y_train_fb

    try:
        resnet_model.eval()
    except Exception:
        pass

    # 推理
    print_info("子步骤2.2", "执行三维风场高精度反演(推理)")
    try:
        data_store["y_pred_linear"] = models["linear"].predict(data_store["X_test"])
        data_store["y_pred_resnet"] = models["resnet"].predict(data_store["X_test"])
        print_success(f"推理完成:线性回归输出={data_store['y_pred_linear'].shape}, ResNet输出={data_store['y_pred_resnet'].shape}")
    except Exception as e:
        print_error(f"推理失败:{e}")
        raise

    print_success("[步骤2完成]三维风场高精度反演完成")
    return models


# ===================== 8. 反演结果评估(第三部分) =====================

def step_evaluate():
    """步骤3:反演结果评估。"""
    print_step("3", "反演结果评估")

    if data_store.get("y_pred_linear") is None or data_store.get("y_pred_resnet") is None:
        print_warning("尚未执行模型推理，自动执行步骤2...")
        step_inference()

    # 线性回归评估
    print_info("评估模型", "Linear Regression")
    try:
        y_true = data_store["y_test"]
        y_pred = data_store["y_pred_linear"]
        _scaler_y = data_store.get("scaler_y")
        if _scaler_y is not None:
            try:
                _y_true_orig = _scaler_y.inverse_transform_y(y_true)
                _y_pred_orig = _scaler_y.inverse_transform_y(y_pred)
            except Exception as e:
                print_warning(f"反标准化失败:{e}")
                _y_true_orig, _y_pred_orig = y_true, y_pred
        else:
            _y_true_orig, _y_pred_orig = y_true, y_pred

        linear_metrics = evaluator.evaluate(_y_true_orig, _y_pred_orig, model_name="Linear Regression")
        data_store["linear_metrics"] = linear_metrics
        print_success("线性回归模型评估完成")
    except Exception as e:
        print_error(f"线性回归评估失败:{e}")
        raise

    linear_r2 = data_store.get("linear_metrics", {}).get("R^2")
    if linear_r2 is not None and not np.isnan(linear_r2):
        if linear_r2 < 0:
            print_warning(f"线性回归 R^2={linear_r2:.4f}<0,但按约定不中断流程")
        else:
            print_success(f"线性回归 R^2={linear_r2:.4f}>=0,数据基础正常")

    # ResNet评估
    print_info("评估模型", "ResNet")
    try:
        y_true = data_store["y_test"]
        y_pred = data_store["y_pred_resnet"]
        _scaler_y2 = data_store.get("scaler_y")
        if _scaler_y2 is not None:
            try:
                _y_true_orig2 = _scaler_y2.inverse_transform_y(y_true)
                _y_pred_orig2 = _scaler_y2.inverse_transform_y(y_pred)
            except Exception as e:
                print_warning(f"反标准化失败:{e}")
                _y_true_orig2, _y_pred_orig2 = y_true, y_pred
        else:
            _y_true_orig2, _y_pred_orig2 = y_true, y_pred

        resnet_metrics = evaluator.evaluate(_y_true_orig2, _y_pred_orig2, model_name="ResNet")
        data_store["resnet_metrics"] = resnet_metrics
        print_success("ResNet模型评估完成")
    except Exception as e:
        print_error(f"ResNet评估失败:{e}")
        raise

    print_info("评估汇总", "======模型性能对比======")
    results_dict = {
        "Linear Regression": data_store.get("linear_metrics", {}),
        "ResNet": data_store.get("resnet_metrics", {}),
    }
    print_model_results(results_dict)
    print_success("[步骤3完成]反演结果评估全部完成")
    return results_dict


# ===================== 9. 绘制预测对比图 =====================

def step_plot(canvas, fig):
    """步骤4:绘制预测对比图。"""
    print_step("4", "绘制反演对比图")

    if data_store.get("y_pred_linear") is None or data_store.get("y_pred_resnet") is None:
        print_warning("尚未执行模型推理,请先执行步骤2。")
        return "请先执行三维风场高精度反演步骤。"

    try:
        y_test = data_store["y_test"]
        y_linear_preds = data_store["y_pred_linear"]
        y_resnet_preds = data_store["y_pred_resnet"]

        _scaler_y3 = data_store.get("scaler_y")
        if _scaler_y3 is not None:
            try:
                y_test = _scaler_y3.inverse_transform_y(y_test)
                y_linear_preds = _scaler_y3.inverse_transform_y(y_linear_preds)
                y_resnet_preds = _scaler_y3.inverse_transform_y(y_resnet_preds)
            except Exception as e:
                print_warning(f"反标准化失败:{e}")

        indices = _preview_indices(len(y_test))
        x_preview = np.arange(len(y_test))[indices]
        y_true_preview = y_test[indices]
        y_linear_preview = y_linear_preds[indices]
        y_resnet_preview = y_resnet_preds[indices]

        fig.clf()
        axs = fig.subplots(N_OUTPUTS, 1)
        if N_OUTPUTS == 1:
            axs = [axs]

        titles = [
            "U分量(东西向风速,m/s)反演对比",
            "V分量(南北向风速,m/s)反演对比",
            "W分量(垂直向风速,m/s)反演对比"
        ]

        for i in range(N_OUTPUTS):
            y_true_smooth = _smooth(y_true_preview[:, i])
            axs[i].plot(x_preview[:len(y_true_smooth)], y_true_smooth,
                        label="真实值(滤波后)", linewidth=1.4, color="#123b63")
            axs[i].plot(x_preview, y_linear_preview[:, i],
                        label="线性回归", linestyle="--", linewidth=1.0,
                        color="#2b7bba", alpha=0.85)
            axs[i].plot(x_preview, y_resnet_preview[:, i],
                        label="ResNet", linestyle="-.", linewidth=1.1,
                        color="#d1495b", alpha=0.9)
            axs[i].set_title(titles[i], fontsize=12)
            axs[i].legend(loc="best", fontsize=9, framealpha=0.9)
            axs[i].grid(True, linestyle="-", alpha=0.45)

        fig.tight_layout()
        canvas.draw()

        save_path = os.path.join(PLOT_DIR, "wind_field_comparison.png")
        fig_full = plt.figure(figsize=(15, 4 * max(1, N_OUTPUTS)), dpi=150)
        try:
            axs_full = fig_full.subplots(N_OUTPUTS, 1)
            if N_OUTPUTS == 1:
                axs_full = [axs_full]
            x_full = np.arange(len(y_test))
            for i in range(N_OUTPUTS):
                y_true_full_smooth = _smooth(y_test[:, i])
                axs_full[i].plot(x_full[:len(y_true_full_smooth)], y_true_full_smooth,
                                 label="真实值(滤波后)", linewidth=1.0, color="#123b63")
                axs_full[i].plot(x_full, y_linear_preds[:, i],
                                 label="线性回归", linestyle="--", linewidth=0.8,
                                 color="#2b7bba", alpha=0.75)
                axs_full[i].plot(x_full, y_resnet_preds[:, i],
                                 label="ResNet", linestyle="-.", linewidth=0.9,
                                 color="#d1495b", alpha=0.8)
                axs_full[i].set_title(titles[i], fontsize=12)
                axs_full[i].legend(loc="best", fontsize=9, framealpha=0.9)
                axs_full[i].grid(True, linestyle="--", alpha=0.45)
            fig_full.tight_layout()
            fig_full.savefig(save_path, dpi=150)
        finally:
            plt.close(fig_full)

        print_success(f"图表绘制成功，已保存至:{save_path}")
        return f"图表绘制成功,已保存至:{save_path}"
    except Exception as e:
        print_error(f"绘图失败:{e}")
        return f"绘图失败:{e}"


# ===================== 9.5 训练集预测可视化 =====================

def step_plot_training_set(canvas, fig):
    """步骤4.5:绘制训练集预测对比图。"""
    print_step("4.5", "绘制训练集反演对比图")

    if data_store["X_train"] is None:
        print_warning("尚未执行数据预处理,请先执行步骤1。")
        return "请先执行数据预处理步骤。"

    try:
        y_train = data_store["y_train"]
        y_train_pred_linear = models["linear"].predict(data_store["X_train"])
        y_train_pred_resnet = models["resnet"].predict(data_store["X_train"])

        _scaler_y4 = data_store.get("scaler_y")
        if _scaler_y4 is not None:
            try:
                y_train = _scaler_y4.inverse_transform_y(y_train)
                y_train_pred_linear = _scaler_y4.inverse_transform_y(y_train_pred_linear)
                y_train_pred_resnet = _scaler_y4.inverse_transform_y(y_train_pred_resnet)
            except Exception as e:
                print_warning(f"反标准化失败:{e}")

        indices = _preview_indices(len(y_train))
        x_preview = np.arange(len(y_train))[indices]
        y_train_preview = y_train[indices]
        y_train_linear_preview = y_train_pred_linear[indices]
        y_train_resnet_preview = y_train_pred_resnet[indices]

        fig.clf()
        axs = fig.subplots(N_OUTPUTS, 1)
        if N_OUTPUTS == 1:
            axs = [axs]

        titles = [
            "训练集-U分量(东西向风速,m/s)反演对比",
            "训练集-V分量(南北向风速,m/s)反演对比",
            "训练集-W分量(垂直向风速,m/s)反演对比"
        ]

        for i in range(N_OUTPUTS):
            y_train_smooth = _smooth(y_train_preview[:, i])
            axs[i].plot(x_preview[:len(y_train_smooth)], y_train_smooth,
                        label="真实值(滤波后)", linewidth=1.4, color="#123b63")
            axs[i].plot(x_preview, y_train_linear_preview[:, i],
                        label="线性回归", linestyle="--", linewidth=1.0,
                        color="#2b7bba", alpha=0.85)
            axs[i].plot(x_preview, y_train_resnet_preview[:, i],
                        label="ResNet", linestyle="-.", linewidth=1.1,
                        color="#d1495b", alpha=0.9)
            axs[i].set_title(titles[i], fontsize=12)
            axs[i].legend(loc="best", fontsize=9, framealpha=0.9)
            axs[i].grid(True, linestyle="--", alpha=0.45)

        fig.tight_layout()
        canvas.draw()

        save_path = os.path.join(PLOT_DIR, "wind_field_comparison_training.png")
        fig_full = plt.figure(figsize=(15, 4 * max(1, N_OUTPUTS)), dpi=150)
        try:
            axs_full = fig_full.subplots(N_OUTPUTS, 1)
            if N_OUTPUTS == 1:
                axs_full = [axs_full]
            x_full = np.arange(len(y_train))
            for i in range(N_OUTPUTS):
                y_train_full_smooth = _smooth(y_train[:, i])
                axs_full[i].plot(x_full[:len(y_train_full_smooth)], y_train_full_smooth,
                                 label="真实值(滤波后)", linewidth=1.0, color="#123b63")
                axs_full[i].plot(x_full, y_train_pred_linear[:, i],
                                 label="线性回归", linestyle="--", linewidth=0.8,
                                 color="#2b7bba", alpha=0.75)
                axs_full[i].plot(x_full, y_train_pred_resnet[:, i],
                                 label="ResNet", linestyle="-.", linewidth=0.9,
                                 color="#d1495b", alpha=0.8)
                axs_full[i].set_title(titles[i], fontsize=12)
                axs_full[i].legend(loc="best", fontsize=9, framealpha=0.9)
                axs_full[i].grid(True, linestyle="-", alpha=0.45)
            fig_full.tight_layout()
            fig_full.savefig(save_path, dpi=150)
        finally:
            plt.close(fig_full)

        print_success(f"训练集图表绘制成功，已保存至:{save_path}")
        return f"训练集图表绘制成功，已保存至:{save_path}"
    except Exception as e:
        print_error(f"训练集绘图失败:{e}")
        return f"训练集绘图失败:{e}"


# ===================== 9.6 训练损失曲线图 =====================

def step_plot_loss_curves(canvas, fig):
    """步骤4.6:绘制训练损失曲线图。"""
    print_step("4.6", "绘制训练损失曲线图(仅保存到文件)")
    try:
        fig.clf()
        ax = fig.add_subplot(111)

        linear_history = data_store.get("linear_history")
        if linear_history is not None and len(linear_history) > 0:
            epochs_linear = np.arange(1, len(linear_history) + 1)
            ax.plot(epochs_linear, linear_history, label="Linear Regression",
                    linewidth=1.5, color="#E85D00")

        resnet_history = data_store.get("resnet_history")
        if resnet_history is not None and len(resnet_history) > 0:
            epochs_resnet = np.arange(1, len(resnet_history) + 1)
            ax.plot(epochs_resnet, resnet_history, label="ResNet",
                    linewidth=1.5, color="#3D9970")

        ax.set_title("训练损失曲线(Training Loss Curves)", fontsize=13)
        ax.set_xlabel("训练轮次(Epoch)", fontsize=11)
        ax.set_ylabel("Loss(MSE+正则化)", fontsize=11)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

        save_path = os.path.join(PLOT_DIR, "training_loss_curves.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        if linear_history:
            print_info("Linear Regression",
                       f"训练{len(linear_history)}轮|初始Loss={linear_history[0]:.6f}|最终 Loss={linear_history[-1]:.6f}")
        if resnet_history:
            print_info("ResNet",
                       f"训练{len(resnet_history)}轮|初始Loss={resnet_history[0]:.6f}|最终 Loss={resnet_history[-1]:.6f}")

        print_success(f"损失曲线已保存到:{save_path}(仅文件，不在程序中展示)")
        return f"损失曲线已保存到:{save_path}(仅文件，不在程序中展示)"
    except Exception as e:
        print_error(f"损失曲线绘图失败:{e}")
        return f"损失曲线绘图失败:{e}"


# ===================== 10. 一键自动全流程 =====================

def run_auto_all(canvas, fig):
    """一键自动全流程。"""
    print_separator()
    print_info("全流程", "开始一键自动全流程...")
    print_separator()
    try:
        step_load_and_preprocess()
        step_inference(train_models=not INFERENCE_ONLY_MODE)
        step_evaluate()
        step_plot(canvas, fig)
        try:
            step_plot_training_set(canvas, fig)
        except Exception as e:
            print_warning(f"训练集可视化跳过:{e}")
        try:
            step_plot_loss_curves(canvas, fig)
        except Exception as e:
            print_warning(f"损失曲线可视化跳过:{e}")

        print_separator()
        print_success("一键自动全流程全部完成!")
        print_separator()
    except Exception as e:
        print_error(f"全流程执行失败:{e}")
        raise


# ===================== 11. GUI主程序 =====================

def main():
    """Tkinter GUI主程序入口。"""
    root = tk.Tk()
    root.title("风场反演系统-三维风场高精度反演平台")
    root.geometry("1360x860")
    root.configure(bg="#0c1a30")
    root.minsize(1100, 700)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("Dark.TFrame", background="#0c1a30")
    style.configure("Dark.TLabel", background="#0c1a30", foreground="#e2e8f0")
    style.configure("Card.TFrame", background="#ffffff")
    style.configure("Card.TLabelframe", background="#f1f5f9",
                    foreground="#1e3a8a", font=("Microsoft YaHei", 11, "bold"))
    style.configure("Card.TLabelframe.Label", background="#f1f5f9",
                    foreground="#1e3a8a", font=("Microsoft YaHei", 11, "bold"))
    style.configure("TButton", font=("Microsoft YaHei", 10, "bold"),
                    padding=(10, 8), foreground="#0f172a", background="#38bdf8")
    style.map("TButton", background=[("active", "#7dd3fc"), ("pressed", "#0284c7")])

    # 左侧控制面板
    LEFT_W = 290
    left_frame = ttk.Frame(root, width=LEFT_W, style="Dark.TFrame")
    left_frame.pack(side="left", fill="y", padx=0, pady=0)
    left_frame.pack_propagate(False)

    bg_canvas = tk.Canvas(left_frame, width=LEFT_W, height=860,
                          highlightthickness=0, background="#08142a")
    bg_canvas.pack(fill="both", expand=True)

    _CARRIER_BG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "carrier_bg.jpg"
    )

    try:
        if os.path.exists(_CARRIER_BG_PATH):
            _bg_img_pil = Image.open(_CARRIER_BG_PATH).convert("RGB")
            scale = max(LEFT_W / _bg_img_pil.width, 860 / _bg_img_pil.height)
            nw = int(_bg_img_pil.width * scale) + 1
            nh = int(_bg_img_pil.height * scale) + 1
            _bg_img_pil = _bg_img_pil.resize((nw, nh), Image.LANCZOS)
            left = (nw - LEFT_W) // 2
            top = (nh - 860) // 2
            _bg_img_pil = _bg_img_pil.crop((left, top, left + LEFT_W, top + 860))
            _bg_img_pil = _bg_img_pil.filter(ImageFilter.GaussianBlur(radius=5))
            overlay = Image.new("RGB", (LEFT_W, 860), (8, 20, 46))
            _bg_img_pil = Image.blend(_bg_img_pil, overlay, alpha=0.65)
            bg_canvas._photo = ImageTk.PhotoImage(_bg_img_pil)
        else:
            raise FileNotFoundError("未配置航母背景图")
    except Exception:
        grad = Image.new("RGB", (LEFT_W, 860), (8, 20, 46))
        d = ImageDraw.Draw(grad)
        for y in range(860):
            r = int(8 + 18 * y / 860)
            g = int(20 + 38 * y / 860)
            b = int(46 + 64 * y / 860)
            d.line([(0, y), (LEFT_W, y)], fill=(r, g, b))
        bg_canvas._photo = ImageTk.PhotoImage(grad)

    bg_canvas.create_image(0, 0, image=bg_canvas._photo, anchor="nw")
    bg_canvas.create_rectangle(0, 0, LEFT_W, 68, fill="#08142a", outline="", stipple="gray25")
    bg_canvas.create_text(LEFT_W // 2, 24, text="风场反演系统",
                          font=("Microsoft YaHei", 17, "bold"), fill="#e0f2fe")
    bg_canvas.create_text(LEFT_W // 2, 48, text="三维风场高精度反演平台",
                          font=("Microsoft YaHei", 10), fill="#7dd3fc")

    # 信息面板
    info_frame = ttk.Frame(left_frame, style="Dark.TFrame")
    info_frame.place(x=14, y=82, width=LEFT_W - 28, height=230)
    ttk.Label(info_frame, text="▎系统运行状态", font=("Microsoft YaHei", 12, "bold"),
               foreground="#7dd3fc", background="#0c1a30", anchor="w").pack(anchor="w")
    info_grid = ttk.Frame(info_frame, style="Dark.TFrame")
    info_grid.pack(fill="x", pady=8)

    _status = {}
    _info_rows = [
        ("运行模式", "推理模式" if INFERENCE_ONLY_MODE else "开发调试模式", "mode"),
        ("数据状态", "未加载", "data"),
        ("训练集/测试集", "— / —", "split"),
        ("Linear模型", "未训练", "linear"),
        ("ResNet模型", "未训练", "resnet"),
    ]

    for i, (k, v, key) in enumerate(_info_rows):
        ttk.Label(info_grid, text=k + "：", font=("Microsoft YaHei", 9),
                   foreground="#94a3b8", background="#0c1a30", anchor="w"
                   ).grid(row=i, column=0, sticky="w", padx=(0, 6), pady=4)
        lbl = ttk.Label(info_grid, text=v, font=("Consolas", 9, "bold"),
                         foreground="#e2e8f0", background="#0c1a30", anchor="w")
        lbl.grid(row=i, column=1, sticky="w", pady=4)
        _status[key] = lbl

    def _update_status(data_loaded=False, train_rows=None, test_rows=None,
                        linear_trained=False, resnet_trained=False):
        if data_loaded:
            _status["data"].configure(text="已加载", foreground="#4ade80")
        n_train = train_rows if train_rows is not None else "—"
        n_test = test_rows if test_rows is not None else "—"
        _status["split"].configure(text=f"{n_train} / {n_test}")
        if linear_trained:
            _status["linear"].configure(text="已训练", foreground="#4ade80")
        if resnet_trained:
            _status["resnet"].configure(text="已训练", foreground="#4ade80")

    _canvas_ref["update_status"] = _update_status

    btn_frame = ttk.Frame(left_frame, style="Dark.TFrame")
    btn_frame.place(x=14, y=330, width=LEFT_W - 28)

    # 右侧内容面板
    right_frame = ttk.Frame(root, style="Card.TFrame")
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    log_outer = ttk.LabelFrame(right_frame, text="运行日志", padding=5, style="Card.TLabelframe")
    log_outer.pack(fill="both", expand=True, side="top")
    log_text = scrolledtext.ScrolledText(log_outer, height=18, wrap="word",
                                          font=("Consolas", 10), background="#0f172a",
                                          foreground="#e2e8f0", insertbackground="#38bdf8")
    log_text.pack(fill="both", expand=True)

    fig_frame = ttk.LabelFrame(right_frame, text="模型反演对比图", padding=5, style="Card.TLabelframe")
    fig_frame.pack(fill="both", expand=True, pady=(8, 0))
    fig = plt.Figure(figsize=(10, 5), dpi=100)
    fig.set_facecolor("#ffffff")
    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    _canvas_ref["canvas"] = canvas
    _canvas_ref["fig"] = fig

    def log(msg, color="white"):
        s = str(msg)
        if "array(" in s[:80] or "shape=(" in s[:80] or "object at 0x" in s:
            return
        if s.strip().startswith("[") and "shape=" in s:
            return
        if s.strip().startswith("'") and s.strip().endswith("',"):
            return
        tag = "white"
        if color == "green":
            tag = "green"
        elif color == "red":
            tag = "red"
        elif color == "blue":
            tag = "blue"
        log_text.insert("end", s.rstrip() + "\n", tag)
        log_text.see("end")
        root.update_idletasks()

    def safe_run(fn, success_msg):
        try:
            result = fn()
            if result:
                log(str(result), "green")
            else:
                log(success_msg, "green")
        except Exception as e:
            log(f"错误: {e}", "red")

    # 按钮1
    def load_and_update_status():
        out = step_load_and_preprocess()
        _update_status(
            data_loaded=True,
            train_rows=int(out["X_train"].shape[0]) if out.get("X_train") is not None else None,
            test_rows=int(out["X_test"].shape[0]) if out.get("X_test") is not None else None,
        )
        log(f"数据读取完成:{int(out['X_raw'].shape[0])}行 × {int(out['X_raw'].shape[1])}列", "green")
        log(f"训练集:{int(out['X_train'].shape[0])}条，测试集:{int(out['X_test'].shape[0])}条", "green")

    ttk.Button(btn_frame, text="1. 多源数据的读取与预处理",
               command=lambda: safe_run(load_and_update_status, "数据读取与预处理完成")
               ).pack(fill="x", pady=5)

    # 按钮2
    def infer_and_update_status():
        step_inference(train_models=not INFERENCE_ONLY_MODE)
        _update_status(
            linear_trained=models.get("linear") is not None,
            resnet_trained=models.get("resnet") is not None,
        )

    ttk.Button(btn_frame, text="2. 三维风场高精度反演",
               command=lambda: safe_run(infer_and_update_status, "三维风场高精度反演完成")
               ).pack(fill="x", pady=5)

    # 按钮3
    def evaluate_and_plot():
        try:
            step_evaluate()
            result = step_plot(_canvas_ref["canvas"], _canvas_ref["fig"])
            if result:
                log(str(result), "green")
        except Exception as e:
            log(f"错误:{e}", "red")

    ttk.Button(btn_frame, text="3. 预测与评估",
               command=evaluate_and_plot).pack(fill="x", pady=5)

    # 按钮4
    def train_and_update_status():
        step_inference(train_models=True)
        _update_status(
            linear_trained=models.get("linear") is not None,
            resnet_trained=models.get("resnet") is not None,
        )

    ttk.Button(btn_frame, text="4. 训练并保存模型",
               command=lambda: safe_run(train_and_update_status, "模型训练并保存完成")
               ).pack(fill="x", pady=5)

    # 按钮5
    def auto_all_wrapper():
        run_auto_all(_canvas_ref["canvas"], _canvas_ref["fig"])
        _update_status(
            data_loaded=True,
            train_rows=int(data_store["X_train"].shape[0]) if data_store.get("X_train") is not None else None,
            test_rows=int(data_store["X_test"].shape[0]) if data_store.get("X_test") is not None else None,
            linear_trained=models.get("linear") is not None,
            resnet_trained=models.get("resnet") is not None,
        )

    ttk.Button(btn_frame, text="5. 一键自动全流程",
               command=lambda: safe_run(auto_all_wrapper, "一键自动全流程完成")
               ).pack(fill="x", pady=(20, 5))

    # 按钮7/8 已隐藏（取消注释可恢复）
    # ttk.Button(left_frame, text="7. 训练集预测对比图",
    #            command=lambda: safe_run(lambda: step_plot_training_set(_canvas_ref["canvas"], _canvas_ref["fig"]), "训练集绘图完成")
    #            ).pack(fill="x", pady=4)
    # ttk.Button(left_frame, text="8. 训练损失曲线图",
    #            command=lambda: safe_run(lambda: step_plot_loss_curves(_canvas_ref["canvas"], _canvas_ref["fig"]), "损失曲线绘图完成")
    #            ).pack(fill="x", pady=4)

    log_text.tag_configure("green", foreground="#4ade80")
    log_text.tag_configure("red", foreground="#f87171")
    log_text.tag_configure("blue", foreground="#60a5fa")
    log_text.tag_configure("white", foreground="#e2e8f0")

    log("系统已就绪，点击按钮开始操作。", "blue")
    log(f"当前模式:{'推理模式' if INFERENCE_ONLY_MODE else '开发调试模式'}", "blue")
    log("提示:首次使用请先点击『训练并保存模型』或『一键自动全流程』", "blue")
    log("提示:按钮7/8(训练集可视化/损失曲线)已隐藏，如需调试可取消注释恢复", "blue")

    root.mainloop()


# ===================== 12. 程序入口 =====================

if __name__ == "__main__":
    main()