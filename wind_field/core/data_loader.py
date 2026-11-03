import numpy as np
import os
import pandas as pd
import chardet
from sklearn.preprocessing import StandardScaler
import json
import csv
from utils.helpers import print_info, print_success, print_error, print_warning

from config.settings import DATA_DIR, MODEL_PARAMS_DIR


def _detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    confidence = result['confidence']
    if confidence < 0.7:
        print_warning(f"文件 {file_path} 编码检测置信度为 {confidence:.2f}，默认为 utf-8")
        return 'utf-8'
    return encoding


def _detect_separator(file_path):
    """自动检测文件分隔符"""
    separators = [',', '\t', ' ', ';']
    sample_lines = []
    encoding = _detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            sample_lines.append(line.strip())
    
    sep_counts = {}
    for sep in separators:
        if sep == ' ':
            count = sum(1 for line in sample_lines if len(line.split()) > 1)
            sep_counts[sep] = count
        elif sep == '\t':
            count = sum(line.count('\t') for line in sample_lines)
            sep_counts[sep] = count
        else:
            count = sum(line.count(sep) for line in sample_lines)
            sep_counts[sep] = count
    
    non_space_seps = {k: v for k, v in sep_counts.items() if k != ' '}
    best_non_space = max(non_space_seps, key=non_space_seps.get) if any(v > 0 for v in non_space_seps.values()) else None
    if best_non_space and sep_counts[best_non_space] > 0:
        return best_non_space
    
    if sep_counts[' '] > 0:
        return ' '
    
    return ','


def _is_numeric(value):
    """判断一个值是否为数值类型"""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _safe_read_single_file(file_path):
    """安全读取单个数据文件"""
    try:
        data = np.genfromtxt(file_path, dtype=float, delimiter=None, skip_header=0)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception as e:
        print_warning(f"标准读取方式失败: {e}，尝试逐行解析...")
        return _fallback_line_by_line(file_path)


def _fallback_line_by_line(file_path):
    """兜底方案：逐行读取并解析数值"""
    encoding = _detect_encoding(file_path)
    
    data = []
    discarded_count = 0
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            numeric_row = []
            for item in row:
                item = item.strip()
                if _is_numeric(item):
                    numeric_row.append(float(item))
            if numeric_row:
                data.append(numeric_row)
            else:
                discarded_count += 1
    
    if not data:
        raise ValueError(f"无法从文件中提取任何有效数值数据: {file_path}")
        
    max_cols = max(len(row) for row in data)
    for i in range(len(data)):
        if len(data[i]) < max_cols:
            data[i].extend([0.0] * (max_cols - len(data[i])))
        elif len(data[i]) > max_cols:
            data[i] = data[i][:max_cols]
    
    if discarded_count > 0:
        print_warning(f"Fallback解析器丢弃了 {discarded_count} 行非数值内容（可能是表头或注释行）。请检查数据文件格式。")
            
    return np.array(data)


def load_data(x_path=None, y_path=None):
    """
    加载数据。
    支持传入单个文件路径，或者包含多个文件的目录路径。
    如果是目录，会自动扫描并合并所有 .txt 和 .csv 文件。
    """
    if x_path is None:
        x_path = os.path.join(DATA_DIR, 'X_data.txt')
    if y_path is None:
        y_path = os.path.join(DATA_DIR, 'y_data.txt')
        
    def _get_data(source):
        if os.path.isdir(source):
            files = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(('.txt', '.csv'))])
            if not files:
                raise FileNotFoundError(f"在目录 {source} 中未找到 .txt 或 .csv 文件")
            data_list = [_safe_read_single_file(f) for f in files]
            return np.concatenate(data_list, axis=0)
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"数据文件不存在: {source}")
            return _safe_read_single_file(source)
            
    try:
        X = _get_data(x_path)
        y = _get_data(y_path)
        
        # NaN/Inf 数据清洗
        nan_mask_x = np.isnan(X) | np.isinf(X)
        nan_mask_y = np.isnan(y) | np.isinf(y)
        
        if nan_mask_x.any():
            nan_cols_x = np.where(nan_mask_x.any(axis=0))[0]
            for col in nan_cols_x:
                col_data = X[:, col]
                valid_mask = ~np.isnan(col_data) & ~np.isinf(col_data)
                fill_val = np.nanmean(col_data[valid_mask]) if valid_mask.any() else 0.0
                col_data[~valid_mask] = fill_val
                X[:, col] = col_data
            print_warning(f"X 数据中存在 NaN/Inf 异常值，已按列均值填充，涉及列: {list(nan_cols_x)}")
        
        if nan_mask_y.any():
            nan_cols_y = np.where(nan_mask_y.any(axis=0))[0]
            for col in nan_cols_y:
                col_data = y[:, col]
                valid_mask = ~np.isnan(col_data) & ~np.isinf(col_data)
                fill_val = np.nanmean(col_data[valid_mask]) if valid_mask.any() else 0.0
                col_data[~valid_mask] = fill_val
                y[:, col] = col_data
            print_warning(f"y 数据中存在 NaN/Inf 异常值，已按列均值填充，涉及列: {list(nan_cols_y)}")
        
        # 行数对齐检查
        if X.shape[0] != y.shape[0]:
            print_warning(f"X 和 y 的行数不一致 (X: {X.shape[0]}, y: {y.shape[0]})，将截断至相同长度")
            min_rows = min(X.shape[0], y.shape[0])
            X = X[:min_rows]
            y = y[:min_rows]
        
        # 【双保险】统一数据类型为 float32，与 main.py 子步骤1.5 对齐
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        print_success(f"数据加载成功: X shape={X.shape}, y shape={y.shape}")
        return X, y
        
    except Exception as e:
        print_error(f"数据加载失败: {e}")
        raise


class WindFieldScaler:
    """
    自定义风场标准化器。
    """
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y) if y is not None else None
        if X_arr.size == 0:
            raise ValueError("fit: 输入X数据为空，无法拟合标准化器")
        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValueError("fit: X数据包含NaN或无穷大，请先清洗数据")
        self.scaler_X.fit(X_arr)
        if y_arr is not None:
            y_arr = np.asarray(y_arr)
            if y_arr.size == 0:
                raise ValueError("fit: 输入y数据为空，无法拟合标准化器")
            if np.isnan(y_arr).any() or np.isinf(y_arr).any():
                raise ValueError("fit: y数据包含NaN或无穷大，请先清洗数据")
            self.scaler_y.fit(y_arr)
        self.is_fitted = True
        print_info("标准化", "标准化器拟合完成。")

    def transform(self, X, y=None):
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 方法拟合数据！")
        X_scaled = self.scaler_X.transform(X)
        if y is not None:
            y_scaled = self.scaler_y.transform(y)
            return X_scaled, y_scaled
        return X_scaled

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 方法拟合数据！")
        
        X = np.array(X)
        if X.ndim == 0 or X.ndim == 1:
            X = X.reshape(1, -1)
        
        if np.isnan(X).any() or np.isinf(X).any():
            print_error("警告：传入的预测数据包含 NaN 或无穷大，无法进行物理值还原")
            print_warning("请检查模型训练是否异常（梯度爆炸/学习率过大），或重新训练模型")
            # 【修复】当 y 不为 None 时，返回 (X, y) 元组以保持接口一致性，防止调用方解包错误
            if y is not None:
                return X, y
            return X
        
        X_orig = self.scaler_X.inverse_transform(X)
        
        if y is not None:
            y = np.array(y)
            if y.ndim == 0 or y.ndim == 1:
                y = y.reshape(1, -1)
            if np.isnan(y).any() or np.isinf(y).any():
                print_error("警告：传入的 y 数据包含 NaN 或无穷大")
                return X_orig, y
            y_orig = self.scaler_y.inverse_transform(y)
            return X_orig, y_orig
        
        return X_orig

    def save_params(self):
        if not self.is_fitted:
            raise ValueError("请先拟合数据！")
            
        params = {
            "X_mean": self.scaler_X.mean_.tolist(),
            "X_scale": self.scaler_X.scale_.tolist(),
            "y_mean": self.scaler_y.mean_.tolist(),
            "y_scale": self.scaler_y.scale_.tolist()
        }
        
        os.makedirs(MODEL_PARAMS_DIR, exist_ok=True)
        save_path = os.path.join(MODEL_PARAMS_DIR, "scaler_params.json")
        
        with open(save_path, 'w') as f:
            json.dump(params, f)
        print_success(f"标准化参数已保存至: {save_path}")

    @staticmethod
    def load_params():
        save_path = os.path.join(MODEL_PARAMS_DIR, "scaler_params.json")
        if not os.path.exists(save_path):
            print_error(f"未找到标准化参数文件: {save_path}，请先运行模型训练。")
            return None
            
        try:
            with open(save_path, 'r') as f:
                params = json.load(f)
            
            actual_features = len(params["X_mean"])
            try:
                from config.settings import N_FEATURES as CFG_N_FEATURES
                if actual_features != CFG_N_FEATURES:
                    print_error(
                        f"标准化参数特征数不匹配！当前系统配置需要 {CFG_N_FEATURES} 个特征，"
                        f"但加载到的参数包含 {actual_features} 个特征。"
                    )
                    print_warning("请重新运行模型训练以生成正确的标准化参数。")
                    return None
            except ImportError:
                print_warning("无法导入 N_FEATURES 配置，跳过特征数校验")

            scaler = WindFieldScaler()
            scaler.scaler_X.mean_ = np.array(params["X_mean"])
            scaler.scaler_X.scale_ = np.array(params["X_scale"])
            if np.any(scaler.scaler_X.scale_ <= 0):
                print_error("X_scale 中存在非正值，标准化参数可能已损坏。请重新运行模型训练。")
                return None
            scaler.scaler_X.is_fitted = True
            scaler.scaler_y.mean_ = np.array(params["y_mean"])
            scaler.scaler_y.scale_ = np.array(params["y_scale"])
            if np.any(scaler.scaler_y.scale_ <= 0):
                print_error("y_scale 中存在非正值，标准化参数可能已损坏。请重新运行模型训练。")
                return None
            scaler.scaler_y.is_fitted = True
            scaler.is_fitted = True
            
            print_success(f"标准化参数已从 {save_path} 加载。")
            return scaler
        except Exception as e:
            print_error(f"加载标准化参数失败: {e}")
            return None

    def inverse_transform_X(self, X_scaled):
        """仅对X做反标准化，还原真实物理值"""
        return self.inverse_transform(X_scaled)

    def inverse_transform_y(self, y_scaled):
        """仅对y(标签)数据做反标准化，还原真实物理值。"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit() 方法拟合数据！")
        y_arr = np.array(y_scaled)
        if y_arr.ndim == 0 or y_arr.ndim == 1:
            y_arr = y_arr.reshape(1, -1)
        if np.isnan(y_arr).any() or np.isinf(y_arr).any():
            print_error("警告：传入的 y 数据包含 NaN 或无穷大")
            return y_arr
        return self.scaler_y.inverse_transform(y_arr)


def load_scaler_params():
    """
    供外部调用的标准化参数加载接口。
    """
    return WindFieldScaler.load_params()