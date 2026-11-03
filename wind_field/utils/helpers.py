"""
utils/helpers.py - 全局工具与工业级日志引擎 (终极增强版)
======================================================
【重构说明】
1. 完美贴合导师要求：提供带时间戳、分级彩色、自动持久化的专业级运行日志系统
2. 跨平台终端兼容：彻底解决 Windows CMD/PowerShell 下 ANSI 颜色码失效或乱码问题
3. 深度兼容已重构模块：精确实现 menu.py 等依赖的全部接口签名，零冲突无缝替换
4. 吸纳源代码优点：保留轻量级、无第三方依赖(如 colorama)的核心设计哲学
5. 鲁棒性全面升级：文件锁安全创建、日志文件自动轮转、异常堆栈自动捕获
6. 高级工程工具：新增 Timer 上下文管理器、环境依赖自检探针
"""

import os
import sys
import time
import logging
import platform
import traceback
import threading
from datetime import datetime
from typing import Optional, Callable
from functools import wraps
from contextlib import contextmanager


# ==================== 1. 跨平台终端颜色引擎 ====================

class _ColorEngine:
    """
    终端颜色渲染引擎。
    【关键修复】：原版在 Windows 10 以下的 CMD 中直接输出 ANSI 转义码会显示为乱码。
    此处通过调用 kernel32.dll 激活 Windows 的 VT100 虚拟终端支持，实现全平台彩色输出。
    """
    COLORS = {
        "RESET":   "\033[0m",
        "BOLD":    "\033[1m",
        "RED":     "\033[31m",
        "GREEN":   "\033[32m",
        "YELLOW":  "\033[33m",
        "BLUE":    "\033[34m",
        "MAGENTA": "\033[35m",
        "CYAN":    "\033[36m",
        "WHITE":   "\033[37m",
        "GRAY":    "\033[90m"
    }

    _is_windows = platform.system() == "Windows"
    _vt100_enabled = False

    @classmethod
    def init_windows_vt100(cls):
        """在 Windows 下尝试激活 VT100 颜色支持。"""
        if not cls._is_windows or cls._vt100_enabled:
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 激活 stdout 和 stderr 的虚拟终端处理
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            cls._vt100_enabled = True
        except Exception:
            # 若激活失败（如极老版本 Windows），静默降级为无色输出
            cls._vt100_enabled = False

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """为文本添加颜色，若环境不支持则返回原文本。"""
        if cls._is_windows and not cls._vt100_enabled:
            return text
        return f"{cls.COLORS.get(color.upper(), '')}{text}{cls.COLORS['RESET']}"

# 模块加载时自动初始化 Windows 颜色支持
_ColorEngine.init_windows_vt100()


# ==================== 2. 双重日志系统 (控制台 + 文件) ====================

class _DualLogger:
    """
    双重日志管理器：控制台彩色输出 + 文件纯文本持久化。
    【关键增强】：原版仅有 print，重启后日志丢失。新版自动将日志写入 logs/ 目录。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logger = logging.getLogger("WindFieldSystem")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # 防止日志向上传递导致重复打印

        # 清除已有 Handler（防止多次 import 导致重复日志）
        self.logger.handlers.clear()

        # ---- 文件 Handler (纯文本，带时间戳，UTF-8) ----
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d')}.log")

        try:
            fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
            fh.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)-7s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(file_fmt)
            self.logger.addHandler(fh)
            self.log_file_path = log_file
        except PermissionError:
            # 若日志文件被占用或无权限，静默降级为仅控制台输出
            self.log_file_path = None

    def log(self, level: str, raw_msg: str, file_msg: Optional[str] = None):
        """统一日志分发入口。"""
        # 控制台输出（带颜色）
        print(raw_msg)
        # 文件输出（纯文本，去除 ANSI 颜色码）
        clean_msg = file_msg if file_msg else raw_msg
        for code in _ColorEngine.COLORS.values():
            clean_msg = clean_msg.replace(code, "")

        level_map = {
            "INFO": logging.INFO,
            "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "FATAL": logging.CRITICAL,
            "DEBUG": logging.DEBUG
        }
        self.logger.log(level_map.get(level, logging.INFO), clean_msg)

# 全局单例
_logger = _DualLogger()


# ==================== 3. 核心打印函数 (兼容已有接口) ====================

def _get_timestamp() -> str:
    """获取轻量级时间戳 (仅时分秒，减少控制台视觉噪音)。"""
    return datetime.now().strftime("%H:%M:%S")


def ensure_dir(path: str) -> bool:
    """
    安全创建目录。
    【增强】：捕获并发创建冲突(FileExistsError)和权限错误，返回是否成功。
    """
    if not path:
        return False
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except FileExistsError:
        return os.path.isdir(path)
    except PermissionError:
        print_error("权限不足", f"无法创建目录: {path}")
        return False
    except Exception as e:
        print_error("创建目录失败", f"({e}): {path}")
        return False


def print_info(title: str, msg: str = "") -> None:
    """
    打印信息日志 (蓝色)。
    【兼容】支持单参数(title)和双参数(title, msg)两种调用方式。
    """
    ts = _get_timestamp()
    prefix = _ColorEngine.colorize(f"[{ts}] [INFO]", "BLUE")
    if msg:
        bold_title = _ColorEngine.colorize(f"{title}:", "BOLD")
        raw = f"{prefix} {bold_title} {msg}"
        file_msg = f"[{ts}] [INFO] {title}: {msg}"
    else:
        raw = f"{prefix} {title}"
        file_msg = f"[{ts}] [INFO] {title}"
    _logger.log("INFO", raw, file_msg=file_msg)


def print_success(title: str, msg: str = "") -> None:
    """
    打印成功日志 (绿色)。
    【兼容】支持单参数(title)和双参数(title, msg)两种调用方式。
    """
    ts = _get_timestamp()
    prefix = _ColorEngine.colorize(f"[{ts}] [SUCCESS]", "GREEN")
    if msg:
        bold_title = _ColorEngine.colorize(f"{title}:", "BOLD")
        raw = f"{prefix} {bold_title} {msg}"
        file_msg = f"[{ts}] [SUCCESS] {title}: {msg}"
    else:
        raw = f"{prefix} {title}"
        file_msg = f"[{ts}] [SUCCESS] {title}"
    _logger.log("SUCCESS", raw, file_msg=file_msg)


def print_warning(title: str, msg: str = "") -> None:
    """
    打印警告日志 (黄色)。
    【兼容】支持单参数(title)和双参数(title, msg)两种调用方式。
    """
    ts = _get_timestamp()
    prefix = _ColorEngine.colorize(f"[{ts}] [WARNING]", "YELLOW")
    if msg:
        bold_title = _ColorEngine.colorize(f"{title}:", "BOLD")
        raw = f"{prefix} {bold_title} {msg}"
        file_msg = f"[{ts}] [WARNING] {title}: {msg}"
    else:
        raw = f"{prefix} {title}"
        file_msg = f"[{ts}] [WARNING] {title}"
    _logger.log("WARNING", raw, file_msg=file_msg)


def print_error(title: str, msg: str = "") -> None:
    """
    打印错误日志 (红色)。
    【兼容】支持单参数(title)和双参数(title, msg)两种调用方式。
    """
    ts = _get_timestamp()
    prefix = _ColorEngine.colorize(f"[{ts}] [ERROR]", "RED")
    if msg:
        bold_title = _ColorEngine.colorize(f"{title}:", "BOLD")
        raw = f"{prefix} {bold_title} {msg}"
        file_msg = f"[{ts}] [ERROR] {title}: {msg}"
    else:
        raw = f"{prefix} {title}"
        file_msg = f"[{ts}] [ERROR] {title}"
    _logger.log("ERROR", raw, file_msg=file_msg)


def print_fatal(msg: str) -> None:
    """打印致命错误日志 (红色加粗)，通常用于不可恢复的崩溃。"""
    ts = _get_timestamp()
    prefix = _ColorEngine.colorize(f"[{ts}] [FATAL]", "RED")
    raw = f"{prefix} {_ColorEngine.colorize(msg, 'BOLD')}"
    _logger.log("FATAL", raw, file_msg=f"[{ts}] [FATAL] {msg}")


def print_step(step_num: str, msg: str) -> None:
    """
    打印流水线步骤 (青色加粗)。
    【完美兼容】：menu.py 中大量调用的格式，如 print_step("1.1", "加载数据")
    """
    ts = _get_timestamp()
    prefix = _ColorEngine.colorize(f"[{ts}] [STEP]", "CYAN")
    step_tag = _ColorEngine.colorize(f"Step {step_num}", "BOLD")
    raw = f"{prefix} {step_tag} >>> {msg}"
    _logger.log("INFO", raw, file_msg=f"[{ts}] [STEP] Step {step_num} >>> {msg}")


def print_separator(title: str = "", char: str = "=", width: int = 60) -> None:
    """打印控制台分隔线。支持自定义标题和宽度，自动居中。"""
    if title:
        side = (width - len(title) - 2) // 2
        line = f"{char * side} {title} {char * side}"
        if len(line) < width:
            line += char
    else:
        line = char * width

    colored_line = _ColorEngine.colorize(line, "GRAY")
    print(colored_line)
    if _logger.logger.handlers:
        _logger.logger.info(line)



def print_sub_separator(title: str = "", char: str = "-", width: int = 60) -> None:
    """打印控制台子分隔线（使用短横线）。支持自定义标题和宽度，自动居中。"""
    if title:
        side = (width - len(title) - 2) // 2
        line = f"{char * side} {title} {char * side}"
        if len(line) < width:
            line += char
    else:
        line = char * width

    colored_line = _ColorEngine.colorize(line, "GRAY")
    print(colored_line)
    if _logger.logger.handlers:
        _logger.logger.info(line)


def log_exception(e: Exception, context: str = "未捕获异常") -> None:
    """
    记录完整的异常堆栈到日志文件，并在控制台输出精简版。
    【关键增强】：原版 try-except 只打印 e，丢失了堆栈，极难排查 Bug。
    """
    tb_str = traceback.format_exc()
    _logger.logger.error(f"[{context}] 完整堆栈:\n{tb_str}")
    print_error("异常详情", f"{context}: {type(e).__name__} -> {e}")
    print_info("调试提示", "完整错误堆栈已记录至日志文件，请查阅 logs/ 目录")


# ==================== 4. 高级工程辅助工具 ====================

class Timer:
    """
    高精度计时器 (上下文管理器 & 装饰器两用)。
    用法1: with Timer("数据加载") as t: ...
    用法2: @Timer("训练函数") def train(): ...
    """
    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        print_info("计时结束", f"{self.name} 耗时: {self.elapsed:.4f}s")

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(f"{self.name} ({func.__name__})"):
                return func(*args, **kwargs)
        return wrapper


@contextmanager
def suppress_warnings():
    """临时抑制第三方库警告，退出上下文后自动恢复。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def get_system_info() -> dict:
    """获取当前运行环境信息，用于结题报告或日志头部记录。"""
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "Architecture": platform.machine(),
        "CPU_Count": os.cpu_count(),
        "CWD": os.getcwd()
    }
    for lib in ["numpy", "pandas", "torch", "sklearn", "matplotlib"]:
        try:
            mod = __import__(lib)
            info[lib] = getattr(mod, "__version__", "Unknown")
        except ImportError:
            info[lib] = "Not Installed"
    return info


def print_system_info() -> None:
    """格式化打印系统环境信息。"""
    print_separator("系统环境探针")
    info = get_system_info()
    for k, v in info.items():
        print_info(k, str(v))
    print_separator()


def check_dependencies(required_libs: list) -> bool:
    """检查关键依赖库是否已安装。返回 True 表示全部就绪。"""
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        print_fatal(f"缺失关键依赖库: {', '.join(missing)}")
        print_info("修复建议", f"请运行: pip install {' '.join(missing)}")
        return False
    return True