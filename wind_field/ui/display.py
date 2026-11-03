import os
import numpy as np  # 【修复】补充缺失的 numpy 导入，供下方 np.isnan 使用
from config.settings import TERMINAL_WIDTH, SEPARATOR_CHAR, SUB_SEPARATOR_CHAR

def print_menu(options):
    title = "操作菜单"
    # 退一步：直接用字符数 * 2 估算显示宽度（适配大多数中文终端）
    title_width = len(title.encode('utf-8')) 
    display_width = len(title) * 2
    padding = max(0, (TERMINAL_WIDTH - display_width) // 2)
    
    print(f"\n{SEPARATOR_CHAR * TERMINAL_WIDTH}")
    print(f"{' ' * padding}{title}")
    print(f"{SUB_SEPARATOR_CHAR * TERMINAL_WIDTH}")
    for key, (name, desc) in options.items():
        print(f"  [{key}] {name:<18} | {desc}")
    print(f"{SUB_SEPARATOR_CHAR * TERMINAL_WIDTH}")
    print(f"  [a/q] 退出程序")
    print(f"{SEPARATOR_CHAR * TERMINAL_WIDTH}")

def print_model_results(results_dict):
    """打印模型评估结果对比表"""
    print(f"\n{SUB_SEPARATOR_CHAR * TERMINAL_WIDTH}")
    print(f"  {'模型名称':<15} | {'MSE':<10} | {'RMSE':<10} | {'MAE':<10} | {'R² Score':<10}")
    print(f"  {'-' * 70}")
    
    for name, metrics in results_dict.items():
        # 【修复】数值格式化：float 保留4位小数，非数值类型原样输出
        mse = metrics.get("MSE", "-")
        rmse = metrics.get("RMSE", "-")
        mae = metrics.get("MAE", "-")
        r2 = metrics.get("R^2", "-")
        
        # 【修复】拆分长行，避免编辑器截断报错，提升可读性
        mse_str = f"{mse:.4f}" if isinstance(mse, (int, float)) and not np.isnan(mse) else str(mse)
        rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) and not np.isnan(rmse) else str(rmse)
        mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) and not np.isnan(mae) else str(mae)
        r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) and not np.isnan(r2) else str(r2)
        
        print(f"  {name:<15} | {mse_str:<10} | {rmse_str:<10} | {mae_str:<10} | {r2_str:<10}")
        
    print(f"{SUB_SEPARATOR_CHAR * TERMINAL_WIDTH}\n")

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')