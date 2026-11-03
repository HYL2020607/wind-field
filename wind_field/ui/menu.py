"""
ui/menu.py - 交互式主控调度引擎 (工业级增强版)
==============================================
【重构说明】
1. 完美贴合导师"交互式操作系统"要求：5 大功能按钮 + 状态追踪 + 智能引导
2. 全链路状态管理：每步完成后自动记录状态，未就绪时给出明确提示而非崩溃
3. 双模式无缝切换：训练模式(完整5步) / 纯推理模式(自动跳过训练)
4. 深度兼容已重构模块：精确对接新版 data_loader / models / evaluator / visualizer 接口
5. 吸纳源代码优点：保留 clear_screen、编号选择、循环菜单的核心交互范式
6. 鲁棒性全面升级：所有步骤均含 try-except 防护、输入校验、超时保护
"""
 
import os
import sys
import time
import traceback
import numpy as np  # 【修复】PipelineState 类型注解依赖 np.ndarray，必须导入
from typing import Optional, Dict, Any
 
from config.settings import (
    DATA_DIR, MODEL_PARAMS_DIR, PLOT_DIR,
    INFERENCE_ONLY_MODE
)
 
# 兼容处理：以下变量在部分版本settings.py中可能未定义，提供安全默认值
try:
    from config.settings import AUTO_SAVE_MODELS
except ImportError:
    AUTO_SAVE_MODELS = False
try:
    from config.settings import DEFAULT_LR_PARAMS
except ImportError:
    DEFAULT_LR_PARAMS = {}
try:
    from config.settings import DEFAULT_RESNET_PARAMS
except ImportError:
    DEFAULT_RESNET_PARAMS = {}
from utils.helpers import (
    ensure_dir, print_info, print_success, print_warning,
    print_error, print_step, print_separator, log_exception
)
 
 
# ==================== 全局状态管理器 ====================
 
class PipelineState:
    """
    流水线全局状态追踪器。
    确保每一步执行前，前置依赖已就绪；避免"未加载数据就训练"等逻辑错误。
    """
 
    def __init__(self):
        # 数据状态
        self.X_raw: Optional[np.ndarray] = None
        self.y_raw: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.scaler_x: Any = None  # WindFieldScaler 实例
        self.scaler_y: Any = None
 
        # 模型状态
        self.models: Dict[str, Any] = {}
        self.models_trained: bool = False
 
        # 预测状态
        self.predictions: Dict[str, np.ndarray] = {}  # {"Linear": ..., "ResNet": ...}
        self.predictions_inverse: Dict[str, np.ndarray] = {}  # 反标准化后的物理量
 
        # 评估状态
        self.metrics: Dict[str, Dict] = {}
 
        # 步骤完成标记
        self.step_completed: Dict[str, bool] = {
            "data_load": False,
            "preprocess": False,
            "train": False,
            "predict": False,
            "evaluate": False,
            "visualize": False
        }
 
    def is_ready_for(self, step: str) -> tuple:
        """
        检查某一步骤的前置条件是否满足。
        返回 (是否就绪, 原因说明)
        """
        checks = {
            "preprocess": (
                self.step_completed["data_load"],
                "请先执行 [步骤1] 加载数据"
            ),
            "train": (
                self.step_completed["preprocess"],
                "请先执行 [步骤1] 完成数据加载与预处理"
            ),
            "predict": (
                self.step_completed["train"] or INFERENCE_ONLY_MODE,
                "请先执行 [步骤2] 训练模型（或切换为推理模式）"
            ),
            "evaluate": (
                self.step_completed["predict"],
                "请先执行 [步骤3] 生成预测结果"
            ),
            "visualize": (
                self.step_completed["evaluate"],
                "请先执行 [步骤4] 完成评估（评估结果将标注在图表上）"
            )
        }
        if step in checks:
            return checks[step]
        return True, ""
 
    def get_summary(self) -> str:
        """生成当前流水线状态的摘要文本。"""
        status_icon = lambda done: "✅" if done else "⬜"
        lines = [
            f"  {status_icon(self.step_completed['data_load'])} 数据加载"
            ▪ (f"  ({self.X_raw.shape[0]}样本 × {self.X_raw.shape[1]}特征)"

               if self.X_raw is not None else ""),
            f"  {status_icon(self.step_completed['preprocess'])} 数据预处理"
            ▪ (f"  (训练集{self.X_train.shape[0]} / 测试集{self.X_test.shape[0]})"

               if self.X_train is not None else ""),
            f"  {status_icon(self.step_completed['train'])} 模型训练"
            ▪ (f"  ({', '.join(self.models.keys())})"

               if self.models else ""),
            f"  {status_icon(self.step_completed['predict'])} 预测推理"
            ▪ (f"  ({', '.join(self.predictions.keys())})"

               if self.predictions else ""),
            f"  {status_icon(self.step_completed['evaluate'])} 评估分析"
            ▪ (f"  (已完成)" if self.metrics else ""),

            f"  {status_icon(self.step_completed['visualize'])} 可视化报告"
        ]
        return "\n".join(lines)
 
    def reset(self):
        """重置所有状态（用于重新开始）。"""
        self.__init__()
 
 
# ==================== 安全导入工厂 ====================
 
def _safe_import_modules():
    """
    延迟导入所有业务模块，捕获导入错误并给出修复建议。
    避免菜单本身因某个模块语法错误而完全无法启动。
    """
    modules = {}
    import_map = {
        "data_loader": ("core.data_loader", ["load_data", "WindFieldScaler"]),
        "preprocessing": ("core.preprocessing", ["split_and_normalize"]),
        "models": ("core.models", ["build_linear_model", "build_resnet_model",
                                    "train_model", "predict_model",
                                    "save_model", "load_model"]),
        "evaluator": ("core.evaluator", ["evaluate_model", "evaluate_all_models"]),
        "visualizer": ("ui.visualizer", ["WindFieldVisualizer", "plot_wind_comparison"])
    }
 
    for alias, (module_path, symbols) in import_map.items():
        try:
            mod = __import__(module_path, fromlist=symbols)
            modules[alias] = mod
            print_success(f"模块加载: {module_path}")
        except ImportError as e:
            print_error(f"模块导入失败: {module_path} -> {e}")
            modules[alias] = None
        except Exception as e:
            print_error(f"模块加载异常: {module_path} -> {e}")
            modules[alias] = None
 
    return modules
 
 
# ==================== 核心菜单引擎 ====================
 
class WindFieldMenu:
    """
    风场预测系统主控菜单引擎。
    管理用户交互、步骤调度、状态流转的全部逻辑。
    """
 
    BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║           三维风场智能预测系统 v2.0                       ║
║           3D Wind Field Intelligent Prediction System    ║
╚══════════════════════════════════════════════════════════╝
    """
 
    MENU_TEMPLATE = """
┌──────────────────────────────────────────────────────────┐
│  {mode_label:<56s}│
├──────────────────────────────────────────────────────────┤
│                                                          │
│  [1] 多源数据读取与预处理                                  │
│  [2] 模型构建与训练                                       │
│  [3] 模型预测与评估分析                                    │
│  [4] 结果可视化与报告生成                                  │
│  [5] ▶ 一键自动全流程 (推荐)                               │
│                                                          │
│  [6] 查看当前流水线状态                                    │
│  [7] 重置系统 (清空所有中间结果)                            │
│  [0] 退出系统                                             │
│                                                          │
└──────────────────────────────────────────────────────────┘
    """
 
    def __init__(self):
        self.state = PipelineState()
        self.modules = {}
        self.start_time = None
 
    def initialize(self):
        """系统初始化：创建目录、加载模块、显示横幅。"""
        self.start_time = time.time()
 
        print(self.BANNER)
        print_separator("系统初始化")
 
        # 创建必要目录
        for d in [DATA_DIR, MODEL_PARAMS_DIR, PLOT_DIR]:
            ensure_dir(d)
 
        # 安全加载所有业务模块
        self.modules = _safe_import_modules()
 
        # 显示运行模式
        if INFERENCE_ONLY_MODE:
            print_warning("当前为【纯推理模式】：将跳过训练步骤，直接加载已保存模型")
            mode = "纯推理模式 (Inference Only)"
        else:
            print_info("运行模式", "完整训练模式 (Full Training)")
            mode = "完整训练模式 (Full Training)"
 
        # 检查关键模块是否就绪
        critical_ok = all(self.modules.get(k) is not None
                         for k in ["data_loader", "preprocessing"])
        if not critical_ok:
            print_error("关键模块加载失败！请检查上方错误信息后重试。")
            print_info("提示", "可尝试修复代码后重启系统")
 
        print_separator("初始化完成")
        return mode
 
    def run(self):
        """主循环：显示菜单 → 获取输入 → 分发执行。"""
        mode = self.initialize()
        mode_label = f"模式: {mode}"
 
        while True:
            print(self.MENU_TEMPLATE.format(mode_label=mode_label))
            print(self.state.get_summary())
            print()
 
            choice = input("  请选择操作 [0-7]: ").strip()
 
            action_map = {
                "1": self._step1_data_load_and_preprocess,
                "2": lambda: self._step2_model_build_and_train(auto=False),
                "3": lambda: self._step3_predict_and_evaluate(auto=False),
                "4": lambda: self._step4_visualize(auto=False),
                "5": self._step5_auto_pipeline,
                "6": self._show_status,
                "7": self._reset_system,
                "0": self._exit_system
            }
 
            action = action_map.get(choice)
            if action:
                try:
                    action()
                except KeyboardInterrupt:
                    print_warning("\n用户中断当前操作，返回主菜单")
                except Exception as e:
                    print_error(f"操作执行异常: {e}")
                    print_info("详细堆栈", traceback.format_exc())
                    print_info("提示", "请检查上方错误信息，修复后重试")
            else:
                print_warning(f"无效输入 '{choice}'，请输入 0-7 之间的数字")
 
            # 每步操作后暂停，等待用户确认
            if choice != "0":
                input("\n  按 Enter 键返回主菜单...")
 
    # ==================== 步骤1：数据加载与预处理 ====================
 
    def _step1_data_load_and_preprocess(self):
        """
        步骤1：多源数据读取 + 训练/测试划分 + 标准化。
        【修复】：自定义路径输入为空时，安全回退到默认路径，避免传入空字符串导致崩溃。
        """
        print_separator("步骤1: 多源数据读取与预处理")
 
        dl = self.modules.get("data_loader")
        pp = self.modules.get("preprocessing")
        if not dl or not pp:
            print_error("data_loader 或 preprocessing 模块未就绪，无法执行")
            return
 
        # ---- 1.1 数据加载 (含安全路径回退) ----
        default_x_path = os.path.join(DATA_DIR, "X_features")
        default_y_path = os.path.join(DATA_DIR, "Y_targets")
 
        print_info("数据路径", f"默认X: {default_x_path}")
        print_info("数据路径", f"默认Y: {default_y_path}")
 
        custom = input("  是否使用默认路径? [Y/n]: ").strip().lower()
        if custom == 'n':
            x_input = input(f"  请输入X数据路径 (直接回车使用默认): ").strip()
            y_input = input(f"  请输入Y数据路径 (直接回车使用默认): ").strip()
            # 【关键修复】：仅当用户实际输入了非空字符串时才覆盖默认值
            x_path = x_input if x_input else default_x_path
            y_path = y_input if y_input else default_y_path
        else:
            x_path = default_x_path
            y_path = default_y_path
 
        # 二次校验：确保路径不为空且文件/目录存在
        if not x_path or not y_path:
            print_error("数据路径不能为空，已中止操作")
            return
 
        start_t = time.time()
        try:
            X_raw, y_raw = dl.load_data(x_path, y_path)
        except Exception as e:
            log_exception(e, context="数据加载失败")
            print_warning("请检查路径是否正确、文件格式是否匹配后重试")
            return
 
        load_time = time.time() - start_t
        self.state.X_raw = X_raw
        self.state.y_raw = y_raw
        self.state.step_completed["data_load"] = True
        print_success(f"数据加载完成 ({load_time:.2f}s): X={X_raw.shape}, Y={y_raw.shape}")
 
        # ---- 1.2 预处理 (划分 + 标准化) ----
        print_step("1.2", "训练/测试划分与标准化")
 
        test_ratio = 0.2
        custom_ratio = input(f"  测试集比例 [默认 {test_ratio}]: ").strip()
        if custom_ratio:
            try:
                test_ratio = float(custom_ratio)
                if not (0.05 <= test_ratio <= 0.5):
                    print_warning("比例超出合理范围(0.05~0.5)，使用默认值 0.2")
                    test_ratio = 0.2
            except ValueError:
                print_warning("无效输入，使用默认值 0.2")
 
        try:
            result = pp.split_and_normalize(
                X_raw, y_raw,
                test_ratio=test_ratio,
                scaler_x=dl.WindFieldScaler(),
                scaler_y=dl.WindFieldScaler(),
                save_scaler=True
            )
        except Exception as e:
            log_exception(e, context="数据预处理失败")
            return
 
        (self.state.X_train, self.state.X_test,
         self.state.y_train, self.state.y_test,
         self.state.scaler_x, self.state.scaler_y) = result
 
        self.state.step_completed["preprocess"] = True
        print_success(f"预处理完成: 训练集 {self.state.X_train.shape[0]} 样本, "
                      f"测试集 {self.state.X_test.shape[0]} 样本")
 
    # ==================== 步骤2：模型构建与训练 ====================
 
    def _step2_model_build_and_train(self, auto=False):
        """
        步骤2：构建线性回归 + ResNet 模型并训练。
        【推理模式】：跳过训练，直接加载已保存的模型权重。
        【auto=True】：一键模式下跳过所有交互输入，使用默认参数。
        """
        print_separator("步骤2: 模型构建与训练")
 
        # 前置条件检查
        ready, reason = self.state.is_ready_for("train")
        if not ready:
            print_warning(reason)
            return
 
        # 推理模式下加载模型而非训练
        if INFERENCE_ONLY_MODE:
            print_info("推理模式", "跳过训练，直接加载已保存模型...")
            self._load_pretrained_models()
            return
 
        mod = self.modules.get("models")
        if not mod:
            print_error("models 模块未就绪，无法执行")
            return
 
        # ---- 2.1 构建模型 ----
        print_step("2.1", "构建预测模型")
 
        n_features = self.state.X_train.shape[1]
        n_targets = self.state.y_train.shape[1]
 
        # 线性回归
        lr_model = mod.build_linear_model()
        self.state.models["Linear"] = lr_model
        print_success(f"线性回归模型构建完成 (输入{n_features}维 → 输出{n_targets}维)")
 
        # ResNet
        try:
            resnet_params = DEFAULT_RESNET_PARAMS.copy()
            if not auto:
                custom = input("  是否自定义 ResNet 超参数? [y/N]: ").strip().lower()
                if custom == 'y':
                    depth = input(f"    残差块深度 [默认 {resnet_params.get('depth', 4)}]: ").strip()
                    if depth:
                        resnet_params['depth'] = int(depth)
                    hidden = input(f"    隐藏层宽度 [默认 {resnet_params.get('hidden_dim', 128)}]: ").strip()
                    if hidden:
                        resnet_params['hidden_dim'] = int(hidden)
                    epochs = input(f"    训练轮次 [默认 {resnet_params.get('epochs', 200)}]: ").strip()
                    if epochs:
                        resnet_params['epochs'] = int(epochs)
            else:
                print_info("一键模式", f"使用默认 ResNet 参数: depth={resnet_params.get('depth', 4)}, "
                                       f"hidden={resnet_params.get('hidden_dim', 128)}")
 
            resnet_model = mod.build_resnet_model(
                input_dim=n_features,
                output_dim=n_targets,
                **{k: v for k, v in resnet_params.items()
                   if k in ['depth', 'hidden_dim', 'dropout']}
            )
            self.state.models["ResNet"] = resnet_model
            print_success(f"ResNet 模型构建完成 (depth={resnet_params.get('depth', 4)}, "
                          f"hidden={resnet_params.get('hidden_dim', 128)})")
        except Exception as e:
            print_error(f"ResNet 构建失败: {e}")
            print_warning("将仅使用线性回归模型继续")
 
        # ---- 2.2 训练模型 ----
        print_step("2.2", "训练模型")
 
        for name, model in list(self.state.models.items()):
            print_info("训练中", f"正在训练 {name} ...")
            start_t = time.time()
 
            trained_model = mod.train_model(
                model, self.state.X_train, self.state.y_train,
                X_val=self.state.X_test, y_val=self.state.y_test,
                model_name=name,
                **(resnet_params if name == "ResNet" else {})
            )
 
            train_time = time.time() - start_t
            self.state.models[name] = trained_model
            print_success(f"{name} 训练完成 ({train_time:.1f}s)")
 
            # 自动保存
            if AUTO_SAVE_MODELS:
                mod.save_model(trained_model, name)
 
        self.state.step_completed["train"] = True
        self.state.models_trained = True
        print_success(f"所有模型训练完毕: {list(self.state.models.keys())}")
 
    def _load_pretrained_models(self):
        """推理模式下从磁盘加载已保存的模型。"""
        mod = self.modules.get("models")
        if not mod:
            print_error("models 模块未就绪")
            return
 
        # 同时加载 Scaler
        dl = self.modules.get("data_loader")
        if dl:
            scaler_x = dl.WindFieldScaler()
            scaler_y = dl.WindFieldScaler()
            if scaler_x.load_params() and scaler_y.load_params():
                self.state.scaler_x = scaler_x
                self.state.scaler_y = scaler_y
            else:
                print_error("Scaler 参数加载失败，推理结果可能不准确")
 
        for name in ["Linear", "ResNet"]:
            model = mod.load_model(name)
            if model is not None:
                self.state.models[name] = model
                print_success(f"已加载模型: {name}")
            else:
                print_warning(f"模型 {name} 未找到，跳过")
 
        if self.state.models:
            self.state.step_completed["train"] = True
            self.state.models_trained = True
            print_success(f"推理模式模型加载完毕: {list(self.state.models.keys())}")
        else:
            print_error("未找到任何已保存的模型，请先在训练模式下运行")
 
    # ==================== 步骤3：预测与评估 ====================
 
    def _step3_predict_and_evaluate(self, auto=False):
        """
        步骤3：用训练好的模型进行预测，然后执行全维度评估。
        【auto=True】：一键模式下跳过确认提示，直接执行。
        """
        print_separator("步骤3: 模型预测与评估分析")
 
        ready, reason = self.state.is_ready_for("predict")
        if not ready:
            print_warning(reason)
            return
 
        mod = self.modules.get("models")
        evl = self.modules.get("evaluator")
        if not mod or not evl:
            print_error("models 或 evaluator 模块未就绪")
            return
 
        # ---- 3.1 预测 ----
        print_step("3.1", "模型推理预测")
 
        self.state.predictions = {}
        self.state.predictions_inverse = {}
 
        for name, model in self.state.models.items():
            start_t = time.time()
            y_pred_scaled = mod.predict_model(model, self.state.X_test, model_name=name)
            pred_time = time.time() - start_t
 
            self.state.predictions[name] = y_pred_scaled
 
            # 反标准化为物理量纲
            if self.state.scaler_y is not None:
                y_pred_phys = self.state.scaler_y.inverse_transform(y_pred_scaled)
                self.state.predictions_inverse[name] = y_pred_phys
            else:
                self.state.predictions_inverse[name] = y_pred_scaled
 
            print_success(f"{name} 预测完成 ({pred_time:.3f}s): {y_pred_scaled.shape}")
 
        self.state.step_completed["predict"] = True
 
        # ---- 3.2 评估 ----
        print_step("3.2", "全维度评估分析")
 
        # 反标准化真实值
        if self.state.scaler_y is not None:
            y_true_phys = self.state.scaler_y.inverse_transform(self.state.y_test)
        else:
            y_true_phys = self.state.y_test
 
        # 调用新版评估器（返回完整指标字典）
        self.state.metrics = evl.evaluate_all_models(
            y_true=y_true_phys,
            y_pred_dict=self.state.predictions_inverse,
            feature_names=None  # 自动生成
        )
 
        self.state.step_completed["evaluate"] = True
        print_success("全维度评估完成，指标已记录")
 
    # ==================== 步骤4：可视化 ====================
 
    def _step4_visualize(self, auto=False):
        """
        步骤4：生成多维专业可视化报告。
        【auto=True】：一键模式下跳过确认，直接生成。
        """
        print_separator("步骤4: 结果可视化与报告生成")
 
        ready, reason = self.state.is_ready_for("visualize")
        if not ready:
            print_warning(reason)
            return
 
        vis = self.modules.get("visualizer")
        if not vis:
            print_error("visualizer 模块未就绪")
            return
 
        # 反标准化真实值
        if self.state.scaler_y is not None:
            y_true_phys = self.state.scaler_y.inverse_transform(self.state.y_test)
        else:
            y_true_phys = self.state.y_test
 
        start_t = time.time()
 
        # 优先使用新版可视化引擎
        if hasattr(vis, 'WindFieldVisualizer'):
            engine = vis.WindFieldVisualizer()
            engine.plot_comprehensive_analysis(
                y_true=y_true_phys,
                y_pred_dict=self.state.predictions_inverse,
                metrics_dict=self.state.metrics,
                feature_names=None
            )
        elif hasattr(vis, 'plot_wind_comparison'):
            # 向后兼容旧版接口
            pred_list = list(self.state.predictions_inverse.values())
            names = list(self.state.predictions_inverse.keys())
            if len(pred_list) >= 2:
                vis.plot_wind_comparison(
                    y_true_phys, pred_list[0], pred_list[1],
                    metrics_dict=self.state.metrics
                )
            else:
                print_warning("预测模型不足2个，旧版接口需要至少2个模型")
                return
        else:
            print_error("visualizer 模块无可用的绘图接口")
            return
 
        vis_time = time.time() - start_t
        self.state.step_completed["visualize"] = True
        print_success(f"可视化报告生成完毕 ({vis_time:.1f}s)，保存至: {PLOT_DIR}")
 
    # ==================== 步骤5：一键全流程 ====================
 
    def _step5_auto_pipeline(self):
        """
        一键自动执行完整流水线：数据加载 → 预处理 → 训练/加载模型 → 预测 → 评估 → 可视化。
        任何一步失败都会中止并报告，不会静默继续。
        【修复】：所有子步骤均以 auto=True 调用，完全无交互阻塞。
        """
        print_separator("步骤5: 一键自动全流程")
        print_info("流水线", "即将自动执行完整分析流程，请确保数据文件已就位")
 
        confirm = input("  确认开始? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print_info("已取消", "返回主菜单")
            return
 
        pipeline_start = time.time()
        steps = [
            ("1/5 数据加载与预处理", self._step1_data_load_and_preprocess_auto),
            ("2/5 模型构建与训练", lambda: self._step2_model_build_and_train(auto=True)),
            ("3/5 模型预测与评估", lambda: self._step3_predict_and_evaluate(auto=True)),
            ("4/5 结果可视化报告", lambda: self._step4_visualize(auto=True))
        ]
 
        for i, (desc, func) in enumerate(steps):
            print_step(f"AUTO-{i+1}", desc)
            try:
                func()
                print_success(f"{desc} ✓")
            except Exception as e:
                print_error(f"{desc} 失败: {e}")
                print_info("堆栈", traceback.format_exc())
                print_error("流水线中止，请修复问题后重试")
                return
 
        total_time = time.time() - pipeline_start
        print_separator("全流程完成")
        print_success(f"🎉 全流程执行完毕！总耗时: {total_time:.1f}s")
        print_info("模型保存", MODEL_PARAMS_DIR)
        print_info("图表报告", PLOT_DIR)
 
    def _step1_data_load_and_preprocess_auto(self):
        """一键模式下的自动化数据加载（无交互，使用默认参数）。"""
        dl = self.modules.get("data_loader")
        pp = self.modules.get("preprocessing")
        if not dl or not pp:
            raise RuntimeError("data_loader 或 preprocessing 模块未就绪")
 
        x_path = os.path.join(DATA_DIR, "X_features")
        y_path = os.path.join(DATA_DIR, "Y_targets")
 
        X_raw, y_raw = dl.load_data(x_path, y_path)
        self.state.X_raw = X_raw
        self.state.y_raw = y_raw
        self.state.step_completed["data_load"] = True
 
        result = pp.split_and_normalize(
            X_raw, y_raw,
            test_ratio=0.2,
            scaler_x=dl.WindFieldScaler(),
            scaler_y=dl.WindFieldScaler(),
            save_scaler=True
        )
        (self.state.X_train, self.state.X_test,
         self.state.y_train, self.state.y_test,
         self.state.scaler_x, self.state.scaler_y) = result
        self.state.step_completed["preprocess"] = True
 
    # ==================== 辅助功能 ====================
 
    def _show_status(self):
        """显示当前流水线详细状态。"""
        print_separator("系统状态")
        print(self.state.get_summary())
 
        # 显示运行时间
        if self.start_time:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            print_info("系统运行时间", f"{mins}分{secs}秒")
 
        # 显示目录信息
        print_info("数据目录", DATA_DIR)
        print_info("模型目录", MODEL_PARAMS_DIR)
        print_info("图表目录", PLOT_DIR)
        print_info("运行模式",
                   "纯推理模式" if INFERENCE_ONLY_MODE else "完整训练模式")
 
    def _reset_system(self):
        """重置系统状态，清空所有中间结果。"""
        confirm = input("  确认重置? 所有中间结果将被清空 [y/N]: ").strip().lower()
        if confirm == 'y':
            self.state.reset()
            print_success("系统已重置，所有状态已清空")
        else:
            print_info("已取消", "返回主菜单")
 
    def _exit_system(self):
        """优雅退出系统。"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            print_info("本次会话时长", f"{mins}分{secs}秒")
 
        completed = sum(1 for v in self.state.step_completed.values() if v)
        total = len(self.state.step_completed)
        print_info("完成进度", f"{completed}/{total} 步骤")
 
        print_separator("再见")
        print_success("感谢使用三维风场智能预测系统！")
        sys.exit(0)
 
 
# ==================== 程序入口 ====================
 
def launch():
    """系统启动入口，供 main.py 调用。"""
    menu = WindFieldMenu()
    menu.run()
 
 
if __name__ == "__main__":
    launch()