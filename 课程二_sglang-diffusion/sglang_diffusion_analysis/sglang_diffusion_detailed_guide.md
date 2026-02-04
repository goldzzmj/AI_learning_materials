# SGLang-Diffusion 框架详细解析文档

> **作者**: AI Assistant  
> **日期**: 2026-02-03  
> **版本**: v1.0  
> **目标读者**: 深度学习工程师、推理优化工程师、研究人员

---

## 目录

1. [框架概述](#1-框架概述)
2. [核心架构设计](#2-核心架构设计)
3. [代码逐行解析](#3-代码逐行解析)
4. [Qwen-Image 加速实现](#4-qwen-image-加速实现)
5. [底层加速原理](#5-底层加速原理)
6. [快速执行脚本](#6-快速执行脚本)
7. [性能优化建议](#7-性能优化建议)

---

## 1. 框架概述

### 1.1 什么是 SGLang-Diffusion？

**SGLang-Diffusion** 是 LMSYS 团队基于 SGLang 推理引擎开发的扩散模型高性能推理框架。它将 SGLang 在大语言模型(LLM)推理中积累的优化经验，扩展到图像和视频生成领域。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **多模型支持** | Qwen-Image、Flux、Wan、HunyuanVideo 等 |
| **高性能推理** | 相比 Diffusers 最高 5.9x 加速 |
| **统一引擎** | 同时支持 LLM 和 Diffusion 模型 |
| **模块化设计** | ComposedPipelineBase + PipelineStage 架构 |
| **多种并行** | USP、TP、CFG Parallel、SP |

---

## 2. 核心架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   CLI       │  │  Python SDK │  │  OpenAI-Compatible API  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Scheduler & Entry Points                      │
│              调度器与入口点管理（复用SGLang调度器）                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ComposedPipelineBase                          │
│              组合式Pipeline基类（核心抽象层）                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │  Executor   │ │Stage Manager│ │Module Loader│ │Config Parser│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PipelineStage Pipeline                        │
│                    Pipeline阶段流水线                            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │Input│→│Text │→│Image│→│Cond │→│Time │→│Latent│→│Denoise│    │
│  │Val  │ │Enc  │ │Enc  │ │Prep │ │Prep │ │Prep  │→│Decoding│   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Components                              │
│                    模型组件层                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Text Enc │ │   VAE   │ │Transformer│ │Scheduler│ │Tokenizer│   │
│  │(CLIP/T5)│ │         │ │   /DiT    │ │         │ │         │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Parallelism & Optimization Layer                    │
│                    并行与优化层                                  │
│  ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐    │
│  │  USP   │ │   TP   │ │CFG Parallel│ │   SP   │ │sgl-kernel│    │
│  └────────┘ └────────┘ └──────────┘ └────────┘ └──────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 代码逐行解析

### 3.1 ComposedPipelineBase - Pipeline组合基类

这是整个框架的核心抽象类，负责管理Pipeline的生命周期和模块加载。

```python
# 文件: python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py

"""
Base class for composed pipelines.
组合式Pipeline的基类。

This module defines the base class for pipelines that are composed of multiple stages.
本模块定义了由多个阶段组成的Pipeline的基类。
"""

# 导入必要的库
import os                           # 操作系统接口，用于文件路径操作
from abc import ABC, abstractmethod # 抽象基类，用于定义抽象接口
from typing import Any, cast        # 类型注解支持

import torch                        # PyTorch深度学习框架
from tqdm import tqdm               # 进度条显示库

# 导入SGLang-Diffusion内部模块
from sglang.multimodal_gen.runtime.loader.component_loader import (
    PipelineComponentLoader,        # Pipeline组件加载器，负责加载各个模型组件
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,               # Pipeline执行器，负责执行各个Stage
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch, Req                 # 批次调度相关类
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,           # 可能下载模型（如果本地不存在）
    verify_model_config_and_directory,  # 验证模型配置和目录结构
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# 初始化日志记录器
logger = init_logger(__name__)


class ComposedPipelineBase(ABC):
    """
    Base class for pipelines composed of multiple stages.
    由多个阶段组成的Pipeline的基类。
    
    This class provides the framework for creating pipelines by composing multiple
    stages together. Each stage is responsible for a specific part of the diffusion
    process, and the pipeline orchestrates the execution of these stages.
    
    这个类提供了一个框架，用于通过组合多个阶段来创建Pipeline。
    每个阶段负责扩散过程的特定部分，Pipeline负责编排这些阶段的执行。
    """
    
    # 类属性定义
    is_video_pipeline: bool = False  # 是否为视频Pipeline（子类可覆盖）
    
    # 需要从配置中加载的模块列表
    _required_config_modules: list[str] = []
    
    # 额外的配置模块映射（用于处理特殊情况）
    _extra_config_module_map: dict[str, str] = {}
    
    # 服务器参数
    server_args: ServerArgs | None = None
    
    # 加载的模块字典
    modules: dict[str, Any] = {}
    
    # Pipeline执行器
    executor: PipelineExecutor | None = None
    
    # 在diffusers中关联的Pipeline名称
    pipeline_name: str

    def is_lora_effective(self):
        """检查LoRA是否生效（子类可覆盖）"""
        return False

    def is_lora_set(self):
        """检查是否设置了LoRA（子类可覆盖）"""
        return False

    def __init__(
        self,
        model_path: str,                                    # 模型路径
        server_args: ServerArgs,                            # 服务器参数
        required_config_modules: list[str] | None = None,   # 必需的配置模块
        loaded_modules: dict[str, torch.nn.Module] | None = None,  # 预加载的模块
        executor: PipelineExecutor | None = None,           # 自定义执行器
    ):
        """
        Initialize the pipeline.
        初始化Pipeline。
        
        After __init__, the pipeline should be ready to use.
        The pipeline should be stateless and not hold any batch state.
        
        __init__之后，Pipeline应该可以立即使用。
        Pipeline应该是无状态的，不持有任何批次状态。
        """
        # 保存服务器参数
        self.server_args = server_args
        
        # 保存模型路径
        self.model_path: str = model_path
        
        # 初始化Stage列表和名称映射
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        
        # 构建或保存执行器
        self.executor = executor or self.build_executor(server_args=server_args)
        
        # 如果提供了必需的配置模块，则使用它
        if required_config_modules is not None:
            self._required_config_modules = required_config_modules
        
        # 检查是否设置了必需的配置模块
        if self._required_config_modules is None:
            raise NotImplementedError("Subclass must set _required_config_modules")
        
        # 初始化内存使用记录字典 [模块名称, GPU内存使用量]
        self.memory_usages: dict[str, float] = {}
        
        # 加载Pipeline模块
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(server_args, loaded_modules)
        
        # 后初始化（创建Pipeline Stages）
        self.__post_init__()

    def build_executor(self, server_args: ServerArgs):
        """
        构建Pipeline执行器。
        默认使用ParallelExecutor支持并行执行。
        """
        from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
            ParallelExecutor,  # 并行执行器，支持多GPU执行
        )
        return ParallelExecutor(server_args=server_args)

    def __post_init__(self) -> None:
        """
        后初始化方法。
        在模块加载完成后调用，用于初始化Pipeline和创建Stages。
        """
        # 确保服务器参数已设置
        assert self.server_args is not None, "server_args must be set"
        
        # 初始化Pipeline（子类可覆盖）
        self.initialize_pipeline(self.server_args)
        
        # 创建Pipeline Stages
        logger.info("Creating pipeline stages...")
        self.create_pipeline_stages(self.server_args)

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        """
        获取指定名称的模块。
        
        Args:
            module_name: 模块名称
            default_value: 如果模块不存在，返回的默认值
        
        Returns:
            模块对象或默认值
        """
        if module_name not in self.modules:
            return default_value
        return self.modules[module_name]

    def add_module(self, module_name: str, module: Any):
        """添加模块到模块字典"""
        self.modules[module_name] = module

    def _load_config(self) -> dict[str, Any]:
        """
        加载模型配置。
        
        Returns:
            模型配置字典
        """
        # 可能需要下载模型（如果是HuggingFace Hub路径）
        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        
        logger.info("Model path: %s", model_path)
        
        # 验证模型配置和目录结构
        config = verify_model_config_and_directory(model_path)
        return cast(dict[str, Any], config)

    @property
    def required_config_modules(self) -> list[str]:
        """
        获取必需的配置模块列表。
        
        这些模块名称应与diffusers目录和model_index.json文件中的名称匹配。
        这些模块将使用PipelineComponentLoader加载，并在modules字典中可用。
        
        Example:
            class ConcretePipeline(ComposedPipelineBase):
                _required_config_modules = ["vae", "text_encoder", "transformer", "scheduler", "tokenizer"]
        """
        return self._required_config_modules

    @property
    def stages(self) -> list[PipelineStage]:
        """获取Pipeline中的所有Stage"""
        return self._stages

    @abstractmethod
    def create_pipeline_stages(self, server_args: ServerArgs):
        """
        创建推理Pipeline的Stages（抽象方法，子类必须实现）。
        
        Args:
            server_args: 服务器参数
        """
        raise NotImplementedError

    def initialize_pipeline(self, server_args: ServerArgs):
        """
        初始化Pipeline（子类可覆盖）。
        
        Args:
            server_args: 服务器参数
        """
        return

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        从配置加载模块。
        
        Args:
            server_args: 服务器参数
            loaded_modules: 可选，预加载的模块字典
        
        Returns:
            加载的模块字典
        """
        # 加载模型配置
        model_index = self._load_config()
        logger.info("Loading pipeline modules from config: %s", model_index)
        
        # 移除非Pipeline模块的键
        model_index.pop("_class_name")          # 移除类名
        model_index.pop("_diffusers_version")   # 移除diffusers版本
        
        # 处理MoE Pipeline的特殊情况（Wan2.2双Transformer）
        if (
            "boundary_ratio" in model_index
            and model_index["boundary_ratio"] is not None
        ):
            has_transformer = (
                "transformer" in model_index
                or "transformer_2" in model_index
                or "transformer" in self.required_config_modules
                or "transformer_2" in self.required_config_modules
            )
            if has_transformer:
                logger.info(
                    "MoE pipeline detected. Adding transformer_2 to self.required_config_modules..."
                )
                if "transformer_2" not in self.required_config_modules:
                    self._required_config_modules.append("transformer_2")
            else:
                logger.info(
                    "Boundary ratio found in model_index.json without transformers; "
                    "using it for pipeline config only."
                )
            # 设置边界比例（用于双Transformer切换）
            server_args.pipeline_config.dit_config.boundary_ratio = model_index[
                "boundary_ratio"
            ]
        
        # 移除已处理的键
        model_index.pop("boundary_ratio", None)
        model_index.pop("expand_timesteps", None)  # Wan2.2 ti2v使用
        
        # 确保至少有一个Pipeline模块
        assert (
            len(model_index) > 1
        ), "model_index.json must contain at least one pipeline module"
        
        # 只保留必需的模块
        model_index = {
            required_module: model_index[required_module]
            for required_module in self.required_config_modules
        }
        
        # 处理额外的配置模块映射
        for module_name in self.required_config_modules:
            if (
                module_name not in model_index
                and module_name in self._extra_config_module_map
            ):
                extra_module_value = self._extra_config_module_map[module_name]
                logger.warning(
                    "model_index.json does not contain a %s module, but found {%s: %s} in _extra_config_module_map, adding to model_index.",
                    module_name,
                    module_name,
                    extra_module_value,
                )
                if extra_module_value in model_index:
                    logger.info(
                        "Using module %s for %s", extra_module_value, module_name
                    )
                    model_index[module_name] = model_index[extra_module_value]
                    continue
                else:
                    raise ValueError(
                        f"Required module key: {module_name} value: {model_index.get(module_name)} was not found in loaded modules {model_index.keys()}"
                    )
        
        # 获取所有必需的组件
        required_modules = self.required_config_modules
        logger.info("Loading required components: %s", required_modules)
        
        # 加载各个组件
        loaded_components = {}
        for module_name, (
            transformers_or_diffusers,  # 使用transformers还是diffusers库
            architecture,                # 架构名称
        ) in tqdm(iterable=model_index.items(), desc="Loading required modules"):
            
            # 跳过值为null的模块
            if transformers_or_diffusers is None:
                logger.warning(
                    "Module %s in model_index.json has null value, removing from required_config_modules",
                    module_name,
                )
                if module_name in self.required_config_modules:
                    self._required_config_modules.remove(module_name)
                continue
            
            # 跳过非必需的模块
            if module_name not in required_modules:
                logger.info("Skipping module %s", module_name)
                continue
            
            # 如果提供了预加载的模块，直接使用
            if loaded_modules is not None and module_name in loaded_modules:
                logger.info("Using module %s already provided", module_name)
                loaded_components[module_name] = loaded_modules[module_name]
                continue
            
            # 使用额外的配置模块映射名称（如果存在）
            if module_name in self._extra_config_module_map:
                load_module_name = self._extra_config_module_map[module_name]
            else:
                load_module_name = module_name
            
            # 处理自定义VAE路径
            if module_name == "vae" and server_args.vae_path is not None:
                component_model_path = server_args.vae_path
                # 如果路径不存在本地，从HuggingFace Hub下载
                if not os.path.exists(component_model_path):
                    component_model_path = maybe_download_model(component_model_path)
                logger.info(
                    "Using custom VAE path: %s instead of default path: %s",
                    component_model_path,
                    os.path.join(self.model_path, load_module_name),
                )
            else:
                component_model_path = os.path.join(self.model_path, load_module_name)
            
            # 使用PipelineComponentLoader加载组件
            module, memory_usage = PipelineComponentLoader.load_component(
                component_name=load_module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                server_args=server_args,
            )
            
            # 记录内存使用量
            self.memory_usages[load_module_name] = memory_usage
            
            # 保存加载的模块
            if module_name in loaded_components:
                logger.warning("Overwriting module %s", module_name)
            loaded_components[module_name] = module
        
        # 检查所有必需模块是否已加载
        for module_name in required_modules:
            if (
                module_name not in loaded_components
                or loaded_components[module_name] is None
            ):
                raise ValueError(
                    f"Required module: {module_name} was not found in loaded modules: {list(loaded_components.keys())}"
                )
        
        logger.debug("Memory usage of loaded modules: %s", self.memory_usages)
        
        return loaded_components

    def add_stage(self, stage_name: str, stage: PipelineStage):
        """
        添加Stage到Pipeline。
        
        Args:
            stage_name: Stage名称
            stage: Stage实例
        """
        assert self.modules is not None, "No modules are registered"
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)  # 将Stage设置为实例属性

    @torch.no_grad()  # 禁用梯度计算，节省内存
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        使用Pipeline生成视频或图像。
        
        Args:
            batch: 要处理的批次
            server_args: 推理参数
        
        Returns:
            包含生成视频或图像的批次
        """
        # LoRA警告
        if self.is_lora_set() and not self.is_lora_effective():
            logger.warning(
                "LoRA adapter is set, but not effective. Please make sure the LoRA weights are merged"
            )
        
        # 执行各个Stage
        if not batch.is_warmup and not batch.suppress_logs:
            logger.info(
                "Running pipeline stages: %s",
                list(self._stage_name_mapping.keys()),
                main_process_only=True,
            )
        
        # 使用执行器执行所有Stage
        return self.executor.execute_with_profiling(self.stages, batch, server_args)
```

### 3.2 PipelineStage - Pipeline阶段基类

```python
# 文件: python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py

"""
Base classes for pipeline stages.
Pipeline阶段的基类。

This module defines the abstract base classes for pipeline stages that can be
composed to create complete diffusion pipelines.
本模块定义了可以组合以创建完整扩散Pipeline的Pipeline阶段的抽象基类。
"""

from abc import ABC, abstractmethod  # 抽象基类和抽象方法
from enum import Enum, auto          # 枚举类型

import torch                         # PyTorch

# 导入SGLang-Diffusion内部模块
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,  # 验证结果类
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class StageParallelismType(Enum):
    """
    Stage并行类型枚举。
    定义Stage可以在哪种并行模式下执行。
    """
    REPLICATED = auto()       # 在所有GPU上执行
    MAIN_RANK_ONLY = auto()   # 仅在主rank上执行
    CFG_PARALLEL = auto()     # 需要CFG并行


class StageVerificationError(Exception):
    """Stage验证失败时抛出的异常"""
    pass


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    所有Pipeline阶段的抽象基类。
    
    A pipeline stage represents a discrete step in the diffusion process that can be
    composed with other stages to create a complete pipeline. Each stage is responsible
    for a specific part of the process, such as prompt encoding, latent preparation, etc.
    
    Pipeline阶段代表扩散过程中的一个离散步骤，可以与其他阶段组合以创建完整的Pipeline。
    每个阶段负责过程的特定部分，如提示词编码、latent准备等。
    """

    def __init__(self):
        """初始化Stage，获取全局服务器参数"""
        self.server_args = get_global_server_args()

    def log_info(self, msg, *args):
        """记录信息日志，带有Stage名称前缀"""
        if self.server_args.comfyui_mode:
            return  # ComfyUI模式下不记录
        logger.info(f"[{self.__class__.__name__}] {msg}", *args)

    def log_warning(self, msg, *args):
        """记录警告日志"""
        logger.warning(f"[{self.__class__.__name__}] {msg}", *args)

    def log_error(self, msg, *args):
        """记录错误日志"""
        logger.error(f"[{self.__class__.__name__}] {msg}", *args)

    def log_debug(self, msg, *args):
        """记录调试日志"""
        logger.debug(f"[{self.__class__.__name__}] {msg}", *args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """
        验证Stage的输入。
        
        Args:
            batch: 输入批次
            server_args: 服务器参数
        
        Returns:
            验证结果
        
        Example:
            def verify_input(self, batch, server_args):
                result = VerificationResult()
                result.add_check("height", batch.height, V.positive_int_divisible(8))
                result.add_check("width", batch.width, V.positive_int_divisible(8))
                result.add_check("image_latent", batch.image_latent, V.is_tensor)
                return result
        """
        # 默认实现 - 不执行验证
        return VerificationResult()

    def maybe_free_model_hooks(self):
        """可能释放模型hooks（子类可覆盖）"""
        pass

    def load_model(self):
        """加载Stage的模型（子类可覆盖）"""
        pass

    def offload_model(self):
        """卸载Stage的模型（子类可覆盖）"""
        pass

    @property
    def parallelism_type(self) -> StageParallelismType:
        """
        获取Stage的并行类型。
        默认在所有rank上复制执行。
        """
        return StageParallelismType.REPLICATED

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """
        验证Stage的输出。
        
        Returns:
            验证结果
        """
        # 默认实现 - 不执行验证
        return VerificationResult()

    def _run_verification(
        self,
        verification_result: VerificationResult,
        stage_name: str,
        verification_type: str,
    ) -> None:
        """
        运行验证，如果有检查失败则抛出错误。
        
        Args:
            verification_result: verify_input或verify_output的结果
            stage_name: 当前Stage的名称
            verification_type: "input" 或 "output"
        """
        if not verification_result.is_valid():
            failed_fields = verification_result.get_failed_fields()
            if failed_fields:
                # 获取详细的失败信息
                detailed_summary = verification_result.get_failure_summary()
                
                failed_fields_str = ", ".join(failed_fields)
                error_msg = (
                    f"{verification_type.capitalize()} verification failed for {stage_name}: "
                    f"Failed fields: {failed_fields_str}\n"
                    f"Details: {detailed_summary}"
                )
                raise StageVerificationError(error_msg)

    @property
    def device(self) -> torch.device:
        """获取Stage的设备"""
        return torch.device(
            current_platform.device_type,
        )

    def set_logging(self, enable: bool):
        """
        启用或禁用Stage的日志记录。
        
        Args:
            enable: 是否启用日志
        """
        self._enable_logging = enable

    def __call__(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        在批次上执行Stage的处理，可选验证和日志记录。
        子类不应覆盖此方法。
        
        Returns:
            经过此Stage处理后的更新批次
        """
        stage_name = self.__class__.__name__
        
        # 执行前输入验证
        try:
            input_result = self.verify_input(batch, server_args)
            self._run_verification(input_result, stage_name, "input")
        except Exception as e:
            logger.error("Input verification failed for %s: %s", stage_name, str(e))
            raise
        
        # 使用统一的性能分析器执行实际的Stage逻辑
        with StageProfiler(
            stage_name,
            logger=logger,
            timings=batch.timings,
            perf_dump_path_provided=batch.perf_dump_path is not None,
            log_stage_start_end=not batch.is_warmup
            and not (self.server_args and self.server_args.comfyui_mode),
        ):
            result = self.forward(batch, server_args)
        
        # 执行后输出验证
        try:
            output_result = self.verify_output(result, server_args)
            self._run_verification(output_result, stage_name, "output")
        except Exception as e:
            logger.error("Output verification failed for %s: %s", stage_name, str(e))
            raise
        
        return result

    @abstractmethod
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Stage处理的前向传播。
        子类必须实现此方法以提供Stage的前向处理逻辑。
        
        Returns:
            经过此Stage处理后的更新批次
        """
        raise NotImplementedError

    def backward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """反向传播（未实现）"""
        raise NotImplementedError
```

### 3.3 DenoisingStage - 去噪阶段（核心加速）

```python
# 文件: python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py

"""
Denoising stage for diffusion pipelines.
扩散Pipeline的去噪阶段。

This stage handles the iterative denoising process that transforms
the initial noise into the final output.
本阶段处理迭代去噪过程，将初始噪声转换为最终输出。
"""

import inspect          # 用于检查函数签名
import math             # 数学函数
import os               # 操作系统接口
import time             # 时间相关函数
import weakref          # 弱引用，避免循环引用
from collections.abc import Iterable
from functools import lru_cache  # LRU缓存装饰器
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange     # 张量重排库
from tqdm.auto import tqdm       # 进度条

# 导入SGLang-Diffusion内部模块
from sglang.multimodal_gen import envs  # 环境变量配置
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType, STA_Mode  # 模型任务类型和STA模式
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_TI2V_5B_Config, WanI2V480PConfig  # Wan模型配置
)
from sglang.multimodal_gen.runtime.distributed import (
    cfg_model_parallel_all_reduce,    # CFG并行all-reduce
    get_local_torch_device,           # 获取本地torch设备
    get_sp_parallel_rank,             # 获取SP并行rank
    get_sp_world_size,                # 获取SP并行world size
    get_world_group,                  # 获取world group
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,  # 序列并行all-gather
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,                    # 获取CFG group
    get_classifier_free_guidance_rank, # 获取CFG rank
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.attention.STA_configuration import (
    configure_sta,                    # 配置STA
    save_mask_search_results,         # 保存mask搜索结果
)
from sglang.multimodal_gen.runtime.loader.transformer_loader import TransformerLoader
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage, StageParallelismType
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V, VerificationResult
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum, current_platform
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.multimodal_gen.utils import dict_to_3d_list, masks_like

logger = init_logger(__name__)


class DenoisingStage(PipelineStage):
    """
    Stage for running the denoising loop in diffusion pipelines.
    在扩散Pipeline中运行去噪循环的阶段。
    
    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    本阶段处理迭代去噪过程，将初始噪声转换为最终输出。
    """

    def __init__(
        self, 
        transformer,           # DiT/Transformer模型
        scheduler,             # 噪声调度器
        pipeline=None,         # 所属Pipeline（弱引用）
        transformer_2=None,    # 第二个Transformer（MoE模型使用）
        vae=None               # VAE模型
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.transformer_2 = transformer_2
        
        # 从配置获取模型参数
        hidden_size = self.server_args.pipeline_config.dit_config.hidden_size
        num_attention_heads = (
            self.server_args.pipeline_config.dit_config.num_attention_heads
        )
        attn_head_size = hidden_size // num_attention_heads  # 每个注意力头的维度
        
        # 对每个Transformer启用torch.compile（如果配置允许）
        for transformer in filter(None, [self.transformer, self.transformer_2]):
            self._maybe_enable_torch_compile(transformer)
        
        self.scheduler = scheduler
        self.vae = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None
        
        # 获取注意力后端
        self.attn_backend = get_attn_backend(
            head_size=attn_head_size,
            dtype=torch.float16,
        )
        
        # CFG相关
        self.guidance = None
        
        # 其他
        self.profiler = None
        
        # Cache-DiT状态（用于延迟挂载和幂等控制）
        self._cache_dit_enabled = False
        self._cached_num_steps = None
        self._is_warmed_up = False

    def _maybe_enable_torch_compile(self, module: object) -> None:
        """
        使用torch.compile编译模块，如果可用则启用inductor重叠优化。
        如果torch compile被禁用或对象不是nn.Module，则不执行任何操作。
        
        Args:
            module: 要编译的模块
        """
        if not self.server_args.enable_torch_compile or not isinstance(
            module, nn.Module
        ):
            return
        try:
            import torch._inductor.config as _inductor_cfg
            # 启用计算通信重叠优化
            _inductor_cfg.reorder_for_compute_comm_overlap = True
        except ImportError:
            pass
        
        # 从环境变量获取编译模式
        mode = os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
        logger.info(f"Compiling transformer with mode: {mode}")
        
        # 编译模块
        module.compile(mode=mode, fullgraph=False, dynamic=None)

    def _maybe_enable_cache_dit(self, num_inference_steps: int, batch: Req) -> None:
        """
        如果配置了，在transformers上启用cache-dit（幂等）。
        
        此方法应在transformer完全加载后、torch.compile应用前调用。
        
        对于双transformer模型（如Wan2.2），在两个transformer上都启用cache-dit
        （可能使用不同的配置）。
        
        Args:
            num_inference_steps: 推理步数
            batch: 当前批次
        """
        # 如果已经启用，检查步数是否变化
        if self._cache_dit_enabled:
            if self._cached_num_steps != num_inference_steps:
                logger.warning(
                    "num_inference_steps changed from %d to %d after cache-dit was enabled. "
                    "Continuing with initial configuration (steps=%d).",
                    self._cached_num_steps,
                    num_inference_steps,
                    self._cached_num_steps,
                )
            return
        
        # 检查是否在配置中启用了cache-dit
        if not envs.SGLANG_CACHE_DIT_ENABLED or batch.is_warmup:
            return
        
        # 导入cache-dit集成模块
        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            CacheDitConfig, enable_cache_on_dual_transformer, 
            enable_cache_on_transformer, get_scm_mask
        )
        from sglang.multimodal_gen.runtime.distributed import (
            get_sp_group, get_tp_group, get_world_size
        )
        
        world_size = get_world_size()
        parallelized = world_size > 1
        
        # 获取SP和TP group
        sp_group = None
        tp_group = None
        if parallelized:
            sp_group_candidate = get_sp_group()
            tp_group_candidate = get_tp_group()
            
            sp_world_size = sp_group_candidate.world_size if sp_group_candidate else 1
            tp_world_size = tp_group_candidate.world_size if tp_group_candidate else 1
            
            has_sp = sp_world_size > 1
            has_tp = tp_world_size > 1
            
            sp_group = sp_group_candidate.device_group if has_sp else None
            tp_group = tp_group_candidate.device_group if has_tp else None
            
            logger.info(
                "cache-dit enabled in distributed environment (world_size=%d, has_sp=%s, has_tp=%s)",
                world_size, has_sp, has_tp,
            )
        
        # === 从环境变量解析SCM配置 ===
        scm_preset = envs.SGLANG_CACHE_DIT_SCM_PRESET
        scm_compute_bins_str = envs.SGLANG_CACHE_DIT_SCM_COMPUTE_BINS
        scm_cache_bins_str = envs.SGLANG_CACHE_DIT_SCM_CACHE_BINS
        scm_policy = envs.SGLANG_CACHE_DIT_SCM_POLICY
        
        # 如果提供了自定义bins，解析它们
        scm_compute_bins = None
        scm_cache_bins = None
        if scm_compute_bins_str and scm_cache_bins_str:
            try:
                scm_compute_bins = [int(x.strip()) for x in scm_compute_bins_str.split(",")]
                scm_cache_bins = [int(x.strip()) for x in scm_cache_bins_str.split(",")]
            except ValueError as e:
                logger.warning("Failed to parse SCM bins: %s. SCM disabled.", e)
                scm_preset = "none"
        elif scm_compute_bins_str or scm_cache_bins_str:
            logger.warning(
                "SCM custom bins require both compute_bins and cache_bins. "
                "Only one was provided (compute=%s, cache=%s). Falling back to preset '%s'.",
                scm_compute_bins_str, scm_cache_bins_str, scm_preset,
            )
        
        # 使用cache-dit的steps_mask()生成SCM mask
        steps_computation_mask = get_scm_mask(
            preset=scm_preset,
            num_inference_steps=num_inference_steps,
            compute_bins=scm_compute_bins,
            cache_bins=scm_cache_bins,
        )
        
        # 为主transformer（高噪声专家）构建配置
        primary_config = CacheDitConfig(
            enabled=True,
            Fn_compute_blocks=envs.SGLANG_CACHE_DIT_FN,           # 前Fn个block计算
            Bn_compute_blocks=envs.SGLANG_CACHE_DIT_BN,           # 后Bn个block计算
            max_warmup_steps=envs.SGLANG_CACHE_DIT_WARMUP,        # 最大warmup步数
            residual_diff_threshold=envs.SGLANG_CACHE_DIT_RDT,    # 残差差异阈值
            max_continuous_cached_steps=envs.SGLANG_CACHE_DIT_MC, # 最大连续缓存步数
            enable_taylorseer=envs.SGLANG_CACHE_DIT_TAYLORSEER,   # 启用TaylorSeer
            taylorseer_order=envs.SGLANG_CACHE_DIT_TS_ORDER,      # TaylorSeer阶数
            num_inference_steps=num_inference_steps,
            steps_computation_mask=steps_computation_mask,        # SCM mask
            steps_computation_policy=scm_policy,                  # SCM策略
        )
        
        if self.transformer_2 is not None:
            # 双transformer情况
            # 为次要transformer（低噪声专家）构建配置
            secondary_config = CacheDitConfig(
                enabled=True,
                Fn_compute_blocks=envs.SGLANG_CACHE_DIT_SECONDARY_FN,
                Bn_compute_blocks=envs.SGLANG_CACHE_DIT_SECONDARY_BN,
                max_warmup_steps=envs.SGLANG_CACHE_DIT_SECONDARY_WARMUP,
                residual_diff_threshold=envs.SGLANG_CACHE_DIT_SECONDARY_RDT,
                max_continuous_cached_steps=envs.SGLANG_CACHE_DIT_SECONDARY_MC,
                enable_taylorseer=envs.SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER,
                taylorseer_order=envs.SGLANG_CACHE_DIT_SECONDARY_TS_ORDER,
                num_inference_steps=num_inference_steps,
                steps_computation_mask=steps_computation_mask,  # 与主transformer共享
                steps_computation_policy=scm_policy,
            )
            
            # 对于双transformer，必须使用BlockAdapter同时在两个transformer上启用cache
            self.transformer, self.transformer_2 = enable_cache_on_dual_transformer(
                self.transformer, self.transformer_2,
                primary_config, secondary_config,
                model_name="wan2.2",
                sp_group=sp_group, tp_group=tp_group,
            )
            logger.info("cache-dit enabled on dual transformers (steps=%d)", num_inference_steps)
        else:
            # 单transformer情况
            self.transformer = enable_cache_on_transformer(
                self.transformer, primary_config,
                model_name="transformer",
                sp_group=sp_group, tp_group=tp_group,
            )
            logger.info(
                "cache-dit enabled on transformer (steps=%d, Fn=%d, Bn=%d, rdt=%.3f)",
                num_inference_steps,
                envs.SGLANG_CACHE_DIT_FN,
                envs.SGLANG_CACHE_DIT_BN,
                envs.SGLANG_CACHE_DIT_RDT,
            )
        
        self._cache_dit_enabled = True
        self._cached_num_steps = num_inference_steps

    @lru_cache(maxsize=8)  # LRU缓存，最多缓存8个结果
    def _build_guidance(self, batch_size, target_dtype, device, guidance_val):
        """
        构建引导张量。此方法被缓存。
        
        Args:
            batch_size: 批次大小
            target_dtype: 目标数据类型
            device: 目标设备
            guidance_val: 引导值
        
        Returns:
            引导张量
        """
        return (
            torch.full(
                (batch_size,),
                guidance_val,
                dtype=torch.float32,
                device=device,
            ).to(target_dtype)
            * 1000.0  # 缩放因子
        )

    def get_or_build_guidance(self, bsz: int, dtype, device):
        """
        获取引导张量，使用缓存版本（如果可用）。
        
        此方法使用_build_guidance获取缓存的引导张量。
        缓存基于批次大小、dtype、设备和引导值，
        防止在去噪循环中重复创建张量。
        
        Args:
            bsz: 批次大小
            dtype: 数据类型
            device: 设备
        
        Returns:
            引导张量或None
        """
        if self.server_args.pipeline_config.should_use_guidance:
            guidance_val = self.server_args.pipeline_config.embedded_cfg_scale
            return self._build_guidance(bsz, dtype, device, guidance_val)
        else:
            return None

    @property
    def parallelism_type(self) -> StageParallelismType:
        """获取并行类型"""
        return StageParallelismType.REPLICATED

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        """
        准备去噪循环所需的所有不变量。
        
        Returns:
            包含去噪循环准备变量的字典
        """
        assert self.transformer is not None
        pipeline = self.pipeline() if self.pipeline else None
        
        # 获取原始推理步数（用于cache-dit初始化）
        cache_dit_num_inference_steps = batch.extra.get(
            "cache_dit_num_inference_steps", batch.num_inference_steps
        )
        
        # 如果transformer未加载，加载它
        if not server_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                server_args.model_paths["transformer"], server_args, "transformer"
            )
            # 在torch.compile之前启用cache-dit（延迟挂载）
            self._maybe_enable_cache_dit(cache_dit_num_inference_steps, batch)
            self._maybe_enable_torch_compile(self.transformer)
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            server_args.model_loaded["transformer"] = True
        else:
            self._maybe_enable_cache_dit(cache_dit_num_inference_steps, batch)
        
        # 准备scheduler的额外参数
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": batch.generator, "eta": batch.eta},
        )
        
        # 设置精度和autocast
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32) and not server_args.disable_autocast
        
        # 获取时间步和计算warmup步数
        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        # 准备图像嵌入（用于I2V生成）
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            image_embeds = [image_embed.to(target_dtype) for image_embed in image_embeds]
        
        # 准备STA参数
        if self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN:
            self.prepare_sta_param(batch, server_args)
        
        # 获取latents和embeddings
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        
        # 准备负提示嵌入（用于CFG）
        neg_prompt_embeds = None
        if batch.do_classifier_free_guidance:
            neg_prompt_embeds = batch.negative_prompt_embeds
        
        # 处理边界比例（Wan2.2双Transformer切换）
        boundary_timestep = self._handle_boundary_ratio(server_args, batch)
        
        # 处理序列并行
        self._preprocess_sp_latents(batch, server_args)
        latents = batch.latents
        
        # 获取引导张量
        guidance = self.get_or_build_guidance(
            latents.shape[0], latents.dtype, latents.device,
        )
        
        # 准备正向条件参数
        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            getattr(self.transformer, "forward", self.transformer),
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            }
            | server_args.pipeline_config.prepare_pos_cond_kwargs(
                batch, self.device,
                getattr(self.transformer, "rotary_emb", None),
                dtype=target_dtype,
            )
            | dict(encoder_hidden_states=server_args.pipeline_config.get_pos_prompt_embeds(batch)),
        )
        
        # 准备负向条件参数（用于CFG）
        if batch.do_classifier_free_guidance:
            neg_cond_kwargs = self.prepare_extra_func_kwargs(
                getattr(self.transformer, "forward", self.transformer),
                {
                    "encoder_hidden_states_2": batch.clip_embedding_neg,
                    "encoder_attention_mask": batch.negative_attention_mask,
                }
                | server_args.pipeline_config.prepare_neg_cond_kwargs(
                    batch, self.device,
                    getattr(self.transformer, "rotary_emb", None),
                    dtype=target_dtype,
                )
                | dict(encoder_hidden_states=server_args.pipeline_config.get_neg_prompt_embeds(batch)),
            )
        else:
            neg_cond_kwargs = {}
        
        return {
            "extra_step_kwargs": extra_step_kwargs,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "timesteps": timesteps,
            "num_inference_steps": num_inference_steps,
            "num_warmup_steps": num_warmup_steps,
            "pos_cond_kwargs": pos_cond_kwargs,
            "neg_cond_kwargs": neg_cond_kwargs,
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "neg_prompt_embeds": neg_prompt_embeds,
            "boundary_timestep": boundary_timestep,
            "guidance": guidance,
        }

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        运行去噪循环。
        
        Args:
            batch: 要处理的批次
            server_args: 推理参数
        
        Returns:
            处理后的批次
        """
        # 准备去噪循环变量
        prepared_vars = self._prepare_denoising_loop(batch, server_args)
        extra_step_kwargs = prepared_vars["extra_step_kwargs"]
        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        num_inference_steps = prepared_vars["num_inference_steps"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        pos_cond_kwargs = prepared_vars["pos_cond_kwargs"]
        neg_cond_kwargs = prepared_vars["neg_cond_kwargs"]
        latents = prepared_vars["latents"]
        boundary_timestep = prepared_vars["boundary_timestep"]
        guidance = prepared_vars["guidance"]
        
        # 初始化ODE轨迹列表
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []
        
        # 记录去噪开始时间
        denoising_start_time = time.time()
        
        # 避免时间步比较导致的设备同步
        is_warmup = batch.is_warmup
        self.scheduler.set_begin_index(0)
        timesteps_cpu = timesteps.cpu()  # 移到CPU进行比较
        num_timesteps = timesteps_cpu.shape[0]
        
        # 使用autocast进行混合精度推理
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            # 显示进度条
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                # 遍历每个时间步
                for i, t_host in enumerate(timesteps_cpu):
                    with StageProfiler(
                        f"denoising_step_{i}",
                        logger=logger,
                        timings=batch.timings,
                        perf_dump_path_provided=batch.perf_dump_path is not None,
                    ):
                        # 获取当前时间步的整数值和设备张量
                        t_int = int(t_host.item())
                        t_device = timesteps[i]
                        
                        # 选择并管理模型（处理MoE双Transformer切换）
                        current_model, current_guidance_scale = (
                            self._select_and_manage_model(
                                t_int=t_int,
                                boundary_timestep=boundary_timestep,
                                server_args=server_args,
                                batch=batch,
                            )
                        )
                        
                        # 扩展latents用于I2V
                        latent_model_input = latents.to(target_dtype)
                        if batch.image_latent is not None:
                            latent_model_input = torch.cat(
                                [latent_model_input, batch.image_latent], dim=1
                            ).to(target_dtype)
                        
                        # 缩放模型输入
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t_device
                        )
                        
                        # 构建注意力元数据
                        attn_metadata = self._build_attn_metadata(i, batch, server_args)
                        
                        # 预测噪声残差（带CFG）
                        noise_pred = self._predict_noise_with_cfg(
                            current_model=current_model,
                            latent_model_input=latent_model_input,
                            timestep=t_device,
                            batch=batch,
                            timestep_index=i,
                            attn_metadata=attn_metadata,
                            target_dtype=target_dtype,
                            current_guidance_scale=current_guidance_scale,
                            pos_cond_kwargs=pos_cond_kwargs,
                            neg_cond_kwargs=neg_cond_kwargs,
                            server_args=server_args,
                            guidance=guidance,
                            latents=latents,
                        )
                        
                        # 计算前一个噪声样本
                        latents = self.scheduler.step(
                            model_output=noise_pred,
                            timestep=t_device,
                            sample=latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]
                        
                        # 如果需要，保存轨迹latents
                        if batch.return_trajectory_latents:
                            trajectory_timesteps.append(t_host)
                            trajectory_latents.append(latents)
                        
                        # 更新进度条
                        if i == num_timesteps - 1 or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()
                        
                        # 性能分析
                        if not is_warmup:
                            self.step_profile()
        
        # 记录去噪结束时间
        denoising_end_time = time.time()
        
        # 记录平均步时间
        if num_timesteps > 0 and not is_warmup:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_end_time - denoising_start_time) / len(timesteps),
            )
        
        # 后处理去噪循环
        self._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            server_args=server_args,
            is_warmup=is_warmup,
        )
        return batch

    def _predict_noise_with_cfg(
        self,
        current_model: nn.Module,
        latent_model_input: torch.Tensor,
        timestep,
        batch: Req,
        timestep_index: int,
        attn_metadata,
        target_dtype,
        current_guidance_scale,
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        server_args,
        guidance,
        latents,
    ):
        """
        使用Classifier-Free Guidance预测噪声残差。
        
        这是去噪阶段的核心函数，执行以下步骤：
        1. 正向传播（条件预测）
        2. 负向传播（无条件预测，如果启用CFG）
        3. 合并预测结果
        
        Args:
            current_model: 当前使用的transformer模型
            latent_model_input: 输入latents
            timestep: 时间步
            batch: 当前批次
            timestep_index: 时间步索引
            attn_metadata: 注意力元数据
            target_dtype: 目标数据类型
            current_guidance_scale: 当前引导比例
            pos_cond_kwargs: 正向条件参数
            neg_cond_kwargs: 负向条件参数
            server_args: 服务器参数
            guidance: 引导张量
            latents: 原始latents
        
        Returns:
            预测的噪声
        """
        noise_pred_cond: torch.Tensor | None = None
        noise_pred_uncond: torch.Tensor | None = None
        cfg_rank = get_classifier_free_guidance_rank()
        
        # ===== 正向传播（条件预测） =====
        if not (server_args.enable_cfg_parallel and cfg_rank != 0):
            batch.is_cfg_negative = False
            with set_forward_context(
                current_timestep=timestep_index,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ):
                noise_pred_cond = self._predict_noise(
                    current_model=current_model,
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    target_dtype=target_dtype,
                    guidance=guidance,
                    **pos_cond_kwargs,
                )
                # 切片噪声预测（某些模型需要）
                noise_pred_cond = server_args.pipeline_config.slice_noise_pred(
                    noise_pred_cond, latents
                )
        
        # 如果禁用CFG，直接返回条件预测
        if not batch.do_classifier_free_guidance:
            return noise_pred_cond
        
        # ===== 负向传播（无条件预测） =====
        if not server_args.enable_cfg_parallel or cfg_rank != 0:
            batch.is_cfg_negative = True
            with set_forward_context(
                current_timestep=timestep_index,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ):
                noise_pred_uncond = self._predict_noise(
                    current_model=current_model,
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    target_dtype=target_dtype,
                    guidance=guidance,
                    **neg_cond_kwargs,
                )
                noise_pred_uncond = server_args.pipeline_config.slice_noise_pred(
                    noise_pred_uncond, latents
                )
        
        # ===== 合并预测结果 =====
        if server_args.enable_cfg_parallel:
            # CFG并行模式：每个rank计算部分贡献，通过all-reduce求和
            # final = s*cond + (1-s)*uncond
            if cfg_rank == 0:
                assert noise_pred_cond is not None
                partial = current_guidance_scale * noise_pred_cond
            else:
                assert noise_pred_uncond is not None
                partial = (1 - current_guidance_scale) * noise_pred_uncond
            
            noise_pred = cfg_model_parallel_all_reduce(partial)
            
            # CFG归一化
            if batch.cfg_normalization and float(batch.cfg_normalization) > 0:
                factor = float(batch.cfg_normalization)
                pred_f = noise_pred.float()
                new_norm = torch.linalg.vector_norm(pred_f)
                if cfg_rank == 0:
                    cond_f = noise_pred_cond.float()
                    ori_norm = torch.linalg.vector_norm(cond_f)
                else:
                    ori_norm = torch.empty_like(new_norm)
                ori_norm = get_cfg_group().broadcast(ori_norm, src=0)
                max_norm = ori_norm * factor
                
                if new_norm > max_norm:
                    noise_pred = noise_pred * (max_norm / new_norm)
            
            # 引导重缩放
            if batch.guidance_rescale > 0.0:
                std_cfg = noise_pred.std(
                    dim=list(range(1, noise_pred.ndim)), keepdim=True
                )
                if cfg_rank == 0:
                    assert noise_pred_cond is not None
                    std_text = noise_pred_cond.std(
                        dim=list(range(1, noise_pred_cond.ndim)), keepdim=True
                    )
                else:
                    std_text = torch.empty_like(std_cfg)
                std_text = get_cfg_group().broadcast(std_text, src=0)
                noise_pred_rescaled = noise_pred * (std_text / std_cfg)
                noise_pred = (
                    batch.guidance_rescale * noise_pred_rescaled
                    + (1 - batch.guidance_rescale) * noise_pred
                )
            return noise_pred
        else:
            # 串行CFG：两个预测都在本地可用
            assert noise_pred_cond is not None and noise_pred_uncond is not None
            noise_pred = noise_pred_uncond + current_guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            
            # CFG归一化
            if batch.cfg_normalization and float(batch.cfg_normalization) > 0:
                factor = float(batch.cfg_normalization)
                cond_f = noise_pred_cond.float()
                pred_f = noise_pred.float()
                ori_norm = torch.linalg.vector_norm(cond_f)
                new_norm = torch.linalg.vector_norm(pred_f)
                max_norm = ori_norm * factor
                
                if new_norm > max_norm:
                    noise_pred = noise_pred * (max_norm / new_norm)
            
            # 引导重缩放
            if batch.guidance_rescale > 0.0:
                noise_pred = self.rescale_noise_cfg(
                    noise_pred,
                    noise_pred_cond,
                    guidance_rescale=batch.guidance_rescale,
                )
            return noise_pred
```

---

## 4. Qwen-Image 加速实现

### 4.1 Qwen-Image Pipeline 定义

```python
# 文件: python/sglang/multimodal_gen/runtime/pipelines/qwen_image.py

"""
Qwen-Image Pipeline Implementation
Qwen-Image Pipeline实现
"""

from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,           # 解码阶段
    DenoisingStage,          # 去噪阶段
    ImageEncodingStage,      # 图像编码阶段
    ImageVAEEncodingStage,   # 图像VAE编码阶段
    InputValidationStage,    # 输入验证阶段
    LatentPreparationStage,  # Latent准备阶段
    TextEncodingStage,       # 文本编码阶段
    TimestepPreparationStage, # 时间步准备阶段
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.conditioning import (
    ConditioningStage,       # 条件准备阶段
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """
    计算shift参数（用于Flow Matching调度器）。
    
    Qwen-Image使用Flow Matching作为其调度器，需要根据图像序列长度
    动态调整shift参数以获得更好的生成质量。
    
    Args:
        image_seq_len: 图像序列长度
        base_seq_len: 基础序列长度
        max_seq_len: 最大序列长度
        base_shift: 基础shift值
        max_shift: 最大shift值
    
    Returns:
        计算得到的mu值
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    """
    准备mu参数（用于Qwen-Image的调度器）。
    
    Args:
        batch: 当前批次
        server_args: 服务器参数
    
    Returns:
        ("mu", mu_value) 元组
    """
    height = batch.height
    width = batch.width
    vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
    
    # 计算图像序列长度
    image_seq_len = (int(height) // vae_scale_factor // 2) * (
        int(width) // vae_scale_factor // 2
    )
    
    # 计算shift参数
    mu = calculate_shift(
        image_seq_len,
        256,    # 基础序列长度
        8192,   # 最大序列长度
        0.5,    # 基础shift
        0.9,    # 最大shift
    )
    return "mu", mu


class QwenImagePipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Qwen-Image文生图Pipeline。
    
    这是Qwen-Image模型的标准文生图Pipeline实现。
    """
    
    pipeline_name = "QwenImagePipeline"
    
    # 必需的配置模块
    _required_config_modules = [
        "text_encoder",   # 文本编码器
        "tokenizer",      # 分词器
        "vae",            # VAE
        "transformer",    # DiT/Transformer
        "scheduler",      # 调度器
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """
        创建Pipeline Stages（依赖注入）。
        
        Args:
            server_args: 服务器参数
        """
        # Stage 1: 输入验证
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        
        # Stage 2: 文本编码
        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        
        # Stage 3: 条件准备
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        
        # Stage 4: 时间步准备
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],  # 准备mu参数
            ),
        )
        
        # Stage 5: Latent准备
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )
        
        # Stage 6: 去噪（核心阶段）
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        
        # Stage 7: 解码
        self.add_stage(
            stage_name="decoding_stage", 
            stage=DecodingStage(vae=self.get_module("vae"))
        )


class QwenImageEditPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Qwen-Image-Edit图像编辑Pipeline。
    
    这是Qwen-Image-Edit模型的图像编辑Pipeline实现，支持：
    - 基于文本指令的图像编辑
    - 图像到图像的转换
    """
    
    pipeline_name = "QwenImageEditPipeline"
    
    # 必需的配置模块（包含processor用于处理输入图像）
    _required_config_modules = [
        "processor",      # 图像处理器
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """创建Pipeline Stages（依赖注入）"""
        
        # Stage 1: 输入验证（带VAE图像处理器）
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(
                vae_image_processor=VaeImageProcessor(
                    vae_scale_factor=server_args.pipeline_config.vae_config.arch_config.vae_scale_factor * 2
                )
            ),
        )
        
        # Stage 2: 图像编码（同时编码文本和图像）
        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=ImageEncodingStage(
                image_processor=self.get_module("processor"),
                text_encoder=self.get_module("text_encoder"),
            ),
        )
        
        # Stage 3: 图像VAE编码（编码输入图像到latent空间）
        self.add_stage(
            stage_name="image_encoding_stage_primary",
            stage=ImageVAEEncodingStage(
                vae=self.get_module("vae"),
            ),
        )
        
        # Stage 4: 时间步准备
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[prepare_mu],
            ),
        )
        
        # Stage 5: Latent准备
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )
        
        # Stage 6: 条件准备
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        
        # Stage 7: 去噪
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        
        # Stage 8: 解码
        self.add_stage(
            stage_name="decoding_stage", 
            stage=DecodingStage(vae=self.get_module("vae"))
        )
```

---

## 5. 底层加速原理

### 5.1 Cache-DiT 加速机制

Cache-DiT 是 SGLang-Diffusion 中最重要的加速技术之一，最高可带来 **169%** 的速度提升。

#### 核心原理

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cache-DiT 加速原理                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传统DiT: 每个时间步都计算所有Block                               │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐              │
│  │Block│ → │Block│ → │Block│ → │Block│ → │Block│  (N steps)   │
│  │  1  │   │  2  │   │  3  │   │  4  │   │  5  │              │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘              │
│                                                                  │
│  Cache-DiT: 只计算必要的Block，复用缓存结果                        │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐              │
│  │Block│ → │Cache│ → │Block│ → │Cache│ → │Block│  (N steps)   │
│  │  1  │   │  ✓  │   │  3  │   │  ✓  │   │  5  │              │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘              │
│                                                                  │
│  关键参数:                                                        │
│  • Fn: 前Fn个block始终计算                                        │
│  • Bn: 后Bn个block始终计算                                        │
│  • RDT: 残差差异阈值，决定何时跳过计算                              │
│  • TaylorSeer: 使用泰勒展开估计跳过block的输出                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 环境变量配置

```bash
# 启用 Cache-DiT
export SGLANG_CACHE_DIT_ENABLED=true

# 前Fn个block计算（默认3）
export SGLANG_CACHE_DIT_FN=3

# 后Bn个block计算（默认3）
export SGLANG_CACHE_DIT_BN=3

# 残差差异阈值（默认0.12）
export SGLANG_CACHE_DIT_RDT=0.12

# SCM预设（fast/balanced/quality）
export SGLANG_CACHE_DIT_SCM_PRESET=fast

# 启用 TaylorSeer
export SGLANG_CACHE_DIT_TAYLORSEER=true
```

### 5.2 并行策略详解

#### 5.2.1 USP (Unified Sequence Parallelism)

USP 结合了 Ulysses-SP 和 Ring-Attention 两种序列并行技术：

```
┌─────────────────────────────────────────────────────────────────┐
│                     USP 统一序列并行                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Ulysses-SP: 在序列维度上分割注意力计算                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ GPU 0   │    │ GPU 1   │    │ GPU 2   │    │ GPU 3   │      │
│  │ [S/4]   │    │ [S/4]   │    │ [S/4]   │    │ [S/4]   │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│       ↓              ↓              ↓              ↓            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │AllGather│ ←→ │AllGather│ ←→ │AllGather│ ←→ │AllGather│      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│                                                                  │
│  Ring-Attention: 环形通信减少内存使用                              │
│  ┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐                │
│  │GPU 0│ ───→ │GPU 1│ ───→ │GPU 2│ ───→ │GPU 3│                │
│  │     │ ←─── │     │ ←─── │     │ ←─── │     │                │
│  └─────┘      └─────┘      └─────┘      └─────┘                │
│                                                                  │
│  优势:                                                           │
│  • 支持超长序列（>100K tokens）                                   │
│  • 内存使用与GPU数量成反比                                         │
│  • 计算和通信重叠                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 CFG Parallel

CFG (Classifier-Free Guidance) 并行同时计算条件和无条件分支：

```
┌─────────────────────────────────────────────────────────────────┐
│                     CFG Parallel 原理                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传统CFG: 串行计算                                                 │
│  ┌─────────┐         ┌─────────┐                                │
│  │  Cond   │ ──────→ │ Uncond  │  (串行，2x时间)                 │
│  │ Forward │         │ Forward │                                │
│  └─────────┘         └─────────┘                                │
│                                                                  │
│  CFG Parallel: 并行计算                                           │
│  ┌─────────┐         ┌─────────┐                                │
│  │  Cond   │ ═══════ │ Uncond  │  (并行，1x时间)                 │
│  │ Forward │  AllReduce Forward                                │
│  └─────────┘         └─────────┘                                │
│       │                   │                                     │
│       └───────┬───────────┘                                     │
│               ↓                                                 │
│         noise_pred = uncond + scale * (cond - uncond)           │
│                                                                  │
│  优势: 2x速度提升（在2+ GPU上）                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 注意力后端优化

SGLang-Diffusion 支持多种注意力后端：

| 后端 | 适用场景 | 加速效果 |
|------|----------|----------|
| **FlashAttention 2/3** | 通用场景 | 2-4x |
| **SageAttention** | 长序列 | 3-5x |
| **Sliding Tile Attention** | 视频生成 | 2-3x |
| **Video Sparse Attention** | 稀疏视频注意力 | 3-5x |

---

## 6. 快速执行脚本

### 6.1 安装脚本

```bash
#!/bin/bash
# 文件: install_sglang_diffusion.sh
# 描述: SGLang-Diffusion 安装脚本

echo "======================================"
echo "SGLang-Diffusion 安装脚本"
echo "======================================"

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 安装uv（如果未安装）
if ! command -v uv &> /dev/null; then
    echo "安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 安装SGLang-Diffusion
echo "安装 SGLang-Diffusion..."
uv pip install 'sglang[diffusion]' --prerelease=allow

# 验证安装
echo "验证安装..."
python3 -c "import sglang; print('SGLang版本:', sglang.__version__)"

echo "安装完成！"
```

### 6.2 Qwen-Image 文生图脚本

```bash
#!/bin/bash
# 文件: generate_qwen_image.sh
# 描述: Qwen-Image 文生图快速生成脚本

# 模型路径
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen-Image"}

# 提示词
PROMPT=${PROMPT:-"一只可爱的猫咪在草地上玩耍"}

# 图像尺寸
WIDTH=${WIDTH:-1024}
HEIGHT=${HEIGHT:-1024}

# 推理步数
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-30}

# 引导比例
GUIDANCE_SCALE=${GUIDANCE_SCALE:-2.5}

# 输出路径
OUTPUT_PATH=${OUTPUT_PATH:-"./output.png"}

echo "======================================"
echo "Qwen-Image 文生图生成"
echo "======================================"
echo "模型: $MODEL_PATH"
echo "提示词: $PROMPT"
echo "尺寸: ${WIDTH}x${HEIGHT}"
echo "步数: $NUM_INFERENCE_STEPS"
echo "引导比例: $GUIDANCE_SCALE"
echo "======================================"

# 执行生成
sglang generate \
    --model-path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --save-output "$OUTPUT_PATH"

echo "生成完成！输出: $OUTPUT_PATH"
```

### 6.3 Qwen-Image-Edit 图像编辑脚本

```bash
#!/bin/bash
# 文件: edit_qwen_image.sh
# 描述: Qwen-Image-Edit 图像编辑快速脚本

# 模型路径
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen-Image-Edit-2511"}

# 输入图像路径
IMAGE_PATH=${IMAGE_PATH:-"./input.jpg"}

# 编辑指令
PROMPT=${PROMPT:-"将背景改为海滩"}

# 输出路径
OUTPUT_PATH=${OUTPUT_PATH:-"./edited_output.png"}

echo "======================================"
echo "Qwen-Image-Edit 图像编辑"
echo "======================================"
echo "模型: $MODEL_PATH"
echo "输入图像: $IMAGE_PATH"
echo "编辑指令: $PROMPT"
echo "======================================"

# 执行编辑
sglang generate \
    --model-path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image-path "$IMAGE_PATH" \
    --save-output "$OUTPUT_PATH"

echo "编辑完成！输出: $OUTPUT_PATH"
```

### 6.4 高性能加速脚本（多GPU + Cache-DiT）

```bash
#!/bin/bash
# 文件: generate_accelerated.sh
# 描述: 启用所有加速技术的高性能生成脚本

# 启用Cache-DiT加速
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=3
export SGLANG_CACHE_DIT_BN=3
export SGLANG_CACHE_DIT_RDT=0.12
export SGLANG_CACHE_DIT_SCM_PRESET=fast
export SGLANG_CACHE_DIT_TAYLORSEER=true

# 启用torch.compile
export SGLANG_ENABLE_TORCH_COMPILE=true
export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs

# 模型和生成参数
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen-Image"}
PROMPT=${PROMPT:-"一只可爱的猫咪在草地上玩耍"}
NUM_GPUS=${NUM_GPUS:-1}
ENABLE_CFG_PARALLEL=${ENABLE_CFG_PARALLEL:-false}

echo "======================================"
echo "SGLang-Diffusion 高性能生成"
echo "======================================"
echo "加速技术:"
echo "  - Cache-DiT: ENABLED"
echo "  - torch.compile: ENABLED"
echo "  - GPU数量: $NUM_GPUS"
echo "  - CFG并行: $ENABLE_CFG_PARALLEL"
echo "======================================"

# 构建命令
CMD="sglang generate \
    --model-path $MODEL_PATH \
    --prompt \"$PROMPT\" \
    --num-gpus $NUM_GPUS"

# 如果GPU数量大于1且启用CFG并行
if [ "$NUM_GPUS" -gt 1 ] && [ "$ENABLE_CFG_PARALLEL" = true ]; then
    CMD="$CMD --enable-cfg-parallel"
fi

# 执行命令
eval $CMD

echo "======================================"
echo "生成完成！"
echo "======================================"
```

### 6.5 Python API 使用示例

```python
#!/usr/bin/env python3
# 文件: python_api_example.py
# 描述: SGLang-Diffusion Python API 使用示例

import sglang
from sglang.multimodal_gen import generate_image

def text_to_image_example():
    """文生图示例"""
    
    # 设置生成参数
    result = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt="一只可爱的猫咪在草地上玩耍，阳光明媚，高清细节",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=2.5,
        seed=42,  # 设置随机种子以获得可复现的结果
    )
    
    # 保存生成的图像
    result.save("output.png")
    print("图像已保存到 output.png")


def image_edit_example():
    """图像编辑示例"""
    
    result = generate_image(
        model_path="Qwen/Qwen-Image-Edit-2511",
        prompt="将背景改为海滩",
        image_path="input.jpg",
        num_inference_steps=30,
    )
    
    result.save("edited_output.png")
    print("编辑后的图像已保存到 edited_output.png")


def batch_generation_example():
    """批量生成示例"""
    
    prompts = [
        "一只可爱的猫咪",
        "一只可爱的狗狗",
        "一只可爱的兔子",
    ]
    
    results = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt=prompts,  # 传入列表进行批量生成
        width=1024,
        height=1024,
        num_inference_steps=30,
    )
    
    for i, result in enumerate(results):
        result.save(f"batch_output_{i}.png")
        print(f"图像 {i} 已保存")


if __name__ == "__main__":
    text_to_image_example()
    # image_edit_example()
    # batch_generation_example()
```

---

## 7. 性能优化建议

### 7.1 硬件配置建议

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| **单图生成** | 1x A100 80GB | 适合个人开发和小规模部署 |
| **批量生成** | 2-4x A100 80GB | 启用CFG并行，2x加速 |
| **视频生成** | 4-8x H100 80GB | 启用USP，支持长序列 |
| **生产部署** | 8x H100 80GB | 启用所有并行策略 |

### 7.2 软件优化建议

1. **启用 Cache-DiT**
   ```bash
   export SGLANG_CACHE_DIT_ENABLED=true
   export SGLANG_CACHE_DIT_SCM_PRESET=fast
   ```

2. **启用 torch.compile**
   ```bash
   export SGLANG_ENABLE_TORCH_COMPILE=true
   ```

3. **使用合适的注意力后端**
   ```bash
   # 长序列推荐SageAttention
   export SGLANG_ATTENTION_BACKEND=sageattention
   ```

4. **调整推理步数**
   - 质量优先: 30-50步
   - 速度优先: 8-20步（配合Lightning模型）

### 7.3 常见问题和解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| OOM错误 | 显存不足 | 启用layer-wise offload或减小batch size |
| 生成速度慢 | 未启用加速 | 检查Cache-DiT和torch.compile是否启用 |
| 图像质量差 | 步数太少 | 增加num_inference_steps |
| 结果不一致 | 随机种子未设置 | 设置seed参数 |

---

## 8. 总结

SGLang-Diffusion 通过以下核心设计实现了扩散模型的高性能推理：

1. **模块化架构**: `ComposedPipelineBase` + `PipelineStage` 设计，易于扩展
2. **多种并行策略**: USP、TP、CFG Parallel、SP，充分利用多GPU
3. **内核优化**: Cache-DiT、FlashAttention、torch.compile 等
4. **统一引擎**: 同时支持 LLM 和 Diffusion 模型

相比传统的 Diffusers 框架，SGLang-Diffusion 在保持兼容性的同时，实现了 **1.2x - 5.9x** 的加速，是生产环境部署扩散模型的理想选择。

---

## 参考资料

1. [SGLang-Diffusion 官方博客](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)
2. [SGLang GitHub](https://github.com/sgl-project/sglang)
3. [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
4. [FastVideo GitHub](https://github.com/hao-ai-lab/FastVideo)
5. [Cache-DiT GitHub](https://github.com/vipshop/cache-dit)

---

*文档版本: v1.0*  
*最后更新: 2026-02-03*
