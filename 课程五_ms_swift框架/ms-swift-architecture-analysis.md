# ms-swift 大模型训练框架 - 整体架构深度分析报告

## 一、框架概述

**ms-swift** (Scalable lightWeight Infrastructure for Fine-Tuning) 是阿里巴巴ModelScope团队开发的轻量级大模型训练推理框架。

### 核心特性
- **支持600+ LLM**: Qwen3, Qwen3-MoE, DeepSeek-R1, GLM4.5, InternLM3, Llama4等
- **支持300+ VLM**: Qwen3-VL, Qwen3-Omni, InternVL3.5, Ovis2.5, GLM4.5v, Llava, Phi4等
- **训练范式**: CPT(持续预训练)、SFT(有监督微调)、DPO、GRPO、PPO、KTO、ORPO等
- **微调方法**: LoRA、QLoRA、DoRA、Adapter、Prompt Tuning、Full Parameter等
- **推理引擎**: Transformers、vLLM、SGLang、LMDeploy
- **并行策略**: DeepSpeed、Megatron、FSDP、Sequence Parallelism

---

## 二、目录结构总览

```
swift/
├── cli/                    # 命令行接口入口
├── arguments/              # 参数定义与管理
├── callbacks/              # 训练回调函数
├── config/                 # 配置文件
├── dataloader/             # 数据加载器
├── dataset/                # 数据集处理
├── hub/                    # 模型仓库交互
├── infer_engine/           # 推理引擎封装
├── loss/                   # 损失函数
├── loss_scale/             # 损失缩放
├── megatron/               # Megatron集成
├── metrics/                # 评估指标
├── model/                  # 模型定义与元数据
├── optimizers/             # 优化器
├── pipelines/              # 核心流程实现
├── ray/                    # Ray分布式支持
├── rewards/                # RLHF奖励模型
├── rlhf_trainers/          # RLHF训练器
├── rollout/                # RLHF Rollout
├── sequence_parallel/      # 序列并行
├── template/               # 对话模板系统
├── trainers/               # 训练器封装
├── tuner_plugin/           # 微调插件系统
├── tuners/                 # 微调方法实现
├── ui/                     # Web UI界面
└── utils/                  # 工具函数
```

---

## 三、核心模块详细分析

### 3.1 CLI层 (swift/cli/)

CLI层是框架的用户入口，提供统一的命令行接口。

**核心文件:**
- `main.py`: 主入口，路由分发到各个子命令
- `sft.py`: 有监督微调命令
- `pt.py`: 预训练/持续预训练命令
- `rlhf.py`: RLHF训练命令
- `infer.py`: 推理命令
- `export.py`: 模型导出命令
- `eval.py`: 模型评估命令
- `deploy.py`: 模型部署命令
- `app.py`: 交互式应用命令
- `merge_lora.py`: LoRA权重合并
- `web_ui.py`: Web界面启动

**路由映射 (main.py):**
```python
ROUTE_MAPPING = {
    'pt': 'swift.cli.pt',           # 预训练
    'sft': 'swift.cli.sft',         # 有监督微调
    'infer': 'swift.cli.infer',     # 推理
    'merge-lora': 'swift.cli.merge_lora',
    'web-ui': 'swift.cli.web_ui',
    'deploy': 'swift.cli.deploy',
    'rollout': 'swift.cli.rollout',
    'rlhf': 'swift.cli.rlhf',       # RLHF训练
    'sample': 'swift.cli.sample',
    'export': 'swift.cli.export',
    'eval': 'swift.cli.eval',
    'app': 'swift.cli.app',
}
```

---

### 3.2 Pipelines层 (swift/pipelines/)

Pipelines层是框架的核心业务逻辑层，实现各种训练推理流程。

**目录结构:**
```
pipelines/
├── base.py              # Pipeline基类
├── utils.py             # Pipeline工具函数
├── app/                 # 交互式应用
├── eval/                # 评估流程
├── export/              # 导出流程
├── infer/               # 推理流程
├── sampling/            # 采样流程
└── train/               # 训练流程
    ├── sft.py           # SFT训练
    ├── pretrain.py      # 预训练
    ├── rlhf.py          # RLHF训练
    ├── kto.py           # KTO训练
    └── tuner.py         # 微调器
```

**核心函数:**
- `sft_main()`: SFT训练主函数
- `pretrain_main()`: 预训练主函数
- `rlhf_main()`: RLHF训练主函数
- `infer_main()`: 推理主函数
- `export_main()`: 导出主函数
- `eval_main()`: 评估主函数

---

### 3.3 Trainers层 (swift/trainers/)

Trainers层封装了HuggingFace Trainer，提供统一的训练接口。

**核心文件:**
- `trainer.py`: 基础Trainer类
- `seq2seq_trainer.py`: 序列到序列Trainer
- `embedding_trainer.py`: Embedding模型Trainer
- `reranker_trainer.py`: Reranker模型Trainer
- `mixin.py`: Trainer功能混合类(SwiftMixin)
- `trainer_factory.py`: Trainer工厂类
- `arguments.py`: 训练参数定义
- `patcher.py`: Trainer补丁

**类继承关系:**
```
Trainer (HuggingFace)
    └── SwiftMixin
            └── Seq2SeqTrainer
            └── EmbeddingTrainer
            └── RerankerTrainer
```

**SwiftMixin核心功能:**
- 学习率调度
- 损失缩放
- 梯度累积
- 检查点保存
- 日志记录
- LoRA适配器管理

---

### 3.4 RLHF Trainers层 (swift/rlhf_trainers/)

RLHF Trainers层实现各种RLHF训练算法。

**核心文件:**
- `dpo_trainer.py`: DPO (Direct Preference Optimization) 训练器
- `ppo_trainer.py`: PPO (Proximal Policy Optimization) 训练器
- `grpo_trainer.py`: GRPO (Group Relative Policy Optimization) 训练器
- `kto_trainer.py`: KTO (Kahneman-Tversky Optimization) 训练器
- `orpo_trainer.py`: ORPO (Odds Ratio Preference Optimization) 训练器
- `cpo_trainer.py`: CPO (Contrastive Preference Optimization) 训练器
- `gkd_trainer.py`: GKD (Generalized Knowledge Distillation) 训练器
- `reward_trainer.py`: 奖励模型训练器
- `rlhf_mixin.py`: RLHF功能混合类
- `rollout_mixin.py`: Rollout功能混合类
- `vllm_client.py`: vLLM客户端

**RLHF算法支持:**
| 算法 | 文件 | 说明 |
|------|------|------|
| DPO | dpo_trainer.py | 直接偏好优化 |
| PPO | ppo_trainer.py | 近端策略优化 |
| GRPO | grpo_trainer.py | 组相对策略优化 |
| KTO | kto_trainer.py | KT优化 |
| ORPO | orpo_trainer.py | 赔率比偏好优化 |
| CPO | cpo_trainer.py | 对比偏好优化 |
| GKD | gkd_trainer.py | 广义知识蒸馏 |

---

### 3.5 Template系统 (swift/template/)

Template系统是ms-swift的核心创新之一，用于处理不同模型的对话格式。

**核心文件:**
- `base.py`: Template基类
- `template_meta.py`: 模板元数据
- `template_inputs.py`: 模板输入定义
- `register.py`: 模板注册器
- `utils.py`: 模板工具函数
- `vision_utils.py`: 视觉模板工具
- `grounding.py`: Grounding模板
- `constant.py`: 模板常量

**模板目录 (templates/):**
```
templates/
├── llm.py          # 通用LLM模板
├── qwen.py         # Qwen系列模板
├── llama.py        # Llama系列模板
├── deepseek.py     # DeepSeek模板
├── glm.py          # GLM系列模板
├── internlm.py     # InternLM模板
├── internvl.py     # InternVL模板
├── llava.py        # Llava模板
├── gemma.py        # Gemma模板
├── yi.py           # Yi模板
├── baai.py         # BAAI模板
├── baidu.py        # 百度模板
├── microsoft.py    # Microsoft模板
├── mistral.py      # Mistral模板
└── ... (30+ 模板文件)
```

**模板注册机制:**
```python
# 通过装饰器注册模板
@register_template('qwen')
class QwenTemplate(Template):
    system_prefix = '<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'
    user_prefix = '<|im_start|>user\n{{QUERY}}<|im_end|>\n'
    assistant_prefix = '<|im_start|>assistant\n{{RESPONSE}}<|im_end|>\n'
```

---

### 3.6 Tuners层 (swift/tuners/)

Tuners层实现各种参数高效微调(PEFT)方法。

**核心文件:**
- `base.py`: Tuner基类
- `lora.py`: LoRA实现
- `lora_layers.py`: LoRA层定义
- `adapter.py`: Adapter实现
- `prompt.py`: Prompt Tuning实现
- `peft.py`: PEFT集成
- `neftune.py`: NEFTune实现
- `reft.py`: ReFT实现
- `llamapro.py`: Llama-Pro实现
- `restuning.py`: ResTuning实现
- `side.py`: Side-Tuning实现
- `part.py`: 部分参数微调
- `mapping.py`: 映射工具

**子目录:**
- `longlora/`: LongLoRA长上下文扩展
- `scetuning/`: SCE-Tuning实现

**支持的微调方法:**
| 方法 | 文件 | 说明 |
|------|------|------|
| LoRA | lora.py | 低秩适应 |
| QLoRA | lora.py + 量化 | 量化LoRA |
| DoRA | lora.py | 权重分解LoRA |
| Adapter | adapter.py | 适配器微调 |
| Prompt Tuning | prompt.py | 提示微调 |
| P-Tuning | prompt.py | P-Tuning v2 |
| NEFTune | neftune.py | 噪声嵌入微调 |
| ReFT | reft.py | 表示微调 |
| Llama-Pro | llamapro.py | 层扩展微调 |

---

### 3.7 Model层 (swift/model/)

Model层管理模型元数据和架构信息。

**核心文件:**
- `model_meta.py`: 模型元数据定义
- `model_arch.py`: 模型架构定义
- `register.py`: 模型注册器
- `patcher.py`: 模型补丁
- `npu_patcher.py`: NPU补丁
- `utils.py`: 模型工具函数
- `constant.py`: 模型常量

**模型目录 (models/):**
包含600+ LLM和300+ VLM的模型定义文件。

**模型注册机制:**
```python
@register_model('qwen2-7b-instruct')
def get_qwen2_7b_instruct_model():
    return ModelMeta(
        model_id='qwen/Qwen2-7B-Instruct',
        template='qwen',
        arch='qwen2',
        task='chat'
    )
```

---

### 3.8 Dataset层 (swift/dataset/)

Dataset层处理数据加载和预处理。

**核心功能:**
- 支持多种数据格式 (JSON, JSONL, Parquet, CSV)
- 数据集混合与采样
- 数据预处理和编码
- 数据集注册机制

**核心类:**
- `EncodePreprocessor`: 编码预处理器
- `load_dataset()`: 数据集加载函数

---

### 3.9 Infer Engine层 (swift/infer_engine/)

Infer Engine层封装多种推理引擎。

**支持的引擎:**
| 引擎 | 类名 | 说明 |
|------|------|------|
| Transformers | TransformersEngine | HF原生推理 |
| vLLM | VllmEngine | 高性能推理 |
| SGLang | SglangEngine | 结构化生成 |
| LMDeploy | LmdeployEngine | 生产级部署 |
| GRPO vLLM | GRPOVllmEngine | RLHF专用 |

**核心类:**
- `InferEngine`: 推理引擎基类
- `InferRequest`: 推理请求
- `RequestConfig`: 请求配置
- `AdapterRequest`: 适配器请求

---

### 3.10 Arguments层 (swift/arguments/)

Arguments层管理所有训练推理参数。

**核心参数类:**
- `BaseArguments`: 基础参数
- `PretrainArguments`: 预训练参数
- `SftArguments`: SFT参数
- `RLHFArguments`: RLHF参数
- `InferArguments`: 推理参数
- `ExportArguments`: 导出参数
- `EvalArguments`: 评估参数
- `DeployArguments`: 部署参数
- `AppArguments`: 应用参数

---

## 四、框架架构图

### 4.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Interface Layer                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │  swift   │ │ swift    │ │ swift    │ │ swift    │ │  swift       │  │
│  │  sft     │ │  infer   │ │  rlhf    │ │  export  │ │  deploy      │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
└───────┼────────────┼────────────┼────────────┼──────────────┼──────────┘
        │            │            │            │              │
        └────────────┴────────────┴────────────┴──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      CLI Layer (swift/cli)  │
                    │  - main.py (路由)            │
                    │  - sft.py, infer.py, etc.   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Pipeline Layer            │
                    │  (swift/pipelines)          │
                    │  - train/sft.py             │
                    │  - infer/infer.py           │
                    │  - export/export.py         │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐    ┌────────────▼────────────┐   ┌────────▼───────┐
│  Trainer Layer │    │    Template Layer       │   │  Model Layer   │
│(swift/trainers)│    │  (swift/template)       │   │ (swift/model)  │
│                │    │                         │   │                │
│- Seq2SeqTrainer│    │ - Template base class   │   │ - ModelMeta    │
│- SwiftMixin    │    │ - 30+ model templates   │   │ - ModelArch    │
│- RLHF Trainers │    │ - Register mechanism    │   │ - 600+ models  │
└───────┬────────┘    └────────────┬────────────┘   └───────┬────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐    ┌────────────▼────────────┐   ┌────────▼───────┐
│   Tuner Layer  │    │    Dataset Layer        │   │  Infer Engine  │
│ (swift/tuners) │    │  (swift/dataset)        │   │(swift/infer_  │
│                │    │                         │   │    engine)     │
│- LoRA/QLoRA    │    │ - load_dataset          │   │                │
│- Adapter       │    │ - EncodePreprocessor    │   │- Transformers  │
│- Prompt Tuning │    │ - Dataset registry      │   │- vLLM          │
│- NEFTune       │    │                         │   │- SGLang        │
└───────┬────────┘    └─────────────────────────┘   └────────────────┘
        │
        │
┌───────▼──────────────────────────────────────────────────────────────┐
│                    Backend Layer (PyTorch/Transformers)              │
│  - HuggingFace Transformers  - DeepSpeed  - Megatron  - vLLM        │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 训练流程数据流图

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │───▶│  Template   │───▶│ DataLoader  │───▶│   Model     │
│   (raw)     │    │ (format)    │    │ (batch)     │    │  (forward)  │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                 │
                    ┌────────────────────────────────────────────┘
                    │
┌─────────────┐    ▼    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Optimizer │◀──Loss──│   Tuner     │◀───│  Trainer    │◀───│  Gradient   │
│  (AdamW...) │         │(LoRA/Adapter)│    │(HF+Swift)   │    │ (backward)  │
└─────────────┘         └─────────────┘    └─────────────┘    └─────────────┘
```

### 4.3 RLHF训练流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RLHF Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │  SFT Model   │────▶│  Reward Model │     │   Policy     │            │
│  │  (Reference) │     │  (Training)   │────▶│  (Training)  │            │
│  └──────────────┘     └──────────────┘     └──────┬───────┘            │
│                                                    │                     │
│                              ┌─────────────────────┘                     │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   Rollout    │────▶│   Reward     │────▶│    PPO/GRPO  │            │
│  │  (vLLM)      │     │  (Scoring)   │     │   Update     │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 五、设计哲学与架构特点

### 5.1 模块化设计

**设计原则:**
1. **单一职责**: 每个模块负责明确的功能
2. **接口隔离**: 通过基类和抽象类定义清晰接口
3. **依赖倒置**: 高层模块不依赖低层模块，都依赖抽象

**模块划分:**
- CLI层: 用户交互
- Pipeline层: 业务流程
- Trainer层: 训练逻辑
- Template层: 数据处理
- Tuner层: 参数微调
- Model层: 模型定义

### 5.2 插件化架构

**注册机制:**
```python
# 模板注册
templates_map = {}
def register_template(name):
    def decorator(cls):
        templates_map[name] = cls
        return cls
    return decorator

# 模型注册
models_map = {}
def register_model(name):
    def decorator(fn):
        models_map[name] = fn
        return fn
    return decorator
```

**扩展方式:**
1. 继承基类
2. 实现抽象方法
3. 使用装饰器注册

### 5.3 配置驱动

**配置层次:**
1. 命令行参数
2. YAML配置文件
3. 环境变量
4. 默认值

**参数继承:**
```
BaseArguments
    ├── PretrainArguments
    ├── SftArguments
    ├── RLHFArguments
    ├── InferArguments
    ├── ExportArguments
    └── EvalArguments
```

### 5.4 多引擎支持

**推理引擎抽象:**
```python
class InferEngine(ABC):
    @abstractmethod
    def infer(self, requests: List[InferRequest]) -> List[Response]:
        pass

class TransformersEngine(InferEngine):
    # HF实现
    
class VllmEngine(InferEngine):
    # vLLM实现
    
class SglangEngine(InferEngine):
    # SGLang实现
```

---

## 六、扩展机制详解

### 6.1 添加新模型

**步骤:**
1. 在 `swift/model/models/` 创建模型定义文件
2. 使用 `@register_model` 装饰器注册
3. 在 `swift/template/templates/` 创建对应模板
4. 使用 `@register_template` 装饰器注册

**示例:**
```python
# swift/model/models/my_model.py
from swift.model import register_model, ModelMeta

@register_model('my-model-7b')
def get_my_model():
    return ModelMeta(
        model_id='org/my-model-7b',
        template='my_model',
        arch='my_arch',
        task='chat'
    )

# swift/template/templates/my_model.py
from swift.template import register_template, Template

@register_template('my_model')
class MyModelTemplate(Template):
    system_prefix = '<s>[SYSTEM]\n{{SYSTEM}}[/SYSTEM]\n'
    user_prefix = '[USER]\n{{QUERY}}[/USER]\n'
    assistant_prefix = '[ASSISTANT]\n{{RESPONSE}}[/ASSISTANT]\n'
```

### 6.2 添加新微调方法

**步骤:**
1. 在 `swift/tuners/` 创建新的tuner文件
2. 继承 `Tuner` 基类
3. 实现 `forward` 和 `backward` 方法
4. 在 `mapping.py` 中注册

**示例:**
```python
# swift/tuners/my_tuner.py
from swift.tuners.base import Tuner

class MyTuner(Tuner):
    def __init__(self, model, config):
        super().__init__(model, config)
        # 初始化逻辑
    
    def forward(self, *args, **kwargs):
        # 前向逻辑
        pass
```

### 6.3 添加新损失函数

**步骤:**
1. 在 `swift/loss/` 创建新的损失函数文件
2. 继承 `BaseLoss` 基类
3. 在 `loss_map` 中注册

### 6.4 添加新回调函数

**步骤:**
1. 在 `swift/callbacks/` 创建新的回调文件
2. 继承 `TrainerCallback` 基类
3. 在 `callbacks_map` 中注册

---

## 七、支持的模型类型

### 7.1 LLM支持 (600+)

**主流模型系列:**
| 系列 | 代表模型 | 模板文件 |
|------|----------|----------|
| Qwen | Qwen3, Qwen2.5, Qwen2 | qwen.py |
| Llama | Llama4, Llama3, Llama2 | llama.py |
| DeepSeek | DeepSeek-R1, DeepSeek-V3 | deepseek.py |
| GLM | GLM4.5, GLM4, ChatGLM3 | glm.py |
| InternLM | InternLM3, InternLM2.5 | internlm.py |
| Mistral | Mistral, Mixtral | mistral.py |
| Gemma | Gemma2, Gemma | gemma.py |
| Yi | Yi-1.5, Yi | yi.py |

### 7.2 VLM支持 (300+)

**主流多模态模型:**
| 系列 | 代表模型 | 模板文件 |
|------|----------|----------|
| Qwen-VL | Qwen3-VL, Qwen2.5-VL | qwen.py |
| InternVL | InternVL3.5, InternVL2.5 | internvl.py |
| Llava | Llava-1.5, Llava-NeXT | llava.py |
| GLM4v | GLM4.5v, GLM4v | glm.py |
| Ovis | Ovis2.5, Ovis1.6 | llava.py |
| Phi | Phi4, Phi3.5 | microsoft.py |

---

## 八、核心类与函数索引

### 8.1 核心类

| 类名 | 文件路径 | 说明 |
|------|----------|------|
| Swift | swift/tuners/base.py | 微调主类 |
| Tuner | swift/tuner_plugin/ | 微调插件基类 |
| Template | swift/template/base.py | 模板基类 |
| ModelMeta | swift/model/model_meta.py | 模型元数据 |
| Seq2SeqTrainer | swift/trainers/seq2seq_trainer.py | SFT训练器 |
| DPOTrainer | swift/rlhf_trainers/dpo_trainer.py | DPO训练器 |
| PPOTrainer | swift/rlhf_trainers/ppo_trainer.py | PPO训练器 |
| GRPOTrainer | swift/rlhf_trainers/grpo_trainer.py | GRPO训练器 |
| InferEngine | swift/infer_engine/ | 推理引擎基类 |
| TransformersEngine | swift/infer_engine/ | Transformers引擎 |
| VllmEngine | swift/infer_engine/ | vLLM引擎 |

### 8.2 核心函数

| 函数名 | 文件路径 | 说明 |
|--------|----------|------|
| sft_main | swift/pipelines/train/sft.py | SFT训练入口 |
| infer_main | swift/pipelines/infer/ | 推理入口 |
| rlhf_main | swift/pipelines/train/rlhf.py | RLHF训练入口 |
| get_template | swift/template/ | 获取模板 |
| load_dataset | swift/dataset/ | 加载数据集 |
| get_model_processor | swift/model/ | 获取模型处理器 |
| merge_lora | swift/pipelines/ | 合并LoRA权重 |

---

## 九、总结

ms-swift框架采用**分层架构设计**，具有以下特点:

1. **清晰的模块划分**: CLI → Pipeline → Trainer → Tuner → Model
2. **强大的插件机制**: 通过注册机制支持600+模型和多种微调方法
3. **统一的模板系统**: 支持不同模型的对话格式
4. **多引擎支持**: 支持多种训练和推理引擎
5. **完善的RLHF支持**: 支持DPO、PPO、GRPO等多种RLHF算法
6. **良好的扩展性**: 易于添加新模型、新微调方法和新功能

该框架的设计哲学是**简单、灵活、可扩展**，通过模块化和插件化架构，实现了对大模型训练和推理的全面支持。
