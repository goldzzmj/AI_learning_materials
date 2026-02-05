# ms-swift框架分布式训练与多集群训练深度技术指南

## 目录
1. [分布式训练技术概览](#1-分布式训练技术概览)
2. [DDP分布式数据并行](#2-ddp分布式数据并行)
3. [DeepSpeed ZeRO集成](#3-deepspeed-zero集成)
4. [FSDP/FSDP2支持](#4-fsdpfsdp2支持)
5. [Megatron并行技术](#5-megatron并行技术)
6. [device_map简单模型并行](#6-device_map简单模型并行)
7. [多机多卡训练配置](#7-多机多卡训练配置)
8. [多集群训练架构](#8-多集群训练架构)
9. [配置示例大全](#9-配置示例大全)
10. [性能优化建议](#10-性能优化建议)
11. [常见问题与解决方案](#11-常见问题与解决方案)

---

## 1. 分布式训练技术概览

ms-swift框架支持多种分布式训练技术，形成完整的分布式训练解决方案：

```
┌─────────────────────────────────────────────────────────────────┐
│                    ms-swift 分布式训练架构                        │
├─────────────────────────────────────────────────────────────────┤
│  用户接口层 (CLI/UI)                                             │
│     swift sft / swift train / swift rlhf                        │
├─────────────────────────────────────────────────────────────────┤
│  任务调度与配置解析                                               │
│     YAML配置 / 命令行参数 / 环境变量                               │
├─────────────────────────────────────────────────────────────────┤
│  分布式训练引擎                                                   │
│  ├─ DDP (DistributedDataParallel)                               │
│  ├─ DeepSpeed (ZeRO-2/ZeRO-3/Offload)                           │
│  ├─ FSDP/FSDP2 (Fully Sharded Data Parallel)                    │
│  ├─ Megatron-LM (TP/PP/SP/CP/EP/VPP)                            │
│  └─ device_map (简单模型并行)                                    │
├─────────────────────────────────────────────────────────────────┤
│  模型管理层                                                       │
│  ├─ 权重自动下载                                                 │
│  ├─ LoRA/QLoRA/DoRA 微调                                         │
│  ├─ 量化模型训练 (BNB/AWQ/GPTQ)                                  │
│  └─ 显存优化 (FlashAttention/GaLore/Liger-Kernel)               │
├─────────────────────────────────────────────────────────────────┤
│  推理与部署服务                                                   │
│  └─ vLLM / SGLang / LMDeploy / OpenAI API                       │
└─────────────────────────────────────────────────────────────────┘
```

### 技术选型对比

| 技术方案 | 适用场景 | 显存节省 | 通信开销 | 配置复杂度 |
|---------|---------|---------|---------|-----------|
| DDP | 单机多卡、中小模型 | ★ | 低 | 简单 |
| DeepSpeed ZeRO-2 | 显存紧张、中等规模 | ★★ | 中 | 中等 |
| DeepSpeed ZeRO-3 | 超大模型、CPU Offload | ★★★ | 高 | 中等 |
| FSDP | PyTorch原生、LoRA友好 | ★★ | 中 | 简单 |
| Megatron TP+PP | 千亿级模型、MoE | ★★★ | 很高 | 复杂 |
| device_map | 推理、快速验证 | ★ | 低 | 极简 |

---

## 2. DDP分布式数据并行

### 2.1 工作原理

DDP (DistributedDataParallel) 是PyTorch原生的数据并行方案：

```
┌──────────────────────────────────────────────────────────────┐
│                    DDP 数据并行架构                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │ GPU 0    │    │ GPU 1    │    │ GPU N    │              │
│   │ ┌──────┐ │    │ ┌──────┐ │    │ ┌──────┐ │              │
│   │ │Model │ │    │ │Model │ │    │ │Model │ │              │
│   │ │Copy 0│ │    │ │Copy 1│ │    │ │Copy N│ │              │
│   │ └──────┘ │    │ └──────┘ │    │ └──────┘ │              │
│   │    ↓     │    │    ↓     │    │    ↓     │              │
│   │  Grad 0  │    │  Grad 1  │    │  Grad N  │              │
│   │    ↓     │    │    ↓     │    │    ↓     │              │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│        │               │               │                     │
│        └───────────────┼───────────────┘                     │
│                        ↓                                      │
│                ┌───────────────┐                              │
│                │  AllReduce    │ ← 梯度同步                    │
│                │  (Ring/Tree)  │                              │
│                └───────────────┘                              │
│                        ↓                                      │
│        ┌───────────────┼───────────────┐                     │
│        ↓               ↓               ↓                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │Updated   │    │Updated   │    │Updated   │              │
│   │Model 0   │    │Model 1   │    │Model N   │              │
│   └──────────┘    └──────────┘    └──────────┘              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 单机多卡DDP训练

**环境变量方式启动：**

```bash
# 单机双卡LoRA微调Qwen2.5-7B
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_ddp \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

**关键参数说明：**

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `NPROC_PER_NODE` | 每个节点启动的进程数（GPU数） | 2, 4, 8 |
| `CUDA_VISIBLE_DEVICES` | 指定使用的GPU编号 | 0,1,2,3 |
| `per_device_train_batch_size` | 每卡batch size | 1-4 |
| `gradient_accumulation_steps` | 梯度累积步数 | 8-32 |

**总batch size计算公式：**
```
total_batch_size = per_device_train_batch_size × NPROC_PER_NODE × gradient_accumulation_steps
                 = 1 × 2 × 8 = 16
```

### 2.3 torchrun方式启动

```bash
torchrun --nproc_per_node=4 --master_port=29500 \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

---

## 3. DeepSpeed ZeRO集成

### 3.1 ZeRO技术原理

ZeRO (Zero Redundancy Optimizer) 通过分片存储来降低显存占用：

```
┌─────────────────────────────────────────────────────────────────┐
│                  ZeRO 三阶段分片策略                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: 优化器状态分片 (Optimizer State Sharding)              │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ GPU0: [Param0, Grad0, OptState0_0]                  │        │
│  │ GPU1: [Param1, Grad1, OptState0_1]  ← 仅优化器状态分片 │        │
│  │ GPU2: [Param2, Grad2, OptState0_2]                  │        │
│  │ GPU3: [Param3, Grad3, OptState0_3]                  │        │
│  └─────────────────────────────────────────────────────┘        │
│  显存节省: 4x (4卡环境)                                          │
│                                                                  │
│  Stage 2: + 梯度分片 (Gradient Sharding)                         │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ GPU0: [Param0, Grad0_0, OptState0_0]                │        │
│  │ GPU1: [Param1, Grad0_1, OptState0_1]  ← 梯度也分片    │        │
│  │ GPU2: [Param2, Grad0_2, OptState0_2]                │        │
│  │ GPU3: [Param3, Grad0_3, OptState0_3]                │        │
│  └─────────────────────────────────────────────────────┘        │
│  显存节省: 8x                                                    │
│                                                                  │
│  Stage 3: + 参数分片 (Parameter Sharding)                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ GPU0: [Param0_0, Grad0_0, OptState0_0]              │        │
│  │ GPU1: [Param0_1, Grad0_1, OptState0_1]  ← 参数也分片  │        │
│  │ GPU2: [Param0_2, Grad0_2, OptState0_2]              │        │
│  │ GPU3: [Param0_3, Grad0_3, OptState0_3]              │        │
│  └─────────────────────────────────────────────────────┘        │
│  显存节省: 与数据并行度线性相关 (N卡节省Nx)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 内置DeepSpeed配置

ms-swift提供内置的DeepSpeed配置，无需手动编写JSON文件：

```bash
# 使用内置ZeRO-2配置
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --deepspeed zero2 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'

# 使用内置ZeRO-3配置
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --deepspeed zero3 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'

# 使用内置ZeRO-2 Offload配置
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --deepspeed zero2_offload \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'

# 使用内置ZeRO-3 Offload配置
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --deepspeed zero3_offload \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'
```

**内置配置对应表：**

| 配置名称 | ZeRO Stage | CPU Offload | 适用场景 |
|---------|-----------|-------------|---------|
| zero0 | 0 | 否 | 基线对比 |
| zero1 | 1 | 否 | 优化器状态分片 |
| zero2 | 2 | 否 | 梯度+优化器分片 |
| zero3 | 3 | 否 | 全分片 |
| zero2_offload | 2 | 是 | 显存极度紧张 |
| zero3_offload | 3 | 是 | 超大模型训练 |

### 3.3 自定义DeepSpeed配置

**ZeRO-2配置文件 (ds_config_zero2.json):**

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

**ZeRO-3配置文件 (ds_config_zero3.json):**

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "stage3_gather_16bit_weights_on_model_save": true,
        "stage3_max_live_parameters": 5e8,
        "stage3_max_reuse_distance": 5e8,
        "stage3_prefetch_bucket_size": 5e8
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

**启动命令：**

```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --deepspeed ds_config_zero3.json \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --gradient_accumulation_steps 8
```

### 3.4 DeepSpeed多机训练

```bash
# Node 0 (主节点)
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR="192.168.1.10" \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --deepspeed ds_config_zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16

# Node 1 (工作节点)
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR="192.168.1.10" \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --deepspeed ds_config_zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
```

---

## 4. FSDP/FSDP2支持

### 4.1 FSDP工作原理

FSDP (Fully Sharded Data Parallel) 是PyTorch原生的全分片数据并行方案：

```
┌─────────────────────────────────────────────────────────────────┐
│                    FSDP 工作原理                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 模型包装 (Model Wrapping)                                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  model = nn.Transformer(...)                        │        │
│  │       ↓                                             │        │
│  │  fsdp_model = FSDP(                                 │        │
│  │      model,                                         │        │
│  │      auto_wrap_policy=transformer_layer_policy,     │        │
│  │      cpu_offload=CPUOffload(offload_params=True),   │        │
│  │      mixed_precision=MixedPrecision(...),           │        │
│  │      use_orig_params=True  # FSDP2关键参数           │        │
│  │  )                                                  │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  2. 前向传播 (Forward)                                           │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Input → AllGather参数 → 计算 → 释放参数             │        │
│  │       ↑___________________________↓                 │        │
│  │  按需动态拉取参数，计算后立即释放                      │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  3. 反向传播 (Backward)                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Grad计算 → ReduceScatter → 各卡保留部分梯度         │        │
│  │        ↑________________________↓                   │        │
│  │  梯度分片存储，避免全量复制                           │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 FSDP配置与启动

**方式1: 命令行参数启动**

```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --deepspeed zero2 \
    --fsdp auto_wrap \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

**方式2: 使用--use_fsdp参数**

```bash
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --use_fsdp \
    --fsdp "full_shard auto_wrap" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1
```

**FSDP关键参数说明：**

| 参数 | 说明 | 可选值 |
|-----|------|-------|
| `fsdp` | FSDP分片策略 | full_shard, shard_grad_op, no_shard |
| `fsdp_config` | FSDP配置文件路径 | path/to/config.json |
| `fsdp_transformer_layer_cls_to_wrap` | 要包装的Transformer层类名 | 'Qwen2DecoderLayer', 'LlamaDecoderLayer' |

### 4.3 FSDP2 (PyTorch 2.1+)

FSDP2引入`use_orig_params=True`参数，解决了LoRA微调的兼容性问题：

```python
# FSDP2配置示例
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.api import MixedPrecision

fsdp_model = FSDP(
    model,
    auto_wrap_policy=transformer_layer_policy,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    ),
    use_orig_params=True,  # 关键！允许LoRA绑定原始权重
)
```

**FSDP vs DeepSpeed对比：**

| 特性 | FSDP | DeepSpeed ZeRO-3 |
|-----|------|-----------------|
| 生态集成 | PyTorch原生 | 第三方库 |
| LoRA兼容性 | 优秀 (FSDP2) | 需要特殊处理 |
| 启动速度 | 快 | 较慢 |
| 调试友好度 | 高 | 中等 |
| 超大规模支持 | 有限 | ZeRO-Infinity |
| 配置复杂度 | 低 | 中等 |

---

## 5. Megatron并行技术

### 5.1 Megatron并行架构

ms-swift深度集成Megatron-LM，支持多种并行策略：

```
┌─────────────────────────────────────────────────────────────────┐
│              Megatron 混合并行架构 (3D并行)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    数据并行 (DP)                         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │ Data 0  │  │ Data 1  │  │ Data 2  │  │ Data 3  │    │    │
│  │  │[TP0-PP0]│  │[TP0-PP0]│  │[TP0-PP0]│  │[TP0-PP0]│    │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    流水线并行 (PP)                       │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │ Stage 0 │→ │ Stage 1 │→ │ Stage 2 │→ │ Stage 3 │    │    │
│  │  │ Layers  │  │ Layers  │  │ Layers  │  │ Layers  │    │    │
│  │  │ 0-11    │  │ 12-23   │  │ 24-35   │  │ 36-47   │    │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    张量并行 (TP)                         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │ TP Rank │  │ TP Rank │  │ TP Rank │  │ TP Rank │    │    │
│  │  │    0    │  │    1    │  │    2    │  │    3    │    │    │
│  │  │[W_0:W/4]│  │[W_1:W/4]│  │[W_2:W/4]│  │[W_3:W/4]│    │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │    │
│  │       ↑            ↑            ↑            ↑          │    │
│  │       └────────────┴────────────┴────────────┘          │    │
│  │                    AllReduce                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 张量并行 (TP)

**工作原理：**
- 在层内对权重矩阵进行切分
- FFN层: 按列切分 → AllReduce合并
- Attention层: QKV投影切分 → AllReduce合并

**配置示例：**

```python
from swift import SwiftConfig, Trainer

config = SwiftConfig(
    model_type="qwen3",
    parallel={
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1
    },
    dtype="bf16"
)

trainer = Trainer(model="Qwen/Qwen3-7B", config=config)
trainer.train(dataset="my_sft_data")
```

### 5.3 流水线并行 (PP)

**工作原理：**
- 按层纵向切分模型
- 每个stage负责连续的若干层
- micro-batch流水执行减少气泡

**配置示例：**

```python
config = SwiftConfig(
    model_type="llama4",
    parallel={
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 8,
        "virtual_pipeline_size": 4  # VPP
    }
)
```

### 5.4 上下文并行 (CP)

用于超长序列训练，降低注意力计算的显存消耗：

```python
config = SwiftConfig(
    model_type="qwen3",
    parallel={
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 2,
        "context_parallel_size": 4,  # CP
        "sequence_parallel": True     # SP
    }
)
```

### 5.5 专家并行 (EP) - MoE专用

```python
config = SwiftConfig(
    model_type="deepseek-moe-16b",
    parallel={
        "tensor_parallel_size": 4,
        "expert_parallel_size": 8,
        "moe_router_load_balancing": True
    },
    training_task="GRPO"
)
```

### 5.6 Megatron混合并行配置

**Qwen3-70B在64卡A100上的配置：**

```yaml
# config.yaml
model: Qwen/Qwen3-70B
parallel:
  tp: 4      # 张量并行度
  pp: 8      # 流水线并行度
  dp: 2      # 数据并行度
  cp: 1      # 上下文并行度
  # 总GPU数 = tp × pp × dp = 4 × 8 × 2 = 64

train_type: dpo
dataset: dpo_data.jsonl
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1e-5
```

**启动命令：**

```bash
# 使用Megatron-Swift启动
python -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes 8 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    swift/cli/_megatron/rlhf.py \
    --config config.yaml
```

---

## 6. device_map简单模型并行

### 6.1 工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│              device_map 模型分片架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    模型层分布                             │    │
│  │                                                          │    │
│  │  Embedding Layer        →  cuda:0    (GPU 24GB)         │    │
│  │  Transformer Layers 0-5 →  cuda:0                       │    │
│  │  Transformer Layers 6-11→  cuda:1    (GPU 24GB)         │    │
│  │  Transformer Layers 12-17→ cuda:2    (GPU 24GB)         │    │
│  │  Transformer Layers 18-23→ cuda:3    (GPU 24GB)         │    │
│  │  LM Head                →  cpu       (CPU Memory)       │    │
│  │                                                          │    │
│  │  总显存: 96GB GPU + CPU Offload                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  特点:                                                          │
│  - 无需修改代码，一行参数启用                                    │
│  - 支持GPU/CPU/NPU异构部署                                       │
│  - 自动处理跨设备数据传输                                        │
│  - 适合推理和轻量微调                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 使用方式

**方式1: 自动分配**

```bash
swift infer \
    --model_type qwen \
    --model_id Qwen/Qwen-70B \
    --device_map auto \
    --offload_folder ./offload_cache
```

**方式2: 手动指定device_map**

```python
from transformers import AutoModelForCausalLM

# 自定义device_map
custom_device_map = {
    "transformer.word_embeddings": "cuda:0",
    "transformer.layers.0": "cuda:0",
    "transformer.layers.1": "cuda:0",
    "transformer.layers.2": "cuda:1",
    "transformer.layers.3": "cuda:1",
    # ... 更多层
    "transformer.ln_f": "cuda:1",
    "lm_head": "cpu"  # 卸载到CPU
}

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-14B",
    device_map=custom_device_map,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
```

### 6.3 结合量化使用

```bash
swift infer \
    --model_id Qwen/Qwen-70B \
    --quantization_bit 4 \
    --device_map auto
```

**显存占用对比：**

| 模型 | 精度 | device_map | 显存需求 |
|-----|------|-----------|---------|
| Qwen-7B | FP16 | 否 | ~14GB |
| Qwen-7B | FP16 | auto | ~7GB (2卡) |
| Qwen-14B | FP16 | 否 | ~28GB |
| Qwen-14B | FP16 | auto | ~14GB (2卡) |
| Qwen-14B | 4-bit | auto | ~7GB (2卡) |

---

## 7. 多机多卡训练配置

### 7.1 启动方式对比

| 启动方式 | 适用场景 | 命令示例 | 特点 |
|---------|---------|---------|------|
| swift | 简单快速 | `swift sft --nnodes 2` | 一键启动，自动处理 |
| torchrun | 标准方式 | `torchrun --nnodes 2` | PyTorch原生，灵活可控 |
| deepspeed | DeepSpeed训练 | `deepspeed --num_gpus 8` | 集成DeepSpeed特性 |
| accelerate | HuggingFace生态 | `accelerate launch` | 与HF工具链集成 |
| DLC | 阿里云深度学习 | `dlc submit` | 云上训练专用 |

### 7.2 环境变量配置

**核心环境变量：**

```bash
# 节点配置
export NNODES=2                    # 总节点数
export NODE_RANK=0                 # 当前节点rank (0为主节点)
export MASTER_ADDR=192.168.1.10    # 主节点IP
export MASTER_PORT=29500           # 通信端口
export NPROC_PER_NODE=8            # 每节点GPU数

# NCCL配置
export NCCL_DEBUG=INFO             # NCCL调试信息
export NCCL_SOCKET_IFNAME=eth0     # 指定网络接口
export NCCL_IB_DISABLE=1           # 禁用InfiniBand
export NCCL_P2P_DISABLE=1          # 禁用P2P通信
export NCCL_TREE_THRESHOLD=0       # 强制使用Tree算法

# PyTorch分布式配置
export PYTHONUNBUFFERED=1          # 无缓冲输出
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 分布式调试
```

### 7.3 单机多卡启动

**方式1: 环境变量方式**

```bash
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'
```

**方式2: torchrun方式**

```bash
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'
```

### 7.4 多机多卡启动

**主节点 (Node 0):**

```bash
# 方式1: swift命令
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=192.168.1.10 \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --deepspeed ds_config_zero3.json \
    --output_dir /shared/output_multinode \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
```

**工作节点 (Node 1):**

```bash
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=192.168.1.10 \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --deepspeed ds_config_zero3.json \
    --output_dir /shared/output_multinode \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
```

**方式2: torchrun多机启动**

```bash
# Node 0
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.10 \
    --master_port=29500 \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --deepspeed ds_config_zero3.json

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.10 \
    --master_port=29500 \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --deepspeed ds_config_zero3.json
```

### 7.5 Docker多机训练

```bash
# 主节点
docker run --rm -it --gpus all -p 29500:29500 \
    --shm-size=50G \
    -v /path/to/ms-swift:/app \
    -v /path/to/models:/models \
    -v /path/to/datasets:/datasets \
    -e NNODES=2 \
    -e NODE_RANK=0 \
    -e MASTER_ADDR=主节点宿主机IP \
    -e MASTER_PORT=29500 \
    -e NPROC_PER_NODE=8 \
    modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.1 \
    bash -c "cd /app && swift sft --model /models/Qwen-72B --train_type lora ..."

# 工作节点
docker run --rm -it --gpus all \
    --shm-size=50G \
    -v /path/to/ms-swift:/app \
    -v /path/to/models:/models \
    -v /path/to/datasets:/datasets \
    -e NNODES=2 \
    -e NODE_RANK=1 \
    -e MASTER_ADDR=主节点宿主机IP \
    -e MASTER_PORT=29500 \
    -e NPROC_PER_NODE=8 \
    modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.0-modelscope1.28.2-swift3.7.1 \
    bash -c "cd /app && swift sft --model /models/Qwen-72B --train_type lora ..."
```

---

## 8. 多集群训练架构

### 8.1 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    多集群训练架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    集群调度层                             │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                  │    │
│  │  │ Cluster │  │ Cluster │  │ Cluster │                  │    │
│  │  │   A     │  │   B     │  │   C     │                  │    │
│  │  │ (K8s)   │  │ (Slurm) │  │ (DLC)   │                  │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘                  │    │
│  │       └─────────────┴─────────────┘                      │    │
│  │              统一调度接口 (Swift Scheduler)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    通信层                                 │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              高速互联网络                         │    │    │
│  │  │    InfiniBand (100-400Gbps) / RoCE / NVLink    │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    存储层                                 │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │   NFS/      │  │  对象存储    │  │  分布式     │      │    │
│  │  │   Lustre    │  │  (OSS/S3)   │  │  文件系统   │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Kubernetes多集群训练

**Master Pod配置：**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-master
  labels:
    role: master
spec:
  restartPolicy: OnFailure
  containers:
  - name: training-container
    image: nvidia/cuda:11.8.0-runtime-ubuntu22.04
    command: ["/bin/bash", "/data/train.sh"]
    env:
    - name: MASTER_ADDR
      value: "training-master"
    - name: MASTER_PORT
      value: "29500"
    - name: NODE_RANK
      value: "0"
    - name: NPROC_PER_NODE
      value: "4"
    - name: NNODES
      value: "3"
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1,2,3"
    resources:
      requests:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4
      limits:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4
    volumeMounts:
    - name: local-data
      mountPath: /data
  volumes:
  - name: local-data
    hostPath:
      path: /path/to/your/local/data
      type: Directory
```

**Worker Pod配置：**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-worker-1
  labels:
    role: worker
spec:
  restartPolicy: OnFailure
  containers:
  - name: training-container
    image: nvidia/cuda:11.8.0-runtime-ubuntu22.04
    command: ["/bin/bash", "/data/train.sh"]
    env:
    - name: MASTER_ADDR
      value: "training-master"
    - name: MASTER_PORT
      value: "29500"
    - name: NODE_RANK
      value: "1"
    - name: NPROC_PER_NODE
      value: "4"
    - name: NNODES
      value: "3"
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1,2,3"
    resources:
      requests:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4
      limits:
        cpu: "16"
        memory: 100G
        nvidia.com/gpu: 4
```

**启动命令：**

```bash
# 启动任务
kubectl apply -f training-master.yaml
kubectl apply -f training-worker-1.yaml
kubectl apply -f training-worker-2.yaml

# 查看状态
kubectl get pods -l role=master,role=worker

# 删除任务
kubectl delete -f training-master.yaml
kubectl delete -f training-worker-1.yaml
kubectl delete -f training-worker-2.yaml
```

### 8.3 昇腾NPU多集群训练

```bash
# 昇腾910B 32节点分布式训练
source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
fi

if [ -z "${RANK}" ]; then
    NNODES=1
    NODE_RANK=0
else
    NODE_RANK=${RANK}
    NNODES=${PET_NNODES}
fi

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=32 \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
NPROC_PER_NODE=8 \
swift sft \
    --model_type internvl2-8b \
    --model_id_or_path /datas/datasets/InternVL2-8B \
    --sft_type lora \
    --use_rslora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_train_epochs 6 \
    --dtype bf16 \
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --deepspeed default-zero2
```

---

## 9. 配置示例大全

### 9.1 单机单卡配置

```bash
# 7B模型LoRA微调 - 单卡24GB显存
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output \
    --gradient_checkpointing true
```

### 9.2 单机多卡DDP配置

```bash
# 7B模型LoRA微调 - 单机4卡
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --output_dir output_ddp \
    --dataloader_num_workers 4
```

### 9.3 单机多卡DeepSpeed ZeRO-2配置

```bash
# 14B模型LoRA微调 - 单机4卡
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-14B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --output_dir output_zero2
```

### 9.4 单机多卡DeepSpeed ZeRO-3配置

```bash
# 32B模型LoRA微调 - 单机8卡A100
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen/Qwen2.5-32B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --gradient_accumulation_steps 2 \
    --max_length 2048 \
    --output_dir output_zero3
```

### 9.5 多机多卡DeepSpeed配置

```bash
# 72B模型LoRA微调 - 2机16卡
# Node 0
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=192.168.1.10 \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --output_dir /shared/output_72b

# Node 1
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=192.168.1.10 \
MASTER_PORT=29500 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-72B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --output_dir /shared/output_72b
```

### 9.6 FSDP配置

```bash
# 7B模型全参数微调 - 单机4卡
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --use_fsdp \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --output_dir output_fsdp
```

### 9.7 Megatron TP+PP配置

```bash
# 70B模型Megatron训练 - 32卡
python -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes 4 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    swift/cli/_megatron/sft.py \
    --model Qwen/Qwen3-70B \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 8 \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --output_dir output_megatron
```

### 9.8 QLoRA + DeepSpeed配置

```bash
# 7B模型QLoRA微调 - 单卡9GB显存
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --quantization_bit 4 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dtype bf16 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output_qlora
```

---

## 10. 性能优化建议

### 10.1 显存优化策略

| 优化技术 | 显存节省 | 速度影响 | 适用场景 |
|---------|---------|---------|---------|
| gradient_checkpointing | ~30% | -20% | 长序列训练 |
| LoRA/QLoRA | ~70% | -5% | 大多数微调任务 |
| DeepSpeed ZeRO-2 | ~50% | -10% | 中等规模模型 |
| DeepSpeed ZeRO-3 | ~80% | -20% | 超大模型 |
| FSDP | ~60% | -15% | PyTorch原生 |
| 量化训练(4-bit) | ~75% | -10% | 显存极度紧张 |
| FlashAttention | ~20% | +30% | 注意力计算 |
| sequence_packing | ~40% | +50% | 变长序列 |

### 10.2 通信优化策略

```bash
# 1. 使用高速网络
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand
export NCCL_IB_DISABLE=0        # 启用IB

# 2. 优化NCCL算法
export NCCL_ALGO=RING          # Ring算法
export NCCL_TREE_THRESHOLD=0   # 强制Tree

# 3. 增大通信缓冲区
export NCCL_BUFFSIZE=2097152   # 2MB

# 4. 启用GPU Direct RDMA
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
```

### 10.3 计算优化策略

```bash
# 1. 使用FlashAttention
--attn_impl flash_attn2

# 2. 启用Liger-Kernel
--use_liger true

# 3. 混合精度训练
--torch_dtype bfloat16  # 或 float16

# 4. 优化数据加载
--dataloader_num_workers 8 \
--dataloader_pin_memory true

# 5. 编译优化 (PyTorch 2.0+)
--torch_compile true
```

### 10.4 最佳实践总结

**小规模训练 (< 13B):**
- 使用DDP或FSDP
- LoRA微调优先
- 单机多卡即可

**中等规模训练 (13B - 70B):**
- 使用DeepSpeed ZeRO-2/3
- TP+DP混合并行
- 多机多卡配置

**大规模训练 (> 70B):**
- 使用Megatron TP+PP+DP
- 启用CP处理长序列
- MoE模型使用EP
- 高速网络(InfiniBand)

---

## 11. 常见问题与解决方案

### 11.1 通信问题

**问题1: NCCL通信超时/卡住**

```
[ERROR] NCCL operation failed: unhandled system error
```

**解决方案：**

```bash
# 1. 检查网络连通性
nc -vz <MASTER_ADDR> <MASTER_PORT>

# 2. 禁用IB测试
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 3. 启用调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 4. 指定网络接口
export NCCL_SOCKET_IFNAME=eth0
```

**问题2: 多机训练卡住无输出**

**解决方案：**

```bash
# 1. 确保所有节点使用相同配置
# 2. 检查防火墙设置
# 3. 使用共享存储
# 4. 同步系统时间

# 5. 设置超时参数
export NCCL_TIMEOUT=1800  # 30分钟
```

### 11.2 显存问题

**问题1: CUDA Out of Memory**

**解决方案：**

```bash
# 1. 减小batch size
--per_device_train_batch_size 1

# 2. 启用gradient checkpointing
--gradient_checkpointing true

# 3. 使用DeepSpeed ZeRO-3
--deepspeed zero3_offload

# 4. 减小序列长度
--max_length 1024

# 5. 使用量化
--quantization_bit 4
```

**问题2: ZeRO-3显存占用不均匀**

**解决方案：**

```json
// ds_config.json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 5e8,
    "stage3_max_reuse_distance": 5e8,
    "stage3_prefetch_bucket_size": 5e8
  }
}
```

### 11.3 训练稳定性问题

**问题1: Loss NaN/Inf**

**解决方案：**

```bash
# 1. 降低学习率
--learning_rate 5e-5

# 2. 启用梯度裁剪
--max_grad_norm 1.0

# 3. 使用float16而非bfloat16
--torch_dtype float16

# 4. 减小batch size
--per_device_train_batch_size 1
```

**问题2: 多卡收敛不一致**

**解决方案：**

```bash
# 1. 设置随机种子
--seed 42

# 2. 禁用deterministic算法
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 3. 使用相同的数据顺序
--dataloader_drop_last true
```

### 11.4 兼容性问题

**问题1: LoRA与ZeRO-3不兼容**

**解决方案：**

```bash
# 方案1: 使用FSDP2
--use_fsdp \
--fsdp "full_shard auto_wrap" \
--fsdp_use_orig_params true

# 方案2: 降级到ZeRO-2
--deepspeed zero2

# 方案3: 使用DeepSpeed的特殊配置
# 在ds_config.json中添加
{
  "zero_optimization": {
    "stage": 3,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

**问题2: 多模态模型OOM**

**解决方案：**

```bash
# 1. 冻结ViT
--freeze_vit true

# 2. 单独设置ViT的gradient checkpointing
--vit_gradient_checkpointing false

# 3. 使用DeepSpeed
--deepspeed zero3_offload
```

### 11.5 性能问题

**问题1: 训练速度过慢**

**解决方案：**

```bash
# 1. 启用FlashAttention
--attn_impl flash_attn2

# 2. 优化数据加载
--dataloader_num_workers 8 \
--dataloader_pin_memory true

# 3. 使用Liger-Kernel
--use_liger true

# 4. 检查通信瓶颈
export NCCL_DEBUG=INFO

# 5. 使用sequence packing
--packing true
```

**问题2: GPU利用率低**

**解决方案：**

```bash
# 1. 增大batch size
--per_device_train_batch_size 2

# 2. 减小梯度累积
--gradient_accumulation_steps 4

# 3. 优化数据预处理
--dataset_num_proc 8

# 4. 使用更快的存储
# 将数据放在SSD/NVMe上
```

### 11.6 检查点问题

**问题: 检查点保存/加载失败**

**解决方案：**

```bash
# 1. 使用共享存储
--output_dir /shared/checkpoints

# 2. 只在主节点保存
--save_on_each_node false

# 3. 限制检查点数量
--save_total_limit 3

# 4. 恢复训练
--resume_from_checkpoint /path/to/checkpoint
```

---

## 附录: 快速参考卡

### 环境变量速查表

| 变量名 | 说明 | 示例值 |
|-------|------|-------|
| `NNODES` | 总节点数 | 2, 4, 8 |
| `NODE_RANK` | 当前节点rank | 0, 1, 2 |
| `MASTER_ADDR` | 主节点IP | 192.168.1.10 |
| `MASTER_PORT` | 通信端口 | 29500 |
| `NPROC_PER_NODE` | 每节点GPU数 | 4, 8 |
| `CUDA_VISIBLE_DEVICES` | 可见GPU | 0,1,2,3 |
| `NCCL_DEBUG` | NCCL调试 | INFO |
| `NCCL_IB_DISABLE` | 禁用IB | 1 |

### 启动命令速查表

| 场景 | 命令 |
|-----|------|
| 单机单卡 | `swift sft --model ...` |
| 单机多卡 | `NPROC_PER_NODE=4 swift sft ...` |
| 多机多卡 | `NNODES=2 NODE_RANK=0 MASTER_ADDR=... swift sft ...` |
| DeepSpeed | `swift sft --deepspeed zero3 ...` |
| FSDP | `swift sft --use_fsdp --fsdp "full_shard auto_wrap" ...` |
| Megatron | `torchrun --nnodes=4 swift/cli/_megatron/sft.py ...` |

### 显存估算公式

```
显存占用 ≈ 模型参数 + 优化器状态 + 梯度 + 激活值
          = P × (4 + 8 + 4) + Activation
          = 16P + Activation (FP32)
          = 2P + 4P + 2P + Activation (FP16)

使用ZeRO-3后:
显存占用 ≈ (16P / N) + Activation

其中:
- P: 参数量 (7B = 7×10^9)
- N: GPU数量
```
