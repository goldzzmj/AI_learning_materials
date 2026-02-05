# ms-swift框架VLM训练方法深度技术分析报告

## 目录
1. [框架概述](#1-框架概述)
2. [VLM模型支持](#2-vlm模型支持)
3. [预训练CPT实现](#3-预训练cpt实现)
4. [监督微调SFT](#4-监督微调sft)
5. [强化学习方法](#5-强化学习方法)
6. [多模态数据处理](#6-多模态数据处理)
7. [不同VLM架构支持](#7-不同vlm架构支持)
8. [训练配置示例](#8-训练配置示例)
9. [最佳实践](#9-最佳实践)

---

## 1. 框架概述

### 1.1 ms-swift简介

**ms-swift** 是魔搭社区(ModelScope)提供的大模型与多模态大模型微调部署框架，是目前最全面的VLM训练框架之一。

**核心能力矩阵：**

| 能力维度 | 支持情况 |
|---------|---------|
| 纯文本大模型 | 600+ |
| 多模态大模型(MLLM) | 300+ |
| 训练任务类型 | CPT/SFT/GRPO/DPO/KTO/RM/PPO等 |
| 内置数据集 | 150+ |
| 轻量微调方法 | LoRA/QLoRA/DoRA/Adapter/GaLore等 |
| 分布式训练 | DDP/DeepSpeed/Megatron/FSDP |

### 1.2 VLM训练流程全景图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ms-swift VLM 训练流程全景                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [数据层]                                                                  │
│      ↓                                                                      │
│   [任务调度器] → [CPT模块] → [SFT模块] → [DPO/GRPO模块]                       │
│                      ↓           ↓             ↓                            │
│              [Megatron并行] [LoRA适配] [vLLM采样+Reward Model]               │
│                      ↓           ↓             ↓                            │
│               [优化器共享] ← [统一训练流] → [模型检查点]                        │
│                      ↓                                                      │
│                 [评测模块 EvalScope]                                         │
│                      ↓                                                      │
│              [量化导出 GPTQ/AWQ/FP8]                                         │
│                      ↓                                                      │
│         [部署引擎 vLLM/SGLang/LMDeploy]                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. VLM模型支持

### 2.1 支持的VLM模型列表

ms-swift支持300+多模态大模型，主要包括以下系列：

| 模型系列 | 模型类型 | 语言支持 | 参数规模 | 特点 |
|---------|---------|---------|---------|------|
| **Qwen-VL/Qwen2-VL/Qwen2.5-VL/Qwen3-VL** | 视觉语言模型 | 中英 | 2B-72B | 通义千问官方VLM |
| **Qwen-Omni/Qwen2.5-Omni/Qwen3-Omni** | 全模态模型 | 中英 | 7B-32B | 图像+视频+音频 |
| **LLaVA/LLaVA-Next/LLaVA-OneVision** | 视觉语言模型 | 英文 | 0.5B-110B | 社区主流VLM |
| **InternVL/InternVL2/InternVL3/InternVL3.5** | 视觉语言模型 | 中英 | 1B-40B | 商汤开源VLM |
| **MiniCPM-V/MiniCPM-V-2.5/MiniCPM-V-2.6** | 视觉语言模型 | 中英 | 3B-9B | 端侧友好 |
| **CogVLM/CogVLM2/CogAgent/GLM4V** | 视觉语言模型 | 中英 | 9B-19B | 智谱开源 |
| **DeepSeek-VL/DeepSeek-VL2** | 视觉语言模型 | 中英 | 1.3B-7B | DeepSeek系列 |
| **Ovis/Ovis2/Ovis2.5** | 视觉语言模型 | 中英 | 7B-16B | 结构化视觉理解 |
| **Phi3-Vision/Phi4** | 视觉语言模型 | 英文 | 4B | 微软开源 |
| **Yi-VL** | 视觉语言模型 | 中英 | 6B-34B | 零一万物 |

### 2.2 多模态任务支持

```
┌────────────────────────────────────────────────────────────┐
│                    VLM 任务类型支持                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  VQA        │  │  Captioning │  │  OCR        │        │
│  │  视觉问答    │  │  图像描述    │  │  文字识别    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Grounding  │  │  Video QA   │  │  Audio QA   │        │
│  │  目标定位    │  │  视频问答    │  │  音频问答    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Agent      │  │  Embedding  │  │  Reranker   │        │
│  │  智能体     │  │  向量表示    │  │  重排序     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 预训练(CPT)实现

### 3.1 CPT原理

**Continual Pre-training (CPT)** 是在已有预训练模型基础上，使用特定领域的大规模无标注数据继续训练，以注入领域知识。

**数学原理：**

```
L_CPT(θ) = -Σ(t=1 to T) log P_θ(x_t | x_<t)
```

其中：
- θ₀ 为预训练模型参数
- Dₜ 为领域特定数据
- 使用自回归语言建模目标

### 3.2 CPT训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     CPT 训练流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: 数据准备                                               │
│  ├── 领域语料收集 (医疗/金融/法律等)                              │
│  ├── 数据清洗与去重                                              │
│  └── 混合20-50%通用数据保持稳定性                                 │
│                          ↓                                      │
│  Stage 2: 模型初始化                                             │
│  ├── 加载预训练权重                                              │
│  ├── 设置较低学习率 (通常2e-5)                                   │
│  └── 配置warmup + linear decay                                  │
│                          ↓                                      │
│  Stage 3: 训练执行                                               │
│  ├── 使用生成式模板 (非对话模板)                                  │
│  ├── 支持LoRA/QLoRA全参数训练                                    │
│  └── Megatron并行加速                                            │
│                          ↓                                      │
│  Stage 4: 效果验证                                               │
│  ├── 领域知识retention测试                                       │
│  └── 通用能力保持评估                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 CPT代码示例

```bash
# 多模态CPT训练示例
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
swift pt \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset medical_image_corpus.jsonl \
    --streaming true \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --output_dir output/cpt_vlm
```

### 3.4 CPT关键参数说明

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `--streaming` | 流式数据加载 | true (大数据集) |
| `--train_type` | 训练类型 | full/lora/qlora |
| `--learning_rate` | 学习率 | 1e-5 ~ 5e-5 |
| `--warmup_ratio` | warmup比例 | 0.05 ~ 0.1 |
| `--freeze_vit` | 冻结视觉编码器 | false (CPT通常解冻) |
| `--freeze_llm` | 冻结语言模型 | false |

---

## 4. 监督微调(SFT)

### 4.1 SFT完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     SFT 训练架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    多模态数据输入                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │  Text   │  │  Image  │  │  Video  │  │  Audio  │    │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │   │
│  │       └─────────────┴─────────────┴─────────────┘       │   │
│  │                         ↓                               │   │
│  │              ┌─────────────────────┐                    │   │
│  │              │   Template Encode   │                    │   │
│  │              │   (多模态编码)        │                    │   │
│  │              └──────────┬──────────┘                    │   │
│  └─────────────────────────┼───────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    VLM 模型结构                          │   │
│  │                                                         │   │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐        │   │
│  │   │   ViT    │───→│ Aligner  │───→│   LLM    │        │   │
│  │   │(视觉编码)│    │(对齐层)  │    │(语言模型)│        │   │
│  │   └──────────┘    └──────────┘    └──────────┘        │   │
│  │        ↑               ↑               ↑               │   │
│  │   freeze_vit     freeze_aligner    freeze_llm         │   │
│  │   (可独立控制)                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    训练优化器                            │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │  LoRA   │  │ QLoRA   │  │ DoRA    │  │  Full   │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 SFT核心代码实现

```python
# ms-swift SFT核心流程伪代码
class SwiftSFTTrainer:
    def __init__(self, args):
        # 1. 加载模型和处理器
        self.model, self.processor = get_model_tokenizer(args.model)
        
        # 2. 获取模板
        self.template = get_template(self.processor)
        
        # 3. 配置训练参数
        self.training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            # ... 其他参数
        )
        
        # 4. 应用LoRA/QLoRA
        if args.tuner_type == 'lora':
            self.model = get_peft_model(self.model, LoraConfig(...))
    
    def train(self):
        # 数据编码流程
        def encode_function(examples):
            # Template编码：处理text+image+video+audio
            inputs = self.template.encode(examples)
            return inputs
        
        # 创建训练器并训练
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            data_collator=self.template.data_collator,
        )
        trainer.train()
```

### 4.3 SFT训练脚本示例

```bash
# Qwen2.5-VL SFT训练
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset custom_vqa_dataset.jsonl \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --max_length 8192 \
    --packing true \
    --warmup_ratio 0.05 \
    --output_dir output/sft_qwen2_5_vl
```

### 4.4 多模态数据格式

```json
// 标准多模态SFT数据格式
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<image>What is in the image?"},
        {"role": "assistant", "content": "The image shows a cat sitting on a chair."}
    ],
    "images": ["/path/to/image.jpg"],
    "videos": [],
    "audios": []
}

// 多图输入
{
    "messages": [
        {"role": "user", "content": "<image><image>What is the difference?"},
        {"role": "assistant", "content": "The first is a cat, the second is a dog."}
    ],
    "images": ["/path/to/cat.jpg", "/path/to/dog.jpg"]
}

// 视频+图像混合
{
    "messages": [
        {"role": "user", "content": "<image>What is in the image, <video>What is in the video?"},
        {"role": "assistant", "content": "Image: elephant; Video: puppy running."}
    ],
    "images": ["/path/to/image.jpg"],
    "videos": ["/path/to/video.mp4"]
}
```

---

## 5. 强化学习方法

### 5.1 强化学习算法矩阵

ms-swift支持丰富的GRPO家族算法和偏好学习方法：

| 算法 | 类型 | 多模态支持 | 特点 |
|-----|------|-----------|------|
| **GRPO** | 在线RL | ✅ | Group Relative Policy Optimization |
| **DAPO** | 在线RL | ✅ | Decoupled Advantage Policy Optimization |
| **GSPO** | 在线RL | ✅ | Group Sampling Policy Optimization |
| **SAPO** | 在线RL | ✅ | Step-wise Advantage Policy Optimization |
| **CISPO** | 在线RL | ✅ | Context-aware IS Policy Optimization |
| **CHORD** | 在线RL | ✅ | 多轮对话奖励传播 |
| **RLOO** | 在线RL | ✅ | Rejection Sampling with Off-policy Optimization |
| **Reinforce++** | 在线RL | ✅ | 重要性采样+方差缩减 |
| **DPO** | 离线偏好 | ✅ | Direct Preference Optimization |
| **KTO** | 离线偏好 | ✅ | 无需成对偏好数据 |
| **SimPO** | 离线偏好 | ✅ | 动态margin替代固定β |
| **ORPO** | 离线偏好 | ✅ | Odds Ratio Preference Optimization |
| **CPO** | 离线偏好 | ✅ | Contrastive Preference Optimization |
| **PPO** | 在线RL | ❌ | Proximal Policy Optimization |

### 5.2 GRPO训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     GRPO 训练架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   1. Rollout阶段                          │   │
│  │                                                         │   │
│  │   Prompt → [Policy Model] → Generate Completions        │   │
│  │        ↑           ↓                                    │   │
│  │        └──── [vLLM加速采样] ←───────────────────┐      │   │
│  │                                                 │      │   │
│  │   支持同步/异步两种模式：                          │      │   │
│  │   - colocate: 模型共置，节省显存                   │      │   │
│  │   - server: 独立服务，更高吞吐                      │      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   2. Reward计算                           │   │
│  │                                                         │   │
│  │   completions ──→ [Reward Functions] ──→ rewards       │   │
│  │                      ↓                                  │   │
│  │   支持多种奖励函数：                                     │   │
│  │   - format: 格式奖励 (如<think>标签)                     │   │
│  │   - accuracy: 正确性奖励                                 │   │
│  │   - custom: 自定义ORM插件                                │   │
│  │   - external_rm: 外部奖励模型                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   3. 策略更新                             │   │
│  │                                                         │   │
│  │   ┌─────────────────────────────────────────────────┐  │   │
│  │   │  L_GRPO = -E[log π_θ(a|s) * A(s,a)] + β*KL      │  │   │
│  │   │                                                 │  │   │
│  │   │  其中 A(s,a) = (r - mean(r)) / std(r)          │  │   │
│  │   │  组内相对优势归一化                              │  │   │
│  │   └─────────────────────────────────────────────────┘  │   │
│  │                                                         │   │
│  │   - num_generations: 每组生成数量 (通常4-24)            │   │
│  │   - beta: KL惩罚系数 (通常0.001-0.1)                    │   │
│  │   - num_iterations: 单次rollout更新次数                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 GRPO多模态训练示例

```bash
# 多模态GRPO训练 - 几何问答任务
WANDB_API_KEY=your_wandb_api_key \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MAX_PIXELS=401408 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/GEOQA_R1V_Train_8K' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --num_iterations 2 \
    --deepspeed zero3 \
    --output_dir output/GRPO_GEOQA
```

### 5.4 自定义奖励函数

```python
# examples/train/grpo/plugin/plugin.py
from swift.plugin import ORM, orms
from typing import List
import re
from math_verify import parse, verify

class MultiModalAccuracyORM(ORM):
    """多模态准确率奖励函数"""
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            
            # 1. 符号验证
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass
            
            # 2. 字符串匹配验证
            if reward == 0.0:
                try:
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                    
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()
                    
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass
            
            rewards.append(reward)
        return rewards

# 注册奖励函数
orms['external_r1v_acc'] = MultiModalAccuracyORM
```

### 5.5 DPO训练示例

```bash
# 多模态DPO训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset preference_data.jsonl \
    --train_type lora \
    --lora_rank 64 \
    --torch_dtype bfloat16 \
    --beta 0.1 \
    --label_smoothing 0.01 \
    --dpo_loss_type sigmoid \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --output_dir output/dpo_vlm
```

### 5.6 DPO数据格式

```json
// DPO偏好数据格式
{
    "images": ["/path/to/image.jpg"],
    "messages": [
        {"role": "user", "content": "<image>Describe this image."}
    ],
    "chosen": {"role": "assistant", "content": "A beautiful sunset over the ocean."},
    "rejected": {"role": "assistant", "content": "I don't know what this is."}
}
```

---

## 6. 多模态数据处理

### 6.1 多模态数据处理架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   多模态数据处理流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Image     │  │   Video     │  │   Audio     │             │
│  │  图像处理    │  │  视频处理    │  │  音频处理    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         ↓                ↓                ↓                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Template._encode() 编码                     │   │
│  │                                                         │   │
│  │  1. 图像编码: processor.image_processor                  │   │
│  │     - resize (MAX_PIXELS控制)                            │   │
│  │     - normalize                                          │   │
│  │     - patchify → image_tokens                            │   │
│  │                                                         │   │
│  │  2. 视频编码: 抽帧 → 图像序列                             │   │
│  │     - FPS_MAX_FRAMES控制最大帧数                          │   │
│  │     - VIDEO_MAX_TOKEN_NUM控制token数                      │   │
│  │                                                         │   │
│  │  3. 音频编码: processor.audio_processor                  │   │
│  │     - mel-spectrogram转换                                │   │
│  │     - AUDIO_MAX_TOKEN_NUM控制                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Template._data_collator()                   │   │
│  │                                                         │   │
│  │  - padding: left/right padding                          │   │
│  │  - packing: 多样本拼接 (提升训练速度100%+)               │   │
│  │  - mixed modality: 混合模态数据训练                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Packing技术详解

**Packing原理：** 将多个短样本拼接成一个长序列，减少padding，提高训练效率。

```
传统方式 (无packing):
┌────────┐  ┌────────┐  ┌────────┐
│ Sample1│  │ Sample2│  │ Sample3│
│ [100]  │  │ [50]   │  │ [80]   │
│ + 924  │  │ + 974  │  │ + 944  │
│ padding│  │ padding│  │ padding│
└────────┘  └────────┘  └────────┘
总token: 3072, 有效token: 230

Packing方式:
┌────────────────────────────────┐
│ Sample1 + Sample2 + Sample3    │
│ [100 + 50 + 80 = 230]          │
│ + 822 padding                  │
└────────────────────────────────┘
总token: 1024, 有效token: 230
效率提升: ~3x
```

**关键参数：**
```bash
--packing true                    # 启用packing
--padding_free true               # 无padding模式
--attn_impl flash_attn            # 必须配合flash attention
```

### 6.3 混合模态数据训练

```bash
# 混合模态训练配置
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset \
        text_dataset.jsonl \
        image_dataset.jsonl \
        video_dataset.jsonl \
    --train_type lora \
    --packing true \
    --lazy_tokenize true \
    --max_length 8192 \
    --output_dir output/mixed_modality
```

### 6.4 环境变量控制

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `MAX_PIXELS` | 图像最大像素数 | 1003520 |
| `MIN_PIXELS` | 图像最小像素数 | 3136 |
| `IMAGE_MAX_TOKEN_NUM` | 图像最大token数 | 1024 |
| `VIDEO_MAX_TOKEN_NUM` | 视频最大token数 | 128 |
| `FPS_MAX_FRAMES` | 视频最大帧数 | 16 |
| `AUDIO_MAX_TOKEN_NUM` | 音频最大token数 | 128 |
| `ROOT_IMAGE_DIR` | 图像根目录 | None |

---

## 7. 不同VLM架构支持

### 7.1 Qwen-VL系列架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen2.5-VL 架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input: Image/Video + Text                               │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Vision Encoder (ViT)                                    │   │
│  │  - 支持动态分辨率 (Dynamic Resolution)                    │   │
│  │  - Window Attention处理高分辨率图像                        │   │
│  │  - 输出: visual tokens                                   │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Visual Projector (Aligner)                              │   │
│  │  - MLP/Linear层对齐视觉和文本空间                          │   │
│  │  - 输出: 与LLM词表维度一致的visual embeddings              │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Large Language Model (Qwen2.5)                          │   │
│  │  - 处理融合后的visual+text tokens                         │   │
│  │  - 自回归生成输出                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  独立控制参数:                                                   │
│  --freeze_vit true/false      # 冻结/解冻视觉编码器              │
│  --freeze_aligner true/false  # 冻结/解冻对齐层                  │
│  --freeze_llm true/false      # 冻结/解冻语言模型                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Qwen-VL训练示例：**

```bash
# Stage 1: 只训练Aligner
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset stage1_data.jsonl \
    --train_type full \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --learning_rate 5e-6 \
    --output_dir output/stage1_aligner

# Stage 2: 全模型训练
swift sft \
    --model output/stage1_aligner \
    --dataset stage2_data.jsonl \
    --train_type full \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --learning_rate 5e-6 \
    --output_dir output/stage2_full
```

### 7.2 LLaVA系列架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLaVA 架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Vision Encoder: CLIP ViT-L/14 or SigLIP                 │   │
│  │  - 固定分辨率处理                                         │   │
│  │  - 输出: 576 image tokens (24x24 patches)                │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Projection Layer (MLP)                                  │   │
│  │  - 2层MLP对齐视觉特征                                     │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLM: Vicuna/Llama/Qwen等                                │   │
│  │  - 支持多种基座模型                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  特点:                                                          │
│  - 视觉编码器通常冻结                                          │
│  - Projection层和LLM可训练                                     │
│  - 支持高分辨率处理 (LLaVA-Next)                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 InternVL系列架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    InternVL2/3 架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Vision Encoder: InternViT                               │   │
│  │  - 6亿参数的视觉编码器                                    │   │
│  │  - 支持动态分辨率 (最高4K)                                │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Pixel Shuffle + MLP                                     │   │
│  │  - 减少视觉token数量                                      │   │
│  │  - 提高处理效率                                           │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLM: InternLM/Qwen/Llama等                              │   │
│  │  - 8K上下文窗口                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  特点:                                                          │
│  - 视觉编码器可训练                                            │
│  - 支持多图对话                                                │
│  - 强大的OCR能力                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 MiniCPM-V架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniCPM-V 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Vision Encoder: SigLIP/SAM                              │   │
│  │  - 轻量级视觉编码器                                       │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Resampler (压缩层)                                       │   │
│  │  - 将视觉token压缩到固定数量                              │   │
│  │  - 减少LLM计算负担                                        │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLM: MiniCPM (2.8B参数)                                 │   │
│  │  - 端侧友好的小模型                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  特点:                                                          │
│  - 端侧部署优化                                                │
│  - 低显存占用 (8GB可运行)                                      │
│  - 支持实时视频理解                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 训练配置示例

### 8.1 完整SFT配置

```bash
#!/bin/bash
# sft_qwen2_5_vl.sh

# 环境变量设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MAX_PIXELS=1003520
export MIN_PIXELS=200704
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 训练命令
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --dataset \
        'AI-ModelScope/LaTeX_OCR#10000' \
        'custom_vqa.jsonl' \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules ALL \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_length 8192 \
    --packing true \
    --padding_free true \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --output_dir output/sft_latex_ocr \
    --deepspeed zero2 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8
```

### 8.2 完整GRPO配置

```bash
#!/bin/bash
# grpo_multimodal.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MAX_PIXELS=401408
export WANDB_API_KEY=your_key

# 启动vLLM服务 (独立模式)
CUDA_VISIBLE_DEVICES=6,7 \
swift rollout \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --vllm_data_parallel_size 2 \
    --tensor_parallel_size 1 &

sleep 30

# GRPO训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/clevr_cogen_a_train' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --num_generations 24 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --num_iterations 2 \
    --deepspeed zero3 \
    --report_to wandb \
    --output_dir output/grpo_clevr
```

### 8.3 Megatron并行配置

```bash
#!/bin/bash
# megatron_multimodal.sh

# 8卡A100，TP=4, PP=2
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
megatron sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset multimodal_dataset.jsonl \
    --tuner_type lora \
    --tensor_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --context_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --attention_backend flash \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --save output/megatron_vlm \
    --deepspeed zero2
```

---

## 9. 最佳实践

### 9.1 训练流程建议

```
┌─────────────────────────────────────────────────────────────────┐
│                    推荐VLM训练流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段1: CPT (可选)                                               │
│  ├── 场景: 需要注入大量领域知识                                   │
│  ├── 数据: 大规模无标注领域数据                                   │
│  └── 配置: 低学习率(2e-5), 长训练时间                             │
│                          ↓                                      │
│  阶段2: SFT (必须)                                               │
│  ├── 场景: 学习指令遵循和对话格式                                 │
│  ├── 数据: 高质量instruction-response对                          │
│  └── 配置: 中等学习率(5e-5), 启用packing                          │
│                          ↓                                      │
│  阶段3: DPO/GRPO (可选)                                          │
│  ├── 场景: 对齐人类偏好，提升回答质量                             │
│  ├── 数据: 偏好对数据或奖励函数                                   │
│  └── 配置: 低学习率(1e-6), 注意KL控制                             │
│                          ↓                                      │
│  阶段4: 评估与部署                                               │
│  ├── EvalScope多维度评估                                         │
│  ├── 量化导出 (GPTQ/AWQ/FP8)                                     │
│  └── vLLM/LMDeploy部署                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 显存优化策略

| 策略 | 显存节省 | 适用场景 |
|-----|---------|---------|
| **LoRA (r=64)** | ~70% | 大多数微调场景 |
| **QLoRA (4-bit)** | ~80% | 单卡有限显存 |
| **Gradient Checkpointing** | ~50% | 全参数训练 |
| **DeepSpeed ZeRO3** | ~80% | 多卡分布式 |
| **Packing** | ~30% | 短样本数据集 |
| **Sequence Parallel** | ~40% | 长序列训练 |
| **Flash Attention 3** | ~20% | H100等支持FA3的GPU |

### 9.3 常见问题解决

```bash
# 问题1: 训练卡住/NCCL错误
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口

# 问题2: OOM显存不足
--gradient_checkpointing true
--attn_impl flash_attn
--packing true
--max_length 4096  # 降低最大长度

# 问题3: 混合模态数据训练卡住
--packing false  # 禁用packing
--lazy_tokenize true

# 问题4: 训练不稳定
--max_grad_norm 0.5  # 梯度裁剪
--warmup_ratio 0.1   # 增加warmup
--learning_rate 1e-6 # 降低学习率
```

### 9.4 性能优化检查清单

- [ ] 启用Flash Attention (`--attn_impl flash_attn`)
- [ ] 启用Packing (`--packing true`)
- [ ] 使用bfloat16 (`--torch_dtype bfloat16`)
- [ ] 配置合适的`dataloader_num_workers`
- [ ] 使用`dataset_num_proc`加速数据预处理
- [ ] 启用DeepSpeed ZeRO2/3
- [ ] 对于长序列，启用Sequence Parallel
- [ ] 对于MoE模型，启用Megatron EP并行

---

## 10. 总结

ms-swift框架为VLM训练提供了业界最全面的支持：

1. **模型覆盖广**: 支持300+多模态大模型，覆盖主流VLM架构
2. **训练方法全**: CPT/SFT/DPO/GRPO/PPO/KTO等全链路支持
3. **多模态能力强**: 图像/视频/音频统一处理，支持混合模态训练
4. **工程优化深**: Packing/Megatron/FlashAttention等性能优化
5. **易用性好**: 统一的命令行接口，丰富的最佳实践文档

通过合理利用ms-swift的各项能力，可以高效完成从领域适应到人类对齐的完整VLM训练流程。

---

*报告生成时间: 2025年*
*基于ms-swift 4.0版本*
