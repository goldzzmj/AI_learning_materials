# 多模态大模型（VLM）训练与微调完全指南

## 基于 ms-swift 框架的深度解析（Qwen3-VL 实战）

---

## 目录

1. [整体框架概览](#1-整体框架概览)
2. [VLM 架构解析（以 Qwen3-VL 为例）](#2-vlm-架构解析以-qwen3-vl-为例)
3. [训练方法详解](#3-训练方法详解)
4. [ms-swift 框架核心代码解析](#4-ms-swift-框架核心代码解析)
5. [训练方法对比与选型](#5-训练方法对比与选型)
6. [面试常考题与解答](#6-面试常考题与解答)

---

## 1. 整体框架概览

### 1.1 VLM 训练流程全景图

VLM（视觉语言模型）的训练流程分为三个主要阶段：

| 阶段 | 名称 | 核心任务 |
|------|------|----------|
| **预训练** | Pre-training | 视觉-语言对齐、多模态理解 |
| **后训练** | Post-training | 指令微调(SFT)、能力学习 |
| **对齐阶段** | Alignment | RM训练、PPO/DPO/GRPO训练 |

### 1.2 训练方法关系图

```
SFT (Supervised Fine-Tuning)
    │
    ├──→ RLHF Pipeline ──→ PPO (Proximal Policy Optimization)
    │            │
    │            └──→ RM (Reward Model)
    │
    ├──→ Direct Alignment ──→ DPO (Direct Preference Optimization)
    │            │
    │            ├──→ SimPO / ORPO / KTO
    │
    └──→ Group-Based RL ──→ GRPO (Group Relative Policy Optimization)
                   │
                   ├──→ GSPO / DAPO / SAPO / CISPO
```

### 1.3 ms-swift 框架架构

ms-swift 是一个支持 200+ VLM 和 300+ LLM 的训练框架，核心模块包括：

- **CLI 层**: `swift sft`, `swift rlhf`, `swift dpo`, `swift grpo`
- **Trainers**: SFTTrainer, PPOTrainer, DPOTrainer, GRPOTrainer
- **Models**: Qwen3-VL, InternVL, LLaVA, MiniCPM-V 等
- **Tuners**: LoRA, QLoRA, DoRA, GaLore, FourierFt
- **Distributed**: DDP, DeepSpeed ZeRO-2/3, FSDP, Megatron TP/PP/CP

---

## 2. VLM 架构解析（以 Qwen3-VL 为例）

### 2.1 Qwen3-VL 整体架构

Qwen3-VL 采用 Encoder-Decoder 多模态架构：

```
[Image Input] → Vision Encoder (ViT) → Projector (DeepStack) → LLM Backbone → [Text Output]
```

**核心组件：**

| 组件 | Qwen2.5-VL | Qwen3-VL | 改进 |
|------|-----------|----------|------|
| Vision Encoder | Patch=14, SiLU | Patch=16, GELU | 更大patch，更稳定 |
| Projector | MLP-only | MLP+DeepStack | 多层特征融合 |
| LLM | Qwen2.5 | Qwen3 (Dense/MoE) | 更强推理 |
| Position | RoPE | MRoPE-Interleave | 更好视频理解 |

### 2.2 核心组件代码实现

```python
# ============================================================
# Qwen3-VL Vision Encoder - 视觉编码器
# ============================================================
import torch
import torch.nn as nn

class Qwen3VisionEncoder(nn.Module):
    """
    Qwen3-VL 视觉编码器
    基于 ViT，使用 Conv3D 进行 patch embedding
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # [核心] Conv3D Patch Embedding
        # 输入: [B, 3, H, W] → 输出: [B, N, D]
        # patch_size=16, 1024x1024 图像 → N = (1024/16)^2 = 4096
        self.patch_embed = nn.Conv3d(
            in_channels=3,                    # RGB 通道
            out_channels=config.hidden_size,  # 1152
            kernel_size=(1, 16, 16),          # (1, patch, patch)
            stride=(1, 16, 16),
        )
        
        # [核心] 2D RoPE 位置编码
        self.pos_embed = self._create_2d_pos_embed(
            num_patches=(config.image_size // 16) ** 2,
            hidden_size=config.hidden_size
        )
        
        # 27层 Transformer
        self.layers = nn.ModuleList([
            Qwen3VisionTransformerLayer(config)
            for _ in range(27)
        ])
        
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [B, 3, H, W]
        Returns:
            hidden_states: [B, N, D]
            intermediate_features: 用于 DeepStack 的中间层特征
        """
        # Patch Embedding
        x = self.patch_embed(pixel_values.unsqueeze(2))
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 加位置编码
        x = x + self.pos_embed
        
        # [关键] 保存中间层特征用于 DeepStack
        intermediate_features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 in [8, 16, 24]:  # 保存第8/16/24层
                intermediate_features.append(x)
        
        return x, intermediate_features


# ============================================================
# DeepStack Projector - 多层特征融合投影器
# ============================================================
class DeepStackProjector(nn.Module):
    """
    DeepStack Projector: 融合多层视觉特征
    """
    
    def __init__(self, config):
        super().__init__()
        
        # MLP 投影层
        self.mlp = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.projector_hidden_size),
            nn.GELU(),
            nn.Linear(config.projector_hidden_size, config.llm_hidden_size)
        )
        
        # Pixel-Shuffle 降采样: 4x4 patches → 1 macro-patch
        self.pixel_shuffle = nn.PixelShuffle(downscale_factor=4)
        
    def forward(self, final_feature, intermediate_features):
        """
        Args:
            final_feature: [B, N, D_v] 最终层特征
            intermediate_features: [[B, N, D_v], ...] 中间层特征
        Returns:
            projected: [B, N//16, D_l] 投影后的特征
        """
        # [关键] 残差融合多层特征
        fused = final_feature
        for feat in intermediate_features:
            fused = fused + feat
        
        # MLP 投影
        projected = self.mlp(fused)  # [B, N, D_l]
        
        # Pixel-Shuffle 降采样
        B, N, D = projected.shape
        H = W = int(N ** 0.5)
        projected = projected.transpose(1, 2).view(B, D, H, W)
        downsampled = self.pixel_shuffle(projected)
        
        B, D_new, H_new, W_new = downsampled.shape
        output = downsampled.view(B, D_new, H_new * W_new).transpose(1, 2)
        
        return output


# ============================================================
# MRoPE-Interleave - 多维交错位置编码
# ============================================================
class MRoPEInterleave(nn.Module):
    """
    Multi-dimensional Rotary Position Embedding (Interleaved)
    用于视频理解，t/h/w 维度交错编码
    """
    
    def __init__(self, dim, max_position=32768):
        super().__init__()
        
        # 为 t(时间), h(高度), w(宽度) 分别创建频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//3, 2).float() / (dim//3)))
        
        self.register_buffer('inv_freq_t', inv_freq)
        self.register_buffer('inv_freq_h', inv_freq)
        self.register_buffer('inv_freq_w', inv_freq)
        
    def forward(self, positions_t, positions_h, positions_w):
        """
        Args:
            positions_t/h/w: [B, L] 各维度位置
        Returns:
            cos, sin: 旋转编码
        """
        # 计算各维度角度
        angles_t = torch.einsum('bl,f->blf', positions_t, self.inv_freq_t)
        angles_h = torch.einsum('bl,f->blf', positions_h, self.inv_freq_h)
        angles_w = torch.einsum('bl,f->blf', positions_w, self.inv_freq_w)
        
        # [关键] 交错排列: t,h,w,t,h,w,...
        angles = torch.stack([angles_t, angles_h, angles_w], dim=-1).flatten(-2)
        
        return torch.cos(angles), torch.sin(angles)
```

---

## 3. 训练方法详解

### 3.1 预训练（Pre-training）

#### 3.1.1 四阶段预训练流程

| 阶段 | 名称 | 训练内容 | 数据量 | 目标 |
|------|------|----------|--------|------|
| Stage 0 | Vision-Language Alignment | 只训练 Projector | 67B tokens | 建立视觉-语言语义桥梁 |
| Stage 1 | Full-Model Pre-training | 全模型训练 | ~1T tokens | 多模态理解能力 |
| Stage 2 | Long-Context Adaptation | 扩展上下文到 32K | - | 长文档/视频理解 |
| Stage 3 | Ultra-Long Context | 扩展到 262K | - | 超长序列处理 |

#### 3.1.2 预训练代码

```python
class VLMPretrainer:
    """VLM 预训练器 - 支持分阶段训练"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def setup_stage(self, stage):
        """配置当前训练阶段"""
        if stage == 0:
            # Stage 0: 只训练 Projector
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.model.llm.parameters():
                param.requires_grad = False
            for param in self.model.projector.parameters():
                param.requires_grad = True
                
        elif stage == 1:
            # Stage 1: 全模型训练
            for param in self.model.parameters():
                param.requires_grad = True
                
        elif stage == 2:
            # Stage 2: 长上下文适应
            self.model.llm.resize_position_embeddings(32768)
            
        elif stage == 3:
            # Stage 3: 超长上下文
            self.model.llm.resize_position_embeddings(262144)
```

---

### 3.2 监督微调（SFT）

#### 3.2.1 SFT 核心原理

**关键设计**: 只计算 assistant 回复部分的损失，prompt 部分设为 -100 (ignore_index)

```python
# ============================================================
# ms-swift SFT 训练器
# ============================================================
from transformers import Trainer
from peft import LoraConfig, get_peft_model, TaskType

class SwiftSFTTrainer(Trainer):
    """SFT 训练器 - 支持 LoRA/QLoRA/全参数微调"""
    
    def __init__(self, model, args, train_dataset, tokenizer,
                 sft_type='lora', lora_rank=8, lora_alpha=32, **kwargs):
        
        # [关键] 配置 PEFT
        if sft_type != 'full':
            model = self._setup_peft_model(model, sft_type, lora_rank, lora_alpha)
        
        super().__init__(model=model, args=args, train_dataset=train_dataset, 
                        tokenizer=tokenizer, **kwargs)
    
    def _setup_peft_model(self, model, sft_type, rank, alpha):
        """配置 LoRA/QLoRA"""
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
                         'gate_proj', 'up_proj', 'down_proj']       # MLP
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias='none',
            quantization_config={
                'load_in_4bit': sft_type == 'qlora',
                'bnb_4bit_compute_dtype': torch.bfloat16,
            } if sft_type == 'qlora' else None
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()  # 打印可训练参数
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算 SFT 损失
        [关键] 只计算 assistant 回复部分的损失
        """
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs.get('images'),
            videos=inputs.get('videos'),
        )
        
        logits = outputs.logits  # [B, L, V]
        labels = inputs['labels']  # [B, L]
        
        # Shift: 预测下一个 token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # [关键] 只计算 assistant 部分 (prompt 为 -100)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss
```

#### 3.2.2 SFT 命令行

```bash
# LoRA 微调
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen3-vl-8b-instruct \
    --dataset data.jsonl \
    --sft_type lora --lora_rank 8 --lora_alpha 32 \
    --learning_rate 1e-4 --num_train_epochs 3

# 全参数微调 (DeepSpeed)
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type qwen3-vl-8b-instruct \
    --dataset data.jsonl \
    --sft_type full --deepspeed default-zero2
```

---

### 3.3 奖励模型训练（RM）

#### 3.3.1 RM 原理

**损失函数:**
```
L = -log σ(r_θ(x, y_c) - r_θ(x, y_r) - m) + λ(r_θ(x, y_c) + r_θ(x, y_r))²
```

- `r_θ(x, y_c)`: chosen 回复的奖励分数
- `r_θ(x, y_r)`: rejected 回复的奖励分数
- `m`: margin (难度区分)
- `λ`: L2 正则化系数

```python
class RewardModel(nn.Module):
    """奖励模型 - 在 SFT 模型上加 value head"""
    
    def __init__(self, base_model_name, center_rewards_coefficient=0.01):
        super().__init__()
        
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # [关键] Value Head: 隐藏状态 → 标量奖励
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.center_rewards_coefficient = center_rewards_coefficient
    
    def forward(self, input_ids, attention_mask):
        """取序列最后一个有效 token 的隐藏状态"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]  # [B, L, D]
        
        # 取最后一个有效位置
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        last_hidden = hidden_states[
            torch.arange(batch_size),
            sequence_lengths
        ]  # [B, D]
        
        rewards = self.value_head(last_hidden).squeeze(-1)  # [B]
        return rewards
    
    def compute_loss(self, batch):
        """Bradley-Terry 损失"""
        chosen_rewards = self.forward(batch['chosen_input_ids'], 
                                      batch['chosen_attention_mask'])
        rejected_rewards = self.forward(batch['rejected_input_ids'], 
                                        batch['rejected_attention_mask'])
        
        margin = batch.get('margin', torch.zeros_like(chosen_rewards))
        
        # 主损失: 鼓励 chosen > rejected
        preference_loss = -torch.log(
            torch.sigmoid(chosen_rewards - rejected_rewards - margin)
        ).mean()
        
        # L2 正则化: 防止奖励漂移
        regularization = self.center_rewards_coefficient * (
            chosen_rewards ** 2 + rejected_rewards ** 2
        ).mean()
        
        return preference_loss + regularization
```

---

### 3.4 近端策略优化（PPO）

#### 3.4.1 PPO 原理

**四模型架构:**
- **Actor**: 策略网络 (训练中)
- **Ref**: 参考模型 (冻结的 SFT)
- **Reward**: 奖励模型 (冻结)
- **Value**: 价值网络 (训练中，从 RM 初始化)

**PPO 损失:**
```
L_PPO = -min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)

其中:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (重要性采样比率)
- A_t: 优势函数 (GAE 计算)
- ε: 裁剪超参数 (通常 0.2)
```

```python
class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(self, actor, ref_model, reward_model, value_model, 
                 tokenizer, kl_coef=0.05, cliprange=0.2, vf_coef=0.1):
        self.actor = actor
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        
        self.kl_coef = kl_coef
        self.cliprange = cliprange
        self.vf_coef = vf_coef
        
        # 冻结参考模型和奖励模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
    
    def generate_responses(self, prompts, max_new_tokens=128):
        """生成回复 (Rollout)"""
        with torch.no_grad():
            outputs = self.actor.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            responses = outputs.sequences[:, prompts.shape[1]:]
            
            # 计算 log probs
            log_probs = []
            for i, score in enumerate(outputs.scores):
                log_prob = F.log_softmax(score, dim=-1)
                token_log_prob = log_prob.gather(
                    dim=-1, index=responses[:, i:i+1]
                ).squeeze(-1)
                log_probs.append(token_log_prob)
            
            log_probs = torch.stack(log_probs, dim=1)
        
        return responses, log_probs
    
    def compute_advantages(self, rewards, values):
        """GAE (Generalized Advantage Estimation)"""
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # 反向计算 GAE
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + self.gamma * self.lam * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_loss(self, old_log_probs, new_log_probs, advantages):
        """PPO 裁剪损失"""
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        clip_fraction = ((ratio - 1).abs() > self.cliprange).float().mean()
        
        return policy_loss, clip_fraction
```

---

### 3.5 直接偏好优化（DPO）

#### 3.5.1 DPO 核心思想

**跳过 Reward Model，直接用偏好数据优化策略**

**数学推导:**
```
1. RLHF 目标: max E[r(x,y) - β log(π(y|x)/π_ref(y|x))]

2. 最优策略闭式解: π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x,y)/β)

3. 反解奖励: r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)

4. DPO 损失: L_DPO = -log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) 
                              - β log(π_θ(y_l|x)/π_ref(y_l|x)))
```

```python
class DPOTrainer(Trainer):
    """DPO 训练器 - 无需 Reward Model"""
    
    def __init__(self, model, ref_model, args, train_dataset, tokenizer,
                 beta=0.1, loss_type='sigmoid', **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                        tokenizer=tokenizer, **kwargs)
        
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_log_probs(self, model, input_ids, attention_mask, labels):
        """计算序列的对数概率 (只算 response 部分)"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs[:, :-1, :].gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # [关键] 只计算 response 部分
        response_mask = (labels[:, 1:] != -100).float()
        log_probs = (log_probs * response_mask).sum(dim=1)
        
        return log_probs
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """DPO 损失"""
        # Chosen
        policy_chosen_logps = self.compute_log_probs(
            model, inputs['chosen_input_ids'], 
            inputs['chosen_attention_mask'], inputs['chosen_labels']
        )
        with torch.no_grad():
            reference_chosen_logps = self.compute_log_probs(
                self.ref_model, inputs['chosen_input_ids'],
                inputs['chosen_attention_mask'], inputs['chosen_labels']
            )
        
        # Rejected
        policy_rejected_logps = self.compute_log_probs(
            model, inputs['rejected_input_ids'],
            inputs['rejected_attention_mask'], inputs['rejected_labels']
        )
        with torch.no_grad():
            reference_rejected_logps = self.compute_log_probs(
                self.ref_model, inputs['rejected_input_ids'],
                inputs['rejected_attention_mask'], inputs['rejected_labels']
            )
        
        # [关键] 隐式奖励差
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        logits = self.beta * (policy_logratios - reference_logratios)
        
        # 根据 loss_type 选择损失
        if self.loss_type == 'sigmoid':
            losses = -F.logsigmoid(logits)
        elif self.loss_type == 'hinge':
            losses = torch.relu(1 - logits)
        elif self.loss_type == 'ipo':
            losses = (logits - 1 / (2 * self.beta)) ** 2
        
        return losses.mean()
```

---

### 3.6 组相对策略优化（GRPO）

#### 3.6.1 GRPO 核心思想

**无需 Value Model，使用组内相对奖励**

**组采样策略:**
```
对于每个 prompt x:
1. 生成 G 个候选回复: {y_1, ..., y_G}  (G 通常 8~32)
2. 计算奖励: {r_1, ..., r_G}
3. 计算组内相对优势: A_i = (r_i - mean(r)) / std(r)
4. 使用相对优势更新策略
```

```python
class GRPOTrainer(Trainer):
    """GRPO 训练器 - 无需 Value Model"""
    
    def __init__(self, model, ref_model, reward_model=None, 
                 group_size=16, kl_coef=0.03, cliprange=0.2, **kwargs):
        super().__init__(model=model, **kwargs)
        
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.cliprange = cliprange
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def generate_group_responses(self, prompt, max_new_tokens=512):
        """为单个 prompt 生成 G 个回复"""
        all_responses = []
        all_log_probs = []
        
        for _ in range(self.group_size):
            with torch.no_grad():
                outputs = self.model.generate(
                    prompt.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                response = outputs.sequences[:, prompt.shape[0]:]
                
                # 计算 log probs
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    log_prob = F.log_softmax(score, dim=-1)
                    token_log_prob = log_prob.gather(
                        dim=-1, index=response[:, i:i+1]
                    ).squeeze(-1)
                    log_probs.append(token_log_prob)
                
                log_probs = torch.cat(log_probs, dim=0)
            
            all_responses.append(response[0])
            all_log_probs.append(log_probs)
        
        # 填充到相同长度
        max_len = max(r.shape[0] for r in all_responses)
        padded_responses = []
        padded_log_probs = []
        
        for resp, lp in zip(all_responses, all_log_probs):
            pad_len = max_len - resp.shape[0]
            padded_resp = torch.cat([
                resp,
                torch.full((pad_len,), self.tokenizer.pad_token_id, device=resp.device)
            ])
            padded_lp = torch.cat([
                lp,
                torch.zeros(pad_len, device=lp.device)
            ])
            padded_responses.append(padded_resp)
            padded_log_probs.append(padded_lp)
        
        return torch.stack(padded_responses), torch.stack(padded_log_probs)
    
    def compute_group_advantages(self, rewards):
        """组内相对优势"""
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages
    
    def compute_loss(self, model, inputs, **kwargs):
        """GRPO 损失"""
        prompts = inputs['input_ids']
        batch_size = prompts.shape[0]
        total_loss = 0
        
        for b in range(batch_size):
            prompt = prompts[b]
            
            # 生成组回复
            with torch.no_grad():
                responses, old_log_probs = self.generate_group_responses(prompt)
            
            # 计算奖励
            rewards, kl_penalty = self.compute_rewards(prompt, responses)
            
            # 组内相对优势
            advantages = self.compute_group_advantages(rewards)
            
            # 重新计算新策略 log probs
            G, L_response = responses.shape
            prompt_expanded = prompt.unsqueeze(0).expand(G, -1)
            sequences = torch.cat([prompt_expanded, responses], dim=1)
            
            outputs = model(sequences)
            logits = outputs.logits[:, prompt.shape[0]-1:-1, :]
            new_log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = new_log_probs.gather(
                dim=-1, index=responses.unsqueeze(-1)
            ).squeeze(-1)
            
            # PPO 裁剪
            ratio = torch.exp(new_log_probs - old_log_probs)
            advantages_expanded = advantages.unsqueeze(1).expand(-1, L_response)
            
            surr1 = ratio * advantages_expanded
            surr2 = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange) * advantages_expanded
            
            policy_loss = -torch.min(surr1, surr2).mean()
            kl_loss = kl_penalty.mean()
            
            loss = policy_loss + self.kl_coef * kl_loss
            total_loss += loss
        
        return total_loss / batch_size
```

---

### 3.7 其他高级方法

#### 3.7.1 DAPO (Decoupled Advantage Policy Optimization)

```python
class DAPOTrainer(GRPOTrainer):
    """DAPO - 解耦优势估计和策略更新"""
    
    def __init__(self, *args, clip_higher=0.3, clip_lower=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_higher = clip_higher
        self.clip_lower = clip_lower
    
    def compute_loss(self, model, inputs, **kwargs):
        # ... 生成回复和计算优势 ...
        
        # [关键] 不对称裁剪
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages_expanded
        surr2 = torch.where(
            advantages_expanded > 0,
            torch.clamp(ratio, 1 - self.clip_lower, 1 + self.clip_higher) * advantages_expanded,
            torch.clamp(ratio, 1 - self.clip_higher, 1 + self.clip_lower) * advantages_expanded
        )
        
        policy_loss = -torch.min(surr1, surr2).mean()
        return loss
```

#### 3.7.2 SimPO (Simple Preference Optimization)

```python
class SimPOTrainer(DPOTrainer):
    """SimPO - 无需参考模型，使用长度归一化"""
    
    def compute_loss(self, model, inputs, **kwargs):
        policy_chosen_logps = self.compute_log_probs(
            model, inputs['chosen_input_ids'],
            inputs['chosen_attention_mask'], inputs['chosen_labels']
        )
        policy_rejected_logps = self.compute_log_probs(
            model, inputs['rejected_input_ids'],
            inputs['rejected_attention_mask'], inputs['rejected_labels']
        )
        
        # [关键] 长度归一化
        chosen_length = (inputs['chosen_labels'] != -100).sum(dim=1).float()
        rejected_length = (inputs['rejected_labels'] != -100).sum(dim=1).float()
        
        normalized_chosen = policy_chosen_logps / chosen_length
        normalized_rejected = policy_rejected_logps / rejected_length
        
        gamma = 0.5  # margin
        logits = self.beta * (normalized_chosen - normalized_rejected - gamma)
        
        return -F.logsigmoid(logits).mean()
```

---

## 4. ms-swift 框架核心代码解析

### 4.1 基础训练器

```python
class SwiftTrainer(Trainer):
    """ms-swift 基础训练器"""
    
    def __init__(self, model=None, sft_type='lora', target_modules=None,
                 lora_rank=8, lora_alpha=32, **kwargs):
        
        if sft_type != 'full' and model is not None:
            model = self._setup_peft(model, sft_type, target_modules, 
                                    lora_rank, lora_alpha)
        
        super().__init__(model=model, **kwargs)
    
    def _setup_peft(self, model, sft_type, target_modules, rank, alpha):
        """配置 LoRA"""
        if target_modules is None:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj']
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=target_modules, bias='none'
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model
```

### 4.2 多模态数据 Collator

```python
class MultimodalDataCollator:
    """多模态数据批处理"""
    
    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def __call__(self, features):
        batch = {}
        
        # 文本
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        batch['input_ids'] = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        batch['attention_mask'] = self._create_attention_mask(input_ids)
        batch['labels'] = self._pad_sequence(labels, -100)
        
        # 图像
        if 'images' in features[0] and features[0]['images']:
            images = [self._process_images(f['images']) for f in features]
            batch['images'] = torch.stack(images)
        
        return batch
```

---

## 5. 训练方法对比与选型

### 5.1 方法对比表

| 方法 | 需要RM | 在线采样 | 优势估计 | 显存需求 | 稳定性 | 适用场景 |
|------|--------|----------|----------|----------|--------|----------|
| **SFT** | ❌ | ❌ | - | ⭐ | ⭐⭐⭐ | 基础能力学习 |
| **PPO** | ✅ | ✅ | GAE | ⭐⭐⭐⭐⭐ | ⭐⭐ | 复杂偏好对齐 |
| **DPO** | ❌ | ❌ | 隐式 | ⭐⭐ | ⭐⭐⭐ | 快速偏好对齐 |
| **GRPO** | 可选 | ✅ | 组相对 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 推理任务 |
| **SimPO** | ❌ | ❌ | 长度归一化 | ⭐⭐ | ⭐⭐⭐ | 长文本生成 |
| **ORPO** | ❌ | ❌ | Odds Ratio | ⭐⭐ | ⭐⭐ | SFT+DPO合并 |
| **DAPO** | 可选 | ✅ | 解耦 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 不稳定训练 |

### 5.2 选型决策树

```
有偏好数据?
├── 否 → SFT
└── 是 → 需要在线采样?
    ├── 否 → DPO/SimPO/ORPO (离线优化)
    └── 是 → 显存充足?
        ├── 否 → GRPO/RLOO (无Critic)
        └── 是 → PPO (有Critic)
```

### 5.3 超参数推荐

```python
HYPERPARAMETERS = {
    'sft': {
        'learning_rate': 1e-5,      # 全参数
        'lora_lr': 1e-4,            # LoRA
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'warmup_ratio': 0.03,
        'num_epochs': 3,
    },
    'dpo': {
        'learning_rate': 5e-7,
        'beta': 0.1,
        'loss_type': 'sigmoid',
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
    },
    'ppo': {
        'learning_rate': 1e-6,
        'kl_coef': 0.05,
        'cliprange': 0.2,
        'vf_coef': 0.1,
        'gamma': 1.0,
        'lam': 0.95,
    },
    'grpo': {
        'learning_rate': 1e-6,
        'group_size': 16,
        'kl_coef': 0.03,
        'cliprange': 0.2,
    },
}
```

---

## 6. 面试常考题与解答

### Q1: VLM 与传统 LLM 有什么区别？

| 维度 | LLM | VLM |
|------|-----|-----|
| 输入 | 仅文本 | 文本+图像/视频 |
| 架构 | Transformer Decoder | ViT+Projector+LLM |
| 训练数据 | 纯文本 | 图文对、交错数据 |
| 应用场景 | 文本生成 | 图像理解、VQA、OCR |

### Q2: 解释 LoRA 原理及其优势

**原理:** `W = W_0 + ΔW = W_0 + BA`，其中 `B∈R^(d×r), A∈R^(r×k), r << min(d,k)`

**优势:**
1. 显存节省 (只训练 <1% 参数)
2. 训练速度快
3. 可切换多任务适配器
4. 不损失预训练知识

### Q3: PPO vs DPO 对比

| 维度 | PPO | DPO |
|------|-----|-----|
| 流程 | SFT→RM→PPO (3阶段) | SFT→DPO (2阶段) |
| 需要RM | ✅ | ❌ |
| 在线采样 | ✅ | ❌ |
| 显存需求 | 高 (4模型) | 低 (2模型) |
| 稳定性 | 较低 | 较高 |

### Q4: 为什么 GRPO 适合推理任务？

**推理任务特点:**
- 奖励稀疏 (只有最后答案对/错)
- 需要探索多种解题路径
- 答案正确性可自动验证

**GRPO 优势:**
1. 组采样自然探索多种解法
2. 相对奖励自动区分好坏
3. 无需训练复杂的 Value Model
4. 可结合规则奖励

### Q5: DPO 为什么可以跳过 Reward Model？

**核心推导:**
```
1. 最优策略: π*(y|x) = (1/Z) * π_ref(y|x) * exp(r(x,y)/β)
2. 反解奖励: r(x,y) = β log(π*/π_ref) + β log Z
3. 代入 BT 模型: p(y_w>y_l|x) = σ(β log(π_θ(y_w)/π_ref(y_w)) - β log(π_θ(y_l)/π_ref(y_l)))
```

**关键洞察:** 最优策略 π* 可以直接表达奖励函数，无需显式建模 r(x,y)

### Q6: Qwen3-VL 的 DeepStack 有什么优势？

**传统 Projector:** 只用 ViT 最后一层

**DeepStack:** 融合第 8/16/24 层 + 最终层

**优势:**
- 多尺度特征 (浅层边缘纹理 + 深层语义)
- 更丰富的表示
- 下游任务适配 (OCR用浅层，理解用深层)

### Q7: 如何处理灾难性遗忘？

1. **PEFT**: 冻结大部分参数，只微调 LoRA
2. **混合数据**: 保留 30% 原始能力数据
3. **KL 约束**: `loss = task_loss + β * KL(π_new || π_ref)`
4. **经验回放**: 保留预训练样本
5. **渐进式微调**: 逐步解冻层

### Q8: DAPO 的"解耦优势"是什么意思？

**传统问题:** 优势估计和 KL 约束耦合

**DAPO 解耦:**
```python
# 传统
A_t = (r_t - mean(r)) / std(r) - β * KL_t

# DAPO: 分离
A_t = (r_t - mean(r)) / std(r)      # 纯优势
loss = policy_loss + β * KL_loss    # 单独约束
```

**不对称裁剪:**
- advantage > 0: clip_range = (1-0.1, 1+0.3)  # 更宽松向上
- advantage < 0: clip_range = (1-0.3, 1+0.1)  # 更严格向下

---

## 附录

### 常用命令速查

```bash
# SFT
swift sft --model_type qwen3-vl-8b-instruct --dataset data.jsonl --sft_type lora

# RM
swift rm --model_type qwen3-vl-8b-instruct --dataset preference.jsonl

# PPO
swift ppo --model_type qwen3-vl-8b-instruct --reward_model_type qwen3-vl-rm

# GRPO
swift grpo --model_type qwen3-vl-8b-instruct --dataset math.jsonl --group_size 16

# DPO
swift dpo --model_type qwen3-vl-8b-instruct --dataset preference.jsonl --beta 0.1

# SimPO
swift dpo --model_type qwen3-vl-8b-instruct --dataset preference.jsonl --loss_type simpo

# 推理
swift infer --model_type qwen3-vl-8b-instruct --adapters checkpoint-xxx

# 导出
swift export --adapters checkpoint-xxx --merge_lora true
```

### 数据格式

```json
// SFT
{
    "messages": [
        {"role": "user", "content": "<image>What is in the image?"},
        {"role": "assistant", "content": "There is a cat."}
    ],
    "images": ["/path/to/image.jpg"]
}

// DPO
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital is Paris.",
    "rejected": "France is a country.",
    "margin": 0.5
}

// GRPO
{
    "problem": "Solve 2x + 3 = 7",
    "answer": "x = 2"
}
```

---

**文档版本**: 1.0  
**最后更新**: 2026-02-05  
**基于**: ms-swift 4.0, Qwen3-VL, Transformers 4.40+
