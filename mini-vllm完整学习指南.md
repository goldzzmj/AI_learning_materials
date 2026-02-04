# Mini-VLLM (Nano-vLLM) 完整学习指南

## 📚 项目概述

**Nano-vLLM** 是一个由 DeepSeek 研究员开发的轻量级 VLLM 推理引擎实现，仅约 **1200 行 Python 代码**，完整实现了生产级 VLLM 的核心功能。它是学习大模型推理引擎的绝佳教材。

### 核心特性
- 🚀 **高性能离线推理** - 速度媲美原版 VLLM
- 📖 **代码简洁易读** - 约 1200 行 Python 代码
- ⚡ **完整优化套件** - 前缀缓存、张量并行、Torch 编译、CUDA Graph 等

---

## 🗂️ 项目结构

```
nano-vllm/
├── nanovllm/
│   ├── __init__.py              # 包入口，导出 LLM 和 SamplingParams
│   ├── config.py                # 配置类定义
│   ├── llm.py                   # LLM 类（继承自 LLMEngine）
│   ├── sampling_params.py       # 采样参数配置
│   ├── engine/                  # 推理引擎核心
│   │   ├── sequence.py          # 序列（请求）状态管理
│   │   ├── block_manager.py     # KV Cache 块管理器（PagedAttention）
│   │   ├── scheduler.py         # 请求调度器
│   │   ├── model_runner.py      # 模型运行器
│   │   └── llm_engine.py        # LLM 引擎主类
│   ├── layers/                  # 神经网络层实现
│   │   ├── attention.py         # 注意力机制（含 Triton Kernel）
│   │   ├── linear.py            # 线性层（支持张量并行）
│   │   ├── layernorm.py         # RMSNorm 层
│   │   ├── activation.py        # 激活函数
│   │   ├── rotary_embedding.py  # 旋转位置编码
│   │   ├── embed_head.py        # 词嵌入和输出头
│   │   └── sampler.py           # 采样器
│   ├── models/                  # 模型定义
│   │   └── qwen3.py             # Qwen3 模型实现
│   └── utils/                   # 工具函数
│       ├── context.py           # 全局上下文管理
│       └── loader.py            # 模型权重加载
├── example.py                   # 使用示例
└── bench.py                     # 性能测试
```

---

## 🎯 推荐学习顺序

### 第一阶段：基础概念（建立整体认知）
1. **sampling_params.py** - 理解采样参数
2. **config.py** - 理解配置系统
3. **utils/context.py** - 理解全局上下文

### 第二阶段：核心数据结构（理解请求管理）
4. **engine/sequence.py** - 理解序列（请求）的生命周期
5. **engine/block_manager.py** - 理解 PagedAttention 核心

### 第三阶段：调度系统（理解批处理）
6. **engine/scheduler.py** - 理解请求调度策略

### 第四阶段：模型执行（理解推理流程）
7. **layers/linear.py** - 理解张量并行
8. **layers/layernorm.py** - 理解归一化层
9. **layers/activation.py** - 理解激活函数
10. **layers/rotary_embedding.py** - 理解位置编码
11. **layers/attention.py** - 理解注意力计算
12. **layers/embed_head.py** - 理解嵌入层
13. **layers/sampler.py** - 理解采样策略

### 第五阶段：模型架构
14. **models/qwen3.py** - 理解完整模型结构
15. **utils/loader.py** - 理解权重加载

### 第六阶段：引擎核心
16. **engine/model_runner.py** - 理解模型运行
17. **engine/llm_engine.py** - 理解引擎主循环
18. **llm.py** - 最终接口

---

## 🔍 逐行代码详解


### 1. sampling_params.py - 采样参数配置

```python
# 导入 dataclass 装饰器，用于创建简洁的数据类
from dataclasses import dataclass


@dataclass  # 自动创建 __init__, __repr__, __eq__ 等方法
class SamplingParams:
    """
    采样参数类 - 控制文本生成的随机性和长度
    
    在 LLM 推理中，采样参数决定了模型如何生成下一个 token：
    - temperature: 控制随机性，值越大输出越多样
    - max_tokens: 生成文本的最大长度限制
    - ignore_eos: 是否忽略结束标记（用于测试）
    """
    temperature: float = 1.0      # 温度参数，默认1.0表示标准采样
    max_tokens: int = 64          # 最大生成token数，默认64
    ignore_eos: bool = False      # 是否忽略EOS标记，默认False

    def __post_init__(self):
        """
        初始化后验证参数有效性
        
        为什么 temperature 不能太小？
        - temperature → 0 时，softmax 退化为 argmax（贪婪解码）
        - 本项目为了简化，禁止使用贪婪解码
        """
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

**核心概念解释：**

| 参数 | 作用 | 典型值 |
|------|------|--------|
| temperature | 控制采样随机性 | 0.6-1.0 |
| max_tokens | 限制生成长度 | 64-2048 |
| ignore_eos | 测试时忽略结束标记 | False |

**温度参数详解：**
```
temperature = 1.0: 标准随机采样
temperature < 1.0: 更保守，倾向于高概率词
temperature > 1.0: 更随机，增加多样性
```

---

### 2. config.py - 配置系统

```python
import os                          # 操作系统接口，用于路径检查
from dataclasses import dataclass  # 数据类装饰器
from transformers import AutoConfig  # HuggingFace 配置加载器


@dataclass
class Config:
    """
    Nano-vLLM 全局配置类
    
    包含所有影响推理行为的参数，分为几类：
    1. 批处理参数：控制同时处理的请求数量和token数
    2. 模型参数：模型路径和长度限制
    3. 显存参数：GPU 内存使用策略
    4. 并行参数：张量并行设置
    5. 优化参数：CUDA Graph 等优化开关
    6. KV Cache参数：块大小和数量
    """
    
    # ==================== 基础参数 ====================
    model: str                              # 模型路径（必需参数）
    
    # ==================== 批处理参数 ====================
    max_num_batched_tokens: int = 16384     # 单次迭代最大token数
    max_num_seqs: int = 512                 # 最大并发序列数
    
    # ==================== 模型参数 ====================
    max_model_len: int = 4096               # 模型最大上下文长度
    
    # ==================== 显存参数 ====================
    gpu_memory_utilization: float = 0.9     # GPU显存使用率（0-1）
    
    # ==================== 并行参数 ====================
    tensor_parallel_size: int = 1           # 张量并行度（GPU数）
    
    # ==================== 优化参数 ====================
    enforce_eager: bool = False             # 强制使用eager模式（禁用CUDA Graph）
    
    # ==================== 内部状态（自动设置）====================
    hf_config: AutoConfig | None = None     # HuggingFace模型配置
    eos: int = -1                           # 结束标记ID（从tokenizer获取）
    
    # ==================== KV Cache参数 ====================
    kvcache_block_size: int = 256           # 每个KV块存储的token数
    num_kvcache_blocks: int = -1            # KV块总数（运行时计算）

    def __post_init__(self):
        """
        配置验证和初始化
        
        执行以下检查：
        1. 模型路径必须是有效目录
        2. 块大小必须是256的倍数（对齐GPU内存）
        3. 张量并行度在有效范围内
        4. 加载HuggingFace配置
        5. 确保max_model_len不超过模型支持的最大长度
        6. 确保批处理token数上限不小于模型长度
        """
        # 验证模型路径存在且是目录
        assert os.path.isdir(self.model)
        
        # 块大小必须是256的倍数 - 这是GPU内存对齐的要求
        # 256个token的块大小是性能和内存管理的平衡点
        assert self.kvcache_block_size % 256 == 0
        
        # 张量并行度限制：至少1个GPU，最多8个
        assert 1 <= self.tensor_parallel_size <= 8
        
        # 从 HuggingFace 加载模型配置
        # 包含：层数、隐藏维度、注意力头数、vocab大小等
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        # 取用户设置和模型支持的最小值作为实际最大长度
        # 防止用户设置超过模型能力的长度
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # 确保批处理token数上限足够大
        # 否则无法处理长序列
        assert self.max_num_batched_tokens >= self.max_model_len
```

**配置参数详解：**

| 参数类别 | 参数名 | 作用 | 默认值 |
|---------|--------|------|--------|
| 批处理 | max_num_batched_tokens | 单次前向传播最大token数 | 16384 |
| 批处理 | max_num_seqs | 最大并发请求数 | 512 |
| 显存 | gpu_memory_utilization | GPU显存使用比例 | 0.9 |
| 并行 | tensor_parallel_size | 张量并行GPU数 | 1 |
| KV Cache | kvcache_block_size | 每块存储token数 | 256 |

---

### 3. utils/context.py - 全局上下文管理

```python
from dataclasses import dataclass    # 数据类装饰器
import torch                          # PyTorch 深度学习框架


@dataclass
class Context:
    """
    推理上下文 - 在模型前向传播时传递关键信息
    
    为什么需要上下文？
    - 注意力计算需要知道当前是 prefill 还是 decode 阶段
    - 需要传递序列长度、块表等运行时信息
    - 避免通过函数参数层层传递，简化代码
    
    类比：就像函数调用的"环境变量"
    """
    
    # ==================== 阶段标识 ====================
    is_prefill: bool = False            # True=预填充阶段，False=解码阶段
    
    # ==================== Prefill阶段参数 ====================
    # cu_seqlens: cumulative sequence lengths（累积序列长度）
    # 用于变长序列的批处理，格式：[0, len1, len1+len2, ...]
    cu_seqlens_q: torch.Tensor | None = None   # Query序列累积长度
    cu_seqlens_k: torch.Tensor | None = None   # Key序列累积长度
    max_seqlen_q: int = 0                      # 最大Query序列长度
    max_seqlen_k: int = 0                      # 最大Key序列长度
    
    # ==================== KV Cache参数 ====================
    # slot_mapping: 每个token在KV Cache中的存储位置
    # 用于将新计算的KV值写入正确的位置
    slot_mapping: torch.Tensor | None = None
    
    # ==================== Decode阶段参数 ====================
    # context_lens: 每个序列的当前长度（用于decode阶段）
    context_lens: torch.Tensor | None = None
    
    # ==================== 块表参数 ====================
    # block_tables: 逻辑块到物理块的映射表
    # shape: [batch_size, max_num_blocks]
    block_tables: torch.Tensor | None = None


# ==================== 全局上下文实例 ====================
# 使用全局变量存储当前上下文
# 注意：这是单线程设计，多线程需要修改
_CONTEXT = Context()


def get_context():
    """
    获取当前全局上下文
    
    使用场景：
    - Attention.forward() 中判断当前阶段
    - 获取slot_mapping写入KV Cache
    """
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, 
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, 
                context_lens=None, block_tables=None):
    """
    设置全局上下文
    
    在每次模型运行前调用，设置正确的上下文信息
    
    参数设计原理：
    - is_prefill 是必需的，其他都是可选的
    - Prefill阶段需要 cu_seqlens 和 max_seqlen
    - Decode阶段需要 context_lens 和 block_tables
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, 
                       max_seqlen_q, max_seqlen_k, slot_mapping, 
                       context_lens, block_tables)


def reset_context():
    """
    重置上下文为默认值
    
    在每次推理完成后调用，防止污染下一次推理
    """
    global _CONTEXT
    _CONTEXT = Context()
```

**上下文使用流程：**

```
┌─────────────────────────────────────────────────────────┐
│  Prefill 阶段                                            │
│  set_context(is_prefill=True, cu_seqlens_q=..., ...)    │
│  model.forward(input_ids, positions)                    │
│  Attention 内部: get_context() 获取信息                  │
│  reset_context()                                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Decode 阶段                                             │
│  set_context(is_prefill=False, context_lens=..., ...)   │
│  model.forward(input_ids, positions)                    │
│  Attention 内部: get_context() 获取信息                  │
│  reset_context()                                        │
└─────────────────────────────────────────────────────────┘
```

---

### 4. engine/sequence.py - 序列（请求）状态管理

```python
from copy import copy               # 浅拷贝函数
from enum import Enum, auto         # 枚举类型
from itertools import count         # 计数器生成器

from nanovllm.sampling_params import SamplingParams  # 采样参数


class SequenceStatus(Enum):
    """
    序列状态枚举
    
    WAITING:  等待调度（刚加入或被打断）
    RUNNING:  正在运行（正在GPU上计算）
    FINISHED: 已完成生成
    
    状态转换图：
    WAITING → RUNNING → FINISHED
        ↑___________|
        (被抢占时回退)
    """
    WAITING = auto()    # 自动分配递增的整数值
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列类 - 表示一个推理请求
    
    一个 Sequence 对应用户的一次请求，包含：
    - 输入的 prompt tokens
    - 生成的 completion tokens
    - 当前的执行状态
    - 分配的 KV Cache 块表
    
    类比：就像一个"任务卡片"，记录了任务的所有信息
    """
    
    # ==================== 类属性 ====================
    block_size = 256                    # 块大小（所有序列共享）
    counter = count()                   # 序列ID生成器，从0开始递增
    
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列
        
        Args:
            token_ids: prompt 的 token ID 列表
            sampling_params: 采样参数
        """
        # -------------------- 基础信息 --------------------
        self.seq_id = next(Sequence.counter)           # 唯一序列ID
        self.status = SequenceStatus.WAITING           # 初始状态：等待
        
        # -------------------- Token序列 --------------------
        self.token_ids = copy(token_ids)               # 所有token（prompt+生成）
        self.last_token = token_ids[-1]                # 最后一个token（用于decode）
        self.num_tokens = len(self.token_ids)          # 当前总token数
        self.num_prompt_tokens = len(token_ids)        # prompt的token数
        
        # -------------------- 前缀缓存 --------------------
        # num_cached_tokens: 命中前缀缓存的token数
        # 这些token不需要重新计算，直接从缓存读取
        self.num_cached_tokens = 0
        
        # -------------------- 块表 --------------------
        # block_table: 逻辑块到物理块的映射
        # 例如：[7, 3, 5] 表示逻辑块0→物理块7，逻辑块1→物理块3...
        self.block_table = []
        
        # -------------------- 采样参数 --------------------
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # ==================== 魔术方法 ====================
    def __len__(self):
        """返回当前序列长度（token数）"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持索引访问 token_ids"""
        return self.token_ids[key]

    # ==================== 属性 ====================
    @property
    def is_finished(self):
        """检查序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的token数（不含prompt）"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取prompt部分的token IDs"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取生成部分的token IDs"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """命中缓存的块数"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """当前需要的总块数（向上取整）"""
        # (num_tokens + block_size - 1) // block_size 是向上取整公式
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中的token数"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # ==================== 方法 ====================
    def block(self, i):
        """
        获取第 i 个逻辑块中的 token IDs
        
        Args:
            i: 块索引（从0开始）
        
        Returns:
            该块包含的 token ID 列表
        """
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = (i + 1) * self.block_size
        return self.token_ids[start:end]

    def append_token(self, token_id: int):
        """
        追加一个新token到序列
        
        在 decode 阶段，每生成一个新token就调用此方法
        
        Args:
            token_id: 新生成的token ID
        """
        self.token_ids.append(token_id)     # 添加到token列表
        self.last_token = token_id           # 更新最后一个token
        self.num_tokens += 1                 # 增加token计数

    # ==================== 序列化支持 ====================
    def __getstate__(self):
        """
        自定义序列化 - 用于进程间通信
        
        优化点：
        - 如果序列已开始生成，只保存最后一个token而不是全部
        - 大幅减少多GPU通信时的数据量
        
        返回的元组：
        (num_tokens, num_prompt_tokens, num_cached_tokens, block_table, token_data)
        """
        token_data = self.token_ids if self.num_completion_tokens == 0 else self.last_token
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, 
                self.block_table, token_data)

    def __setstate__(self, state):
        """
        自定义反序列化
        
        根据序列化时的状态恢复完整序列
        """
        # 解包前4个固定字段
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        
        # 根据是否已开始生成，恢复token数据
        if self.num_completion_tokens == 0:
            # 还未开始生成，恢复完整的token列表
            self.token_ids = state[-1]
        else:
            # 已开始生成，最后一个token就是当前token
            self.last_token = state[-1]
```

**Sequence 核心概念图：**

```
┌────────────────────────────────────────────────────────────────┐
│  Sequence 结构                                                  │
├────────────────────────────────────────────────────────────────┤
│  seq_id: 0                                                      │
│  status: RUNNING                                                │
│  token_ids: [The, cat, sat, on, the, mat, and, looked]         │
│  num_tokens: 8                                                  │
│  num_prompt_tokens: 5  (The cat sat on the)                    │
│  num_completion_tokens: 3  (mat, and, looked)                  │
├────────────────────────────────────────────────────────────────┤
│  Block Table (block_size=4)                                     │
│  ┌─────────────┬─────────────┬─────────────┐                   │
│  │ Logical 0   │ Logical 1   │ Logical 2   │                   │
│  │  → Phys 7   │  → Phys 3   │  → Phys 5   │                   │
│  │ [The,cat,   │ [sat,on,    │ [mat,and,   │                   │
│  │  sat,on]    │  the,mat]   │  looked]    │                   │
│  └─────────────┴─────────────┴─────────────┘                   │
│  num_blocks: 2 (实际需要2个完整块)                              │
│  last_block_num_tokens: 3 (第2个块只有3个token)                │
└────────────────────────────────────────────────────────────────┘
```

---


### 5. engine/block_manager.py - PagedAttention 核心实现

```python
from collections import deque        # 双端队列，用于高效的头尾操作
import xxhash                        # 高性能非加密哈希库
import numpy as np                   # 数值计算库

from nanovllm.engine.sequence import Sequence  # 序列类


class Block:
    """
    KV Cache 物理块
    
    类比：就像内存分页系统中的"物理页框"
    每个块存储固定数量的token的KV值
    
    关键设计：
    - ref_count: 引用计数，支持块共享（copy-on-write）
    - hash: 块内容的哈希值，用于前缀缓存查找
    - token_ids: 块中存储的token IDs（用于验证缓存命中）
    """
    
    def __init__(self, block_id):
        """
        初始化块
        
        Args:
            block_id: 物理块ID（在blocks数组中的索引）
        """
        self.block_id = block_id        # 物理块ID
        self.ref_count = 0              # 引用计数，0表示空闲
        self.hash = -1                  # 内容哈希，-1表示未计算
        self.token_ids = []             # 块中的token IDs

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的内容信息
        
        在块被填满时调用，计算并存储哈希值
        
        Args:
            hash: 块内容的哈希值
            token_ids: 块中的token IDs
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块为初始状态
        
        在分配块时调用
        """
        self.ref_count = 1              # 新分配的块引用计数为1
        self.hash = -1                  # 哈希未计算
        self.token_ids = []             # 清空token列表


class BlockManager:
    """
    块管理器 - PagedAttention 的核心实现
    
    类比：就像操作系统的内存管理器
    - 管理物理块的分配和回收
    - 维护逻辑块到物理块的映射
    - 实现前缀缓存（通过哈希查找）
    
    核心数据结构：
    1. blocks: 所有物理块的数组
    2. hash_to_block_id: 哈希值到物理块ID的映射（前缀缓存）
    3. free_block_ids: 空闲块ID队列
    4. used_block_ids: 已使用块ID集合
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器
        
        Args:
            num_blocks: 物理块总数（由显存大小决定）
            block_size: 每个块存储的token数
        """
        self.block_size = block_size
        
        # 创建所有物理块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        
        # 前缀缓存：哈希值 → 物理块ID
        # 用于快速查找相同内容的块
        self.hash_to_block_id: dict[int, int] = dict()
        
        # 空闲块队列 - 使用deque实现O(1)的popleft
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
        # 已使用块集合 - 用于快速判断块状态
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算token序列的哈希值
        
        使用xxhash64算法，特点：
        - 速度快（比MD5/SHA快得多）
        - 碰撞率低（足够用于缓存）
        - 非加密（不需要安全性）
        
        支持前缀哈希链：
        - prefix参数是前一个块的哈希值
        - 这样可以检测连续块序列是否匹配
        
        Args:
            token_ids: token ID列表
            prefix: 前缀哈希值（-1表示无前缀）
        
        Returns:
            64位哈希值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前缀哈希作为种子，实现哈希链
            h.update(prefix.to_bytes(8, "little"))
        # 将token数组转为字节序列
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配指定ID的块
        
        内部方法，将块从空闲队列移到使用集合
        
        Args:
            block_id: 要分配的块ID
        
        Returns:
            分配后的Block对象
        """
        block = self.blocks[block_id]
        # 断言：只有空闲块才能被分配
        assert block.ref_count == 0
        block.reset()                           # 重置块状态
        self.free_block_ids.remove(block_id)    # 从空闲队列移除
        self.used_block_ids.add(block_id)       # 添加到使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放指定ID的块
        
        内部方法，将块从使用集合移到空闲队列
        
        Args:
            block_id: 要释放的块ID
        """
        # 断言：只有引用计数为0的块才能被释放
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)    # 从使用集合移除
        self.free_block_ids.append(block_id)    # 添加到空闲队列尾部

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够空闲块分配给序列
        
        Args:
            seq: 要分配的序列
        
        Returns:
            是否有足够空闲块
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配块 - PagedAttention的核心算法
        
        这是整个系统最复杂的逻辑之一：
        1. 遍历序列的每个逻辑块
        2. 计算块的哈希值（支持前缀链）
        3. 尝试从缓存中找到匹配的物理块
        4. 如果缓存命中，增加引用计数
        5. 如果缓存未命中，分配新块
        
        Args:
            seq: 要分配块的序列
        """
        # 断言：序列不能已有块表（防止重复分配）
        assert not seq.block_table
        
        h = -1                      # 初始哈希值（无前缀）
        cache_miss = False          # 是否发生缓存未命中
        
        # 遍历序列的每个逻辑块
        for i in range(seq.num_blocks):
            # 获取当前逻辑块的token IDs
            token_ids = seq.block(i)
            
            # 计算块的哈希值
            # 只有完整块（填满block_size个token）才计算哈希
            # 不完整块哈希设为-1，不参与缓存
            if len(token_ids) == self.block_size:
                # 使用前缀哈希链
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id.get(h, -1)
            else:
                h = -1
                block_id = -1
            
            # 验证缓存命中（哈希可能碰撞，需要二次验证）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 缓存未命中
                cache_miss = True
            
            if cache_miss:
                # 缓存未命中：分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中！
                seq.num_cached_tokens += self.block_size  # 增加缓存命中计数
                
                if block_id in self.used_block_ids:
                    # 块已被使用，增加引用计数（共享）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块在缓存但未被使用，重新分配
                    block = self._allocate_block(block_id)
            
            # 如果是完整块，更新块的哈希和token信息
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            # 将物理块ID添加到序列的块表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的所有块
        
        在序列完成或被抢占时调用
        
        Args:
            seq: 要释放的序列
        """
        # 逆序遍历块表（从后往前释放）
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1            # 减少引用计数
            
            # 引用计数为0时，真正释放块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        # 重置序列的缓存状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以向序列追加token
        
        在decode阶段，可能需要分配新块
        
        Args:
            seq: 要追加的序列
        
        Returns:
            是否可以追加
        """
        # 条件：序列长度对block_size取模等于1时，需要新块
        # 例如：block_size=4，当len=5,9,13...时需要新块
        # len % block_size == 1 表示刚进入新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        处理序列追加token时的块操作
        
        在decode阶段，当序列追加新token时：
        1. 如果刚进入新块，分配新块
        2. 如果刚填满块，计算并存储哈希
        
        Args:
            seq: 正在追加的序列
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        # 情况1：刚进入新块（需要分配）
        if len(seq) % self.block_size == 1:
            # 断言：上一个块必须有哈希（已填满）
            assert last_block.hash != -1
            # 分配新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        
        # 情况2：刚填满块（需要计算哈希）
        elif len(seq) % self.block_size == 0:
            # 断言：当前块不应该有哈希（刚填满）
            assert last_block.hash == -1
            
            # 获取当前块的所有token
            token_ids = seq.block(seq.num_blocks - 1)
            
            # 获取前缀哈希（如果有前一个块）
            if len(block_table) > 1:
                prefix = self.blocks[block_table[-2]].hash
            else:
                prefix = -1
            
            # 计算并存储哈希
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        
        # 情况3：块未填满（无需操作）
        else:
            assert last_block.hash == -1
```

**PagedAttention 核心概念图解：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PagedAttention 原理                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  逻辑块（Logical Blocks）                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │  Block 0    │  │  Block 1    │  │  Block 2    │                     │
│  │ [The,cat,   │  │ [sat,on,    │  │ [mat,and,   │                     │
│  │  sat,on]    │  │  the,mat]   │  │  looked]    │                     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                     │
│         │                │                │                            │
│         ▼                ▼                ▼                            │
│  Block Table: [7, 3, 5]  ← 逻辑到物理的映射                              │
│                                                                         │
│  物理块（Physical Blocks）                                               │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  0   1   2   3   4   5   6   7   8   9   10  11  ...       │       │
│  │ [ ] [ ] [ ] [B] [ ] [C] [ ] [A] [ ] [ ] [ ] [ ] ...       │       │
│  └─────────────────────────────────────────────────────────────┘       │
│         A=Block0  B=Block1  C=Block2                                   │
│                                                                         │
│  关键特性：                                                              │
│  1. 物理块不连续 - 解决内存碎片问题                                       │
│  2. 块表映射 - 灵活管理内存                                               │
│  3. 引用计数 - 支持块共享（copy-on-write）                               │
│  4. 哈希缓存 - 前缀匹配避免重复计算                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**前缀缓存示例：**

```
序列A: "The cat sat on the mat" → 块表 [7, 3]
序列B: "The cat sat on the table" → 块表 [7, 5]  (共享Block 0)

哈希表:
┌─────────────────────┬──────────┐
│ Hash("The cat...")  │ → 7      │
│ Hash("sat on...")   │ → 3      │
│ Hash("sat on...")   │ → 5      │ (不同内容，相同前缀长度)
└─────────────────────┴──────────┘
```

---

### 6. engine/scheduler.py - 请求调度器

```python
from collections import deque         # 双端队列

from nanovllm.config import Config    # 配置类
from nanovllm.engine.sequence import Sequence, SequenceStatus  # 序列相关
from nanovllm.engine.block_manager import BlockManager         # 块管理器


class Scheduler:
    """
    调度器 - 决定哪些请求在何时执行
    
    类比：就像操作系统的进程调度器
    - 管理等待队列和运行队列
    - 决定下一个执行哪个请求
    - 处理资源不足时的抢占
    
    调度策略：
    1. 优先执行 prefill（新请求）
    2. 然后执行 decode（生成中请求）
    3. 资源不足时抢占低优先级请求
    
    这种策略保证了：
    - 新请求能快速得到响应（低延迟）
    - 生成中的请求能持续进行（高吞吐）
    """
    
    def __init__(self, config: Config):
        """
        初始化调度器
        
        Args:
            config: 全局配置
        """
        # 从配置读取调度参数
        self.max_num_seqs = config.max_num_seqs                   # 最大并发数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 最大批处理token数
        self.eos = config.eos                                      # 结束标记ID
        
        # 创建块管理器
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, 
            config.kvcache_block_size
        )
        
        # 等待队列 - 新请求或被打断的请求
        self.waiting: deque[Sequence] = deque()
        
        # 运行队列 - 正在GPU上执行的请求
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        检查是否所有请求都已完成
        
        Returns:
            True if 等待队列和运行队列都为空
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新请求到等待队列
        
        Args:
            seq: 要添加的序列
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度请求 - 核心调度算法
        
        返回要执行的请求列表和是否是prefill阶段
        
        调度策略：
        1. 首先尝试调度 waiting 队列中的请求（prefill）
        2. 如果没有新请求，调度 running 队列中的请求（decode）
        3. 资源不足时，抢占 running 队列末尾的请求
        
        Returns:
            (scheduled_seqs, is_prefill): 调度的序列列表和是否是prefill
        """
        # ==================== Phase 1: Prefill ====================
        # 优先处理等待队列中的新请求
        scheduled_seqs = []           # 本次调度的序列
        num_seqs = 0                  # 已调度序列数
        num_batched_tokens = 0        # 已调度token数
        
        # 循环从等待队列取请求，直到达到上限
        while self.waiting and num_seqs < self.max_num_seqs:
            # 查看队列头部请求（不取出）
            seq = self.waiting[0]
            
            # 检查是否超过批处理token上限
            # 注意：只计算未缓存的token（缓存的不需要计算）
            new_tokens = len(seq) - seq.num_cached_tokens
            if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
                break
            
            # 检查是否有足够块分配给这个请求
            if not self.block_manager.can_allocate(seq):
                break
            
            # 通过所有检查，正式调度这个请求
            num_seqs += 1
            self.block_manager.allocate(seq)           # 分配KV块
            num_batched_tokens += new_tokens           # 累加token数
            
            # 更新序列状态
            seq.status = SequenceStatus.RUNNING
            
            # 从等待队列移除，加入运行队列
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # 如果调度了任何请求，返回进行prefill
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # ==================== Phase 2: Decode ====================
        # 没有新请求，处理正在生成的请求
        # Decode阶段每个请求只处理1个token
        while self.running and num_seqs < self.max_num_seqs:
            # 从运行队列头部取请求
            seq = self.running.popleft()
            
            # 检查是否可以追加token（可能需要新块）
            while not self.block_manager.can_append(seq):
                # 资源不足，需要抢占
                if self.running:
                    # 抢占运行队列末尾的请求（最少优先）
                    self.preempt(self.running.pop())
                else:
                    # 没有其他请求可抢占，只能抢占当前请求
                    self.preempt(seq)
                    break
            else:
                # can_append返回True，可以执行decode
                num_seqs += 1
                self.block_manager.may_append(seq)     # 处理可能的块分配
                scheduled_seqs.append(seq)
        
        # Decode阶段必须至少调度一个请求
        assert scheduled_seqs
        
        # 将调度的请求放回运行队列头部（保持顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占请求 - 将运行中的请求放回等待队列
        
        在资源不足时调用，释放该请求占用的块
        
        Args:
            seq: 要抢占的序列
        """
        seq.status = SequenceStatus.WAITING        # 改回等待状态
        self.block_manager.deallocate(seq)         # 释放所有块
        self.waiting.appendleft(seq)               # 放到等待队列头部（优先调度）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理 - 处理模型生成的token
        
        将生成的token添加到序列，检查是否完成
        
        Args:
            seqs: 本次处理的序列
            token_ids: 生成的token ID列表
        """
        for seq, token_id in zip(seqs, token_ids):
            # 追加生成的token
            seq.append_token(token_id)
            
            # 检查是否满足结束条件
            should_finish = False
            
            # 条件1：生成了EOS标记（且不允许忽略）
            if not seq.ignore_eos and token_id == self.eos:
                should_finish = True
            
            # 条件2：达到最大生成长度
            if seq.num_completion_tokens == seq.max_tokens:
                should_finish = True
            
            if should_finish:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)     # 释放块
                self.running.remove(seq)                # 从运行队列移除
```

**调度流程图解：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           调度流程                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   等待队列 (Waiting)        运行队列 (Running)         GPU              │
│   ┌─────────────┐          ┌─────────────┐                             │
│   │  Seq A      │          │  Seq D      │ ────────► Prefill/Decode    │
│   │  Seq B      │          │  Seq E      │                             │
│   │  Seq C      │          └─────────────┘                             │
│   └──────┬──────┘                                                     │
│          │                                                              │
│          ▼ schedule()                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Phase 1: Prefill                                               │  │
│   │  - 从waiting取请求                                              │  │
│   │  - 检查token上限和块可用性                                       │  │
│   │  - 分配块，加入running                                          │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          │ (如果没有新请求)                                             │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Phase 2: Decode                                                │  │
│   │  - 从running取请求                                              │  │
│   │  - 检查是否可以追加token                                         │  │
│   │  - 资源不足时抢占末尾请求                                        │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   抢占示例：                                                             │
│   running = [D, E, F], 需要块但不足                                      │
│   → 抢占 F → waiting = [F, A, B, C]                                     │
│   → D, E 继续 decode                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---


### 7. layers/linear.py - 线性层（张量并行实现）

```python
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块
import torch.nn.functional as F       # 神经网络函数
import torch.distributed as dist      # 分布式训练/推理


def divide(numerator, denominator):
    """
    整除断言函数
    
    确保分子能被分母整除，用于张量并行切分验证
    
    Args:
        numerator: 被除数
        denominator: 除数
    
    Returns:
        整除结果
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    线性层基类 - 提供张量并行的基础功能
    
    张量并行（Tensor Parallelism, TP）原理：
    - 将大矩阵按行或列切分到多个GPU
    - 每个GPU只存储部分权重
    - 前向传播后通过all-reduce合并结果
    
    类比：就像把一个大任务分给多个人做，最后汇总结果
    """
    
    def __init__(
        self,
        input_size: int,                  # 输入维度
        output_size: int,                 # 输出维度
        bias: bool = False,               # 是否使用偏置
        tp_dim: int | None = None,        # 张量并行切分维度（0=列，1=行）
    ):
        super().__init__()
        self.tp_dim = tp_dim              # 切分维度
        self.tp_rank = dist.get_rank()    # 当前GPU的rank
        self.tp_size = dist.get_world_size()  # 总GPU数
        
        # 创建权重参数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 附加weight_loader方法，用于加载切分后的权重
        self.weight.weight_loader = self.weight_loader
        
        # 可选的偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """子类必须实现前向传播"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制线性层 - 权重在所有GPU上完全相同
    
    用于不需要切分的层（如最终的输出层）
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载权重 - 直接复制完整权重
        
        Args:
            param: 目标参数
            loaded_weight: 加载的权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准线性变换"""
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层 - 按输出维度切分
    
    切分方式：output_size → output_size / tp_size
    
    数学原理：
    Y = X @ W^T
    将 W 按列切分：[W1, W2, ..., Wn]
    每个GPU计算：Yi = X @ Wi^T
    结果在输出维度拼接
    
    图示：
    输入 X: [batch, input_size]
    权重 W: [output_size, input_size]
    
    GPU 0: W0 = W[0:output//2, :]      GPU 1: W1 = W[output//2:, :]
           ↓                                   ↓
    Y0 = X @ W0^T                      Y1 = X @ W1^T
           ↓                                   ↓
    Y = [Y0, Y1]  (在输出维度拼接)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 计算切分后的输出维度
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载列切分的权重
        
        Args:
            param: 目标参数（已切分大小）
            loaded_weight: 完整权重
        """
        param_data = param.data
        # 计算当前GPU负责的切片
        shard_size = param_data.size(self.tp_dim)  # 切分后的大小
        start_idx = self.tp_rank * shard_size      # 起始索引
        # 从完整权重中切取对应部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """列并行线性变换"""
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层 - 支持多个输出合并切分
    
    用于 QKV 投影中的 gate_proj + up_proj 合并
    
    典型用例：
    gate_up_proj = MergedColumnParallelLinear(hidden_size, [inter_size, inter_size])
    实际创建大小为 2 * inter_size 的权重，逻辑上分为两部分
    """
    
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],          # 多个输出大小
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 总输出大小 = 所有输出大小之和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        加载合并权重的特定部分
        
        Args:
            param: 目标参数
            loaded_weight: 加载的权重
            loaded_shard_id: 要加载的部分索引（0或1）
        """
        param_data = param.data
        # 计算该部分在切分后权重中的偏移
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        
        # 对加载的权重也进行切分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV并行线性层 - 专门用于注意力QKV投影
    
    将 Q、K、V 三个投影合并为一个线性层
    输出格式：[Q_part, K_part, V_part]
    
    设计原因：
    1. 减少kernel launch开销（一个矩阵乘代替三个）
    2. 统一处理张量并行切分
    
    权重布局：
    [Q_heads * head_dim, K_heads * head_dim, V_heads * head_dim]
    """
    
    def __init__(
        self,
        hidden_size: int,                 # 隐藏层大小
        head_size: int,                   # 每个头的大小
        total_num_heads: int,             # 总头数（Q）
        total_num_kv_heads: int | None = None,  # K,V头数（可能少于Q）
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        self.head_size = head_size
        # 每个GPU负责的头数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        
        # 总输出大小 = Q大小 + K大小 + V大小
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        加载Q、K或V的权重
        
        Args:
            param: 目标参数
            loaded_weight: 加载的权重
            loaded_shard_id: "q", "k", 或 "v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        # 计算该部分在合并权重中的位置
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层 - 按输入维度切分
    
    切分方式：input_size → input_size / tp_size
    
    数学原理：
    Y = X @ W^T
    将 W 按行切分，X 也对应切分
    
    每个GPU计算：Yi = Xi @ Wi^T
    结果通过 all-reduce 求和
    
    图示：
    GPU 0: X0 = X[:, 0:input//2]       GPU 1: X1 = X[:, input//2:]
           W0 = W[:, 0:input//2]            W1 = W[:, input//2:]
           ↓                                   ↓
    Y0 = X0 @ W0^T                     Y1 = X1 @ W1^T
           ↓                                   ↓
    Y = Y0 + Y1  (all-reduce求和)
    
    注意：行并行需要 all-reduce，列并行不需要
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 计算切分后的输入维度
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载行切分的权重"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行并行线性变换
        
        注意：只有rank 0的偏置有效，避免重复加
        """
        # 线性变换
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        
        # 多GPU时，通过all-reduce合并结果
        if self.tp_size > 1:
            dist.all_reduce(y)
        
        return y
```

**张量并行对比图：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        张量并行策略对比                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Column Parallel (列并行)                                               │
│  ┌─────────────────┐                                                    │
│  │  输入 X         │                                                    │
│  │  [batch, in]    │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│     ┌─────┴─────┐                                                       │
│     ▼           ▼                                                       │
│  ┌───────┐  ┌───────┐                                                   │
│  │GPU 0  │  │GPU 1  │                                                   │
│  │W0     │  │W1     │  W = [W0; W1] 按行拼接                            │
│  │[in,out│  │[in,out│                                                   │
│  │  /2]  │  │  /2]  │                                                   │
│  └───┬───┘  └───┬───┘                                                   │
│      │          │                                                       │
│      ▼          ▼                                                       │
│  Y0 = X@W0   Y1 = X@W1                                                  │
│      │          │                                                       │
│      └────┬─────┘                                                       │
│           ▼                                                             │
│       Y = [Y0, Y1]  在输出维度拼接                                       │
│                                                                         │
│  特点：无需通信，输出维度翻倍                                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Row Parallel (行并行)                                                  │
│  ┌─────────────────┐                                                    │
│  │  输入 X         │                                                    │
│  │  [batch, in]    │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│     ┌─────┴─────┐                                                       │
│     ▼           ▼                                                       │
│  ┌───────┐  ┌───────┐                                                   │
│  │GPU 0  │  │GPU 1  │                                                   │
│  │X0     │  │X1     │  X = [X0, X1] 按列拼接                            │
│  │W0     │  │W1     │  W = [W0, W1] 按列拼接                            │
│  │[in/2, │  │[in/2, │                                                   │
│  │ out]  │  │ out]  │                                                   │
│  └───┬───┘  └───┬───┘                                                   │
│      │          │                                                       │
│      ▼          ▼                                                       │
│  Y0 = X0@W0  Y1 = X1@W1                                                 │
│      │          │                                                       │
│      └────┬─────┘                                                       │
│           ▼                                                             │
│       Y = Y0 + Y1  (all-reduce求和)                                      │
│                                                                         │
│  特点：需要all-reduce，输出维度不变                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 8. layers/layernorm.py - RMSNorm 层

```python
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    
    相比传统 LayerNorm，RMSNorm 去掉了均值计算，只使用均方根：
    - LayerNorm: (x - mean) / sqrt(var + eps)
    - RMSNorm: x / sqrt(mean(x^2) + eps)
    
    优势：
    1. 计算更简单（少一次均值计算）
    2. 在LLM中效果相当甚至更优
    3. 与残差连接配合更好
    
    公式：RMSNorm(x) = x / RMS(x) * weight
          其中 RMS(x) = sqrt(mean(x^2))
    
    类比：就像对向量做"长度归一化"，保持方向不变
    """
    
    def __init__(
        self,
        hidden_size: int,               # 隐藏层大小
        eps: float = 1e-6,              # 数值稳定性常数
    ) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile                      # PyTorch 2.0 编译优化
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        标准RMSNorm前向传播
        
        Args:
            x: 输入张量 [..., hidden_size]
        
        Returns:
            归一化后的张量
        """
        # 保存原始数据类型（可能是fp16/bf16）
        orig_dtype = x.dtype
        
        # 转为float32进行计算（提高数值稳定性）
        x = x.float()
        
        # 计算均方值：mean(x^2, dim=-1, keepdim=True)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 归一化：x / sqrt(var + eps)
        # rsqrt = 1 / sqrt，效率更高
        x.mul_(torch.rsqrt(var + self.eps))
        
        # 转回原始类型，并应用可学习权重
        x = x.to(orig_dtype).mul_(self.weight)
        
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合残差连接的RMSNorm
        
        将 x + residual 和 RMSNorm 融合为一步，减少内存访问
        
        标准流程：
        hidden = x + residual
        residual = hidden
        hidden = RMSNorm(hidden)
        
        融合后：
        hidden = RMSNorm(x + residual), residual = x + residual
        
        Args:
            x: 输入张量
            residual: 残差连接
        
        Returns:
            (归一化结果, 更新后的残差)
        """
        orig_dtype = x.dtype
        
        # 融合加法：x + residual
        x = x.float().add_(residual.float())
        
        # 保存残差（用于下一层）
        residual = x.to(orig_dtype)
        
        # RMSNorm计算
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        统一前向接口
        
        Args:
            x: 输入张量
            residual: 可选的残差连接
        
        Returns:
            无residual: 归一化结果
            有residual: (归一化结果, 更新后的残差)
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
```

**RMSNorm vs LayerNorm：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        归一化方法对比                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LayerNorm                                                              │
│  ┌─────────────────┐                                                    │
│  │  x = [1, 2, 3]  │                                                    │
│  │  mean = 2       │  ← 计算均值                                        │
│  │  var = 2/3      │  ← 计算方差                                        │
│  │  x_norm =       │                                                    │
│  │  (x-mean)/sqrt  │                                                    │
│  │  (var+eps)      │                                                    │
│  └─────────────────┘                                                    │
│                                                                         │
│  RMSNorm (LLM常用)                                                      │
│  ┌─────────────────┐                                                    │
│  │  x = [1, 2, 3]  │                                                    │
│  │  rms = sqrt(    │  ← 只计算均方根                                    │
│  │    mean(x²))    │                                                    │
│  │  = sqrt(14/3)   │                                                    │
│  │  x_norm = x/rms │                                                    │
│  └─────────────────┘                                                    │
│                                                                         │
│  区别：RMSNorm去掉了减均值操作，计算更快                                  │
│  原理：在预训练Transformer中，均值为0的假设往往成立                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 9. layers/activation.py - 激活函数

```python
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块
import torch.nn.functional as F       # 神经网络函数


class SiluAndMul(nn.Module):
    """
    SwiGLU 激活函数 - SiLU + 逐元素乘法
    
    这是 LLaMA、Qwen 等现代 LLM 使用的激活函数
    
    公式：SwiGLU(x) = SiLU(x1) * x2
          其中 x = [x1, x2]（在最后一维切分）
          SiLU(x) = x * sigmoid(x)
    
    为什么有效？
    1. SiLU（Swish）是平滑的非线性激活
    2. 门控机制（与x2相乘）控制信息流
    3. 在Transformer中表现优于ReLU/GELU
    
    类比：就像一个"智能阀门"，根据输入决定通过多少信息
    """
    
    def __init__(self):
        super().__init__()

    @torch.compile                      # 编译优化
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU前向传播
        
        Args:
            x: 输入张量，最后一维大小为2*n（会被切分为两部分）
        
        Returns:
            SiLU(前半部分) * 后半部分
        """
        # 在最后一维将x切分为两半
        # 例如：x.shape = [batch, 2*inter_size]
        # x1.shape = x2.shape = [batch, inter_size]
        x1, x2 = x.chunk(2, -1)
        
        # SiLU(x1) * x2
        # SiLU(x) = x * sigmoid(x)，是平滑的门控函数
        return F.silu(x1) * x2
```

**激活函数对比：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        激活函数对比                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ReLU:    f(x) = max(0, x)                                              │
│           简单但不够平滑，负数完全截断                                    │
│                                                                         │
│  GELU:    f(x) = x * Φ(x)  (Φ是标准正态CDF)                             │
│           平滑，计算较复杂                                                │
│                                                                         │
│  SiLU/Swish: f(x) = x * sigmoid(x)                                      │
│           平滑，自门控，LLM常用                                           │
│           形状：先下降后上升，类似Swish                                   │
│                                                                         │
│  SwiGLU:  f(x, y) = SiLU(x) * y                                         │
│           双输入门控，现代LLM标配                                         │
│           gate_proj和up_proj合并后使用                                    │
│                                                                         │
│  图示：                                                                 │
│  ReLU     ████          GELU      ▄▄▄▄                                  │
│          █             (平滑版ReLU)                                     │
│  ────────┴────────     ────────┬────────                                │
│                                                                         │
│  SiLU      ╭─╮           SwiGLU需要两个输入                             │
│           ╱   ╲                                                         │
│  ────────┬────────     x ──► SiLU ──┐                                   │
│          (平滑下凹)                  * ──► 输出                          │
│                              y ─────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 10. layers/rotary_embedding.py - 旋转位置编码 (RoPE)

```python
from functools import lru_cache      # 缓存装饰器
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量
    
    RoPE 核心思想：通过旋转矩阵将位置信息编码到Query/Key中
    
    数学原理：
    对于二维向量 [x1, x2]，旋转θ角度：
    [x1']   [cosθ  -sinθ] [x1]
    [x2'] = [sinθ   cosθ] [x2]
    
    即：x1' = x1*cosθ - x2*sinθ
        x2' = x2*cosθ + x1*sinθ
    
    扩展到高维：将特征维度两两配对，每对应用不同频率的旋转
    
    Args:
        x: 输入张量 [num_tokens, num_heads, head_dim]
        cos: 余弦值 [num_tokens, 1, head_dim//2]
        sin: 正弦值 [num_tokens, 1, head_dim//2]
    
    Returns:
        应用旋转编码后的张量
    """
    # 将最后一维切分为两半
    # x1, x2 形状: [num_tokens, num_heads, head_dim//2]
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    
    # 应用旋转矩阵
    # y1 = x1 * cos - x2 * sin
    # y2 = x2 * cos + x1 * sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    # 拼接回原始维度
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块
    
    RoPE 的优势：
    1. 相对位置编码：内积只与相对位置有关
    2. 长序列外推：可以处理比训练更长的序列
    3. 与注意力天然结合：直接作用于Q、K
    
    频率计算公式：
    θ_i = base^(-2i/d)  其中 i ∈ [0, d/2), d = head_dim
    
    位置 m 的旋转角度：m * θ_i
    """
    
    def __init__(
        self,
        head_size: int,                 # 每个头的大小
        rotary_dim: int,                # 应用旋转编码的维度
        max_position_embeddings: int,   # 最大位置数
        base: float,                    # 频率基数（通常是10000或1000000）
    ) -> None:
        super().__init__()
        self.head_size = head_size
        
        # 当前实现要求 rotary_dim == head_size
        # 部分实现支持只旋转部分维度
        assert rotary_dim == head_size
        
        # 计算频率的倒数（逆频率）
        # 公式：1 / (base^(i/rotary_dim))，其中 i = 0, 2, 4, ..., rotary_dim-2
        # 形状: [rotary_dim//2]
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 所有可能的位置索引 [0, 1, 2, ..., max_position-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 计算每个位置、每个维度的频率
        # freqs[m, i] = m * inv_freq[i]
        # 形状: [max_position, rotary_dim//2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 计算余弦和正弦值
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 拼接cos和sin，并增加维度用于广播
        # 形状: [max_position, 1, rotary_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        
        # 注册为buffer（不是参数，不参与训练）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,        # 位置索引 [num_tokens]
        query: torch.Tensor,            # Query [num_tokens, num_heads, head_dim]
        key: torch.Tensor,              # Key [num_tokens, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码到Q和K
        
        Args:
            positions: 每个token的位置索引
            query: Query张量
            key: Key张量
        
        Returns:
            (旋转后的Query, 旋转后的Key)
        """
        # 根据位置索引获取对应的cos/sin值
        # cos_sin形状: [num_tokens, 1, head_dim]
        cos_sin = self.cos_sin_cache[positions]
        
        # 切分为cos和sin
        # 形状: [num_tokens, 1, head_dim//2]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # 分别应用到Q和K
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取RoPE实例（带缓存）
    
    使用lru_cache确保相同参数的RoPE只创建一次
    节省内存，提高性能
    
    Args:
        head_size: 头大小
        rotary_dim: 旋转维度
        max_position: 最大位置
        base: 频率基数
        rope_scaling: 位置插值配置（当前不支持）
    
    Returns:
        RotaryEmbedding实例
    """
    assert rope_scaling is None  # 当前不支持位置插值
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

**RoPE 原理图解：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        旋转位置编码 (RoPE) 原理                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想：通过旋转将位置信息编码到向量中                                  │
│                                                                         │
│  二维旋转示例：                                                          │
│  ┌─────────────────────────────────────────┐                           │
│  │                                         │                           │
│  │      ╱│╲  旋转θ后   ╱│╲                 │                           │
│  │     ╱ │ ╲  ──────► ╱ │ ╲                │                           │
│  │    ╱  │  ╲        ╱  │  ╲               │                           │
│  │   ╱   │   ╲      ╱   │   ╲              │                           │
│  │  ╱────┼────╲    ╱────┼────╲             │                           │
│  │       x            x'                   │                           │
│  │                                         │                           │
│  │  [x1, x2] ──旋转θ──► [x1*cosθ-x2*sinθ,  │                           │
│  │                       x2*cosθ+x1*sinθ]  │                           │
│  └─────────────────────────────────────────┘                           │
│                                                                         │
│  高维扩展：将head_dim维分成head_dim/2对，每对独立旋转                      │
│  旋转频率：θ_i = position * base^(-2i/head_dim)                          │
│                                                                         │
│  为什么有效？                                                            │
│  1. 相对位置：dot(q_m, k_n) 只与 (m-n) 有关                               │
│  2. 长序列外推：可以处理超过训练长度的序列                                │
│  3. 与注意力天然结合：直接修改Q、K                                        │
│                                                                         │
│  频率可视化：                                                            │
│  dim 0: ████████████████████ 高频（旋转快）                               │
│  dim 1: ██████████████                                              │
│  dim 2: ████████                                                    │
│  ...                                                                │
│  dim d: ██ 低频（旋转慢）                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---


### 11. layers/attention.py - 注意力机制（含Triton Kernel）

```python
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块
import triton                         # Triton GPU编程语言
import triton.language as tl          # Triton语言接口

# FlashAttention高效实现
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context  # 全局上下文


# ==================== Triton Kernel ====================
@triton.jit
def store_kvcache_kernel(
    key_ptr,           # Key张量指针
    key_stride,        # Key的步长
    value_ptr,         # Value张量指针
    value_stride,      # Value的步长
    k_cache_ptr,       # K Cache指针
    v_cache_ptr,       # V Cache指针
    slot_mapping_ptr,  # 槽位映射指针
    D: tl.constexpr,   # 头维度（编译时常数）
):
    """
    Triton Kernel：将计算的KV值存储到Paged KV Cache
    
    为什么用Triton？
    1. 比PyTorch更高效（避免中间张量）
    2. 直接控制内存访问模式
    3. 融合多个操作
    
    每个线程处理一个token的一个头
    """
    # 获取当前线程的索引（对应一个token）
    idx = tl.program_id(0)
    
    # 加载该token在KV Cache中的存储位置
    slot = tl.load(slot_mapping_ptr + idx)
    
    # slot = -1 表示不需要存储（已缓存）
    if slot == -1:
        return
    
    # 计算Key/Value的内存偏移
    # key_offsets: 该token的所有头的所有维度
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 从输入加载Key和Value
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算KV Cache的存储位置
    cache_offsets = slot * D + tl.arange(0, D)
    
    # 存储到KV Cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor
):
    """
    将KV值存储到Paged KV Cache（Python接口）
    
    Args:
        key: 计算的Key [num_tokens, num_heads, head_dim]
        value: 计算的Value [num_tokens, num_kv_heads, head_dim]
        k_cache: K Cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: V Cache [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: 每个token的存储位置 [num_tokens]
    """
    # 获取维度信息
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 每个token的总维度
    
    # 验证内存布局（确保是连续的）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    
    # 启动Triton Kernel
    # grid=(N,): N个线程，每个处理一个token
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D
    )


# ==================== Attention 模块 ====================
class Attention(nn.Module):
    """
    注意力模块 - 支持Prefill和Decode阶段
    
    核心功能：
    1. 使用Triton Kernel将KV写入Paged Cache
    2. Prefill阶段：使用FlashAttention进行完整注意力计算
    3. Decode阶段：使用FlashAttention的KV Cache优化版本
    
    设计要点：
    - 通过Context判断当前阶段
    - 支持前缀缓存（block_tables不为None时）
    """
    
    def __init__(
        self,
        num_heads,         # 头数（当前GPU）
        head_dim,          # 每个头的大小
        scale,             # 缩放因子（1/sqrt(head_dim)）
        num_kv_heads,      # K,V头数（可能少于Q）
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        
        # 初始化时空Cache（会在allocate_kv_cache时设置）
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        注意力前向传播
        
        Args:
            q: Query [num_tokens, num_heads, head_dim]
            k: Key [num_tokens, num_kv_heads, head_dim]
            v: Value [num_tokens, num_kv_heads, head_dim]
        
        Returns:
            注意力输出 [num_tokens, num_heads, head_dim]
        """
        # 获取当前上下文
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # ==================== 存储KV到Cache ====================
        # 如果Cache已分配，将新计算的KV写入
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # ==================== Prefill 阶段 ====================
        if context.is_prefill:
            # 检查是否有前缀缓存
            if context.block_tables is not None:
                # 前缀缓存命中：使用完整的KV Cache
                k, v = k_cache, v_cache
            
            # 使用FlashAttention进行高效注意力计算
            # flash_attn_varlen_func 支持变长序列批处理
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,      # 最大Query长度
                cu_seqlens_q=context.cu_seqlens_q,      # Query累积长度
                max_seqlen_k=context.max_seqlen_k,      # 最大Key长度
                cu_seqlens_k=context.cu_seqlens_k,      # Key累积长度
                softmax_scale=self.scale,               # 缩放因子
                causal=True,                            # 因果掩码（只看前面）
                block_table=context.block_tables        # 块表（前缀缓存）
            )
        
        # ==================== Decode 阶段 ====================
        else:
            # Decode阶段：每个token只计算一个query
            # 使用flash_attn_with_kvcache优化
            # q需要增加序列维度：[batch, 1, num_heads, head_dim]
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),                         # Query
                k_cache,                                # 完整的K Cache
                v_cache,                                # 完整的V Cache
                cache_seqlens=context.context_lens,     # 每个序列的当前长度
                block_table=context.block_tables,       # 块表
                softmax_scale=self.scale,
                causal=True
            )
        
        return o
```

**注意力计算流程图：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      注意力计算流程                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Prefill 阶段                              │   │
│  │                                                                 │   │
│  │  Input: prompt tokens [t1, t2, t3, t4]                          │   │
│  │                      ↓                                          │   │
│  │  QKV Projection → Q, K, V                                       │   │
│  │                      ↓                                          │   │
│  │  store_kvcache(K, V) ───────────────► KV Cache                  │   │
│  │                      ↓                                          │   │
│  │  flash_attn_varlen_func(Q, K, V)                                │   │
│  │  - 一次性计算所有token的注意力                                   │   │
│  │  - 使用cu_seqlens处理变长序列                                    │   │
│  │                      ↓                                          │   │
│  │  Output: 所有位置的注意力结果                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Decode 阶段                               │   │
│  │                                                                 │   │
│  │  Input: last token [t_new]                                      │   │
│  │                      ↓                                          │   │
│  │  QKV Projection → Q_new, K_new, V_new                           │   │
│  │                      ↓                                          │   │
│  │  store_kvcache(K_new, V_new) ──────► KV Cache                   │   │
│  │                      ↓                                          │   │
│  │  flash_attn_with_kvcache(Q_new, KV_Cache)                       │   │
│  │  - 只计算新token的注意力                                        │   │
│  │  - 复用之前存储的KV Cache                                       │   │
│  │                      ↓                                          │   │
│  │  Output: 新token的注意力结果                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  关键区别：                                                              │
│  - Prefill: 计算所有token，并行度高，计算密集                            │
│  - Decode: 只计算1个token，内存带宽密集                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 12. layers/embed_head.py - 词嵌入和输出头

```python
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块
import torch.nn.functional as F       # 神经网络函数
import torch.distributed as dist      # 分布式

from nanovllm.utils.context import get_context  # 全局上下文


class VocabParallelEmbedding(nn.Module):
    """
    词表并行嵌入层 - 按词表维度切分
    
    当词表很大时（如100k+），嵌入矩阵会非常大
    词表并行将词表切分到多个GPU，每个GPU只存储部分词向量
    
    切分方式：
    - GPU 0: 词表 [0, vocab_size//2)
    - GPU 1: 词表 [vocab_size//2, vocab_size)
    
    前向传播：
    1. 将输入ID映射到本地词表索引
    2. 只有属于本地词表的token才会产生非零输出
    3. 通过all-reduce合并所有GPU的结果
    """
    
    def __init__(
        self,
        num_embeddings: int,           # 词表大小
        embedding_dim: int,            # 嵌入维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 确保词表可以被GPU数整除
        assert num_embeddings % self.tp_size == 0
        
        self.num_embeddings = num_embeddings
        # 每个GPU负责的词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        
        # 本地词表范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 创建本地嵌入矩阵
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, 
            embedding_dim
        ))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载词表切分的权重"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        词表并行嵌入前向
        
        Args:
            x: 输入token IDs [batch, seq_len]
        
        Returns:
            嵌入向量 [batch, seq_len, embedding_dim]
        """
        if self.tp_size > 1:
            # 创建掩码：标记哪些token属于本地词表
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            
            # 将全局ID映射到本地ID（不属于本地的设为0）
            x = mask * (x - self.vocab_start_idx)
        
        # 查找嵌入向量
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 应用掩码：不属于本地的token嵌入设为0
            y = mask.unsqueeze(1) * y
            
            # 通过all-reduce合并所有GPU的结果
            dist.all_reduce(y)
        
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型输出头
    
    继承自VocabParallelEmbedding，但前向逻辑不同：
    - Embedding: 输入token ID，输出嵌入向量
    - LMHead: 输入隐藏状态，输出每个词的对数几率
    
    注意：LMHead通常与输入嵌入共享权重（tie_word_embeddings）
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,            # 通常不使用偏置
    ):
        assert not bias  # 当前不支持偏置
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        语言模型输出头前向
        
        Args:
            x: 隐藏状态 [num_tokens, hidden_size]
        
        Returns:
            对数几率 [num_tokens, vocab_size]（只在rank 0）
        """
        context = get_context()
        
        # Prefill阶段：只取每个序列的最后一个位置
        if context.is_prefill:
            # cu_seqlens_q[1:] - 1 是每个序列最后一个token的索引
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # 线性变换：hidden -> vocab
        # 使用嵌入矩阵的转置作为输出权重（权重共享）
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # 多GPU时，需要gather所有GPU的结果
            if self.tp_rank == 0:
                # rank 0收集所有结果
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            else:
                all_logits = None
            
            dist.gather(logits, all_logits, 0)
            
            # rank 0拼接所有结果
            if self.tp_rank == 0:
                logits = torch.cat(all_logits, -1)
        
        return logits
```

**词表并行图解：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        词表并行 (Vocab Parallelism)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题：词表100k，维度4096，嵌入矩阵大小 = 100k * 4096 * 2B ≈ 800MB       │
│                                                                         │
│  解决方案：将词表切分到2个GPU                                            │
│                                                                         │
│  GPU 0:                    GPU 1:                                       │
│  ┌─────────────────┐      ┌─────────────────┐                          │
│  │  词表 [0-50k)   │      │  词表 [50k-100k)│                          │
│  │  权重 W0        │      │  权重 W1        │                          │
│  │  [50k, 4096]    │      │  [50k, 4096]    │                          │
│  └────────┬────────┘      └────────┬────────┘                          │
│           │                        │                                    │
│           │  输入: token IDs       │                                    │
│           │  [12, 50001, 34, 99999]│                                    │
│           │                        │                                    │
│     ┌─────┴─────┐            ┌─────┴─────┐                             │
│     ▼           ▼            ▼           ▼                             │
│  mask: [1,0,1,0]          mask: [0,1,0,1]                              │
│  local_id: [12,0,34,0]    local_id: [0,1,0,49999]                      │
│     │           │            │           │                             │
│     ▼           ▼            ▼           ▼                             │
│  embed: E0    zeros       zeros       E1                               │
│     │           │            │           │                             │
│     └─────┬─────┘            └─────┬─────┘                             │
│           │                        │                                    │
│           │    all_reduce(E0+E1)   │                                    │
│           └──────────┬─────────────┘                                    │
│                      ▼                                                  │
│                   完整嵌入                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 13. layers/sampler.py - 采样器

```python
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块


class Sampler(nn.Module):
    """
    采样器 - 从模型输出的logits中采样下一个token
    
    使用温度采样（Temperature Sampling）：
    1. logits除以temperature（控制随机性）
    2. softmax转换为概率
    3. 使用指数分布进行采样
    
    采样公式：
    token = argmax(softmax(logits / T) / exp(-U))
    其中 U ~ Exponential(1)，T = temperature
    
    这等价于从softmax(logits / T)中采样
    """
    
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        采样下一个token
        
        Args:
            logits: 模型输出的对数几率 [batch, vocab_size]
            temperatures: 每个序列的温度 [batch]
        
        Returns:
            采样的token IDs [batch]
        """
        # 转为float32提高数值稳定性
        logits = logits.float()
        
        # 应用温度：logits / temperature
        # temperature > 1: 更随机
        # temperature < 1: 更确定
        logits.div_(temperatures.unsqueeze(dim=1))
        
        # softmax转换为概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # Gumbel采样技巧：
        # argmax(probs / exp(-U)) 其中 U ~ Exponential(1)
        # 等价于从probs中采样
        # 
        # torch.empty_like(probs).exponential_(1) 生成指数分布随机数
        # clamp_min_(1e-10) 避免除零
        # probs / random 然后取argmax
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        
        return sample_tokens
```

**采样方法对比：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        采样方法对比                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  贪婪解码 (Greedy Decoding)                                             │
│  token = argmax(logits)                                                 │
│  特点：确定性，总是选概率最高的                                           │
│  缺点：输出单一，缺乏多样性                                               │
│                                                                         │
│  温度采样 (Temperature Sampling)                                        │
│  probs = softmax(logits / T)                                            │
│  token = sample(probs)                                                  │
│                                                                         │
│  T = 0.3:  ████████████████████  保守，接近贪婪                          │
│  T = 0.7:  ██████████████        平衡                                    │
│  T = 1.0:  ████████              标准随机                                │
│  T = 1.5:  ████                  更随机                                  │
│                                                                         │
│  Top-k 采样                                                             │
│  只从概率最高的k个token中采样                                             │
│  避免选中极低概率的token                                                  │
│                                                                         │
│  Top-p (Nucleus) 采样                                                   │
│  从累积概率达到p的最小集合中采样                                          │
│  动态调整候选集大小                                                       │
│                                                                         │
│  Nano-vLLM使用：温度采样（最常用）                                        │
│                                                                         │
│  Gumbel采样技巧：                                                         │
│  不直接采样，而是：argmax(probs / exp(-U))                               │
│  这样可以用argmax实现采样效果，更高效                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---


### 14. models/qwen3.py - Qwen3 模型实现

```python
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块
import torch.distributed as dist      # 分布式
from transformers import Qwen3Config  # HuggingFace配置

# 导入所有层
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 注意力层
    
    结构：
    1. QKV投影（合并为一个线性层）
    2. 可选的Q/K归一化
    3. 旋转位置编码
    4. 注意力计算
    5. 输出投影
    
    支持：
    - 张量并行
    - GQA (Grouped Query Attention)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 当前GPU的头数
        
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        
        # QKV大小
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # 缩放因子
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        
        # QKV投影（列并行）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        
        # 输出投影（行并行）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
        # 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # 注意力计算模块
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        
        # 可选的Q/K归一化（无bias时使用）
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,           # 位置索引
        hidden_states: torch.Tensor,       # 隐藏状态
    ) -> torch.Tensor:
        """
        注意力前向传播
        
        Args:
            positions: 位置索引 [num_tokens]
            hidden_states: 隐藏状态 [num_tokens, hidden_size]
        
        Returns:
            注意力输出 [num_tokens, hidden_size]
        """
        # QKV投影
        qkv = self.qkv_proj(hidden_states)
        
        # 切分为Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # reshape为多头格式
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # 可选的Q/K归一化
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        
        # 注意力计算
        o = self.attn(q, k, v)
        
        # 输出投影
        output = self.o_proj(o.flatten(1, -1))
        
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 MLP层（前馈网络）
    
    结构：
    1. gate_proj 和 up_proj 合并
    2. SwiGLU激活
    3. down_proj
    
    公式：down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        
        # gate_proj 和 up_proj 合并（列并行）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,   # 两个输出，每个大小为intermediate_size
            bias=False,
        )
        
        # down_proj（行并行）
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        
        # 激活函数
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """MLP前向传播"""
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 解码器层
    
    结构（Pre-LN）：
    1. RMSNorm
    2. Self-Attention
    3. RMSNorm
    4. MLP
    
    使用残差连接
    """
    
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        # 两个LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,     # 残差连接
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        解码器层前向传播
        
        Args:
            positions: 位置索引
            hidden_states: 隐藏状态
            residual: 残差（第一层为None）
        
        Returns:
            (新的隐藏状态, 更新后的残差)
        """
        # 第一层：初始化残差
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Self-Attention
        hidden_states = self.self_attn(positions, hidden_states)
        
        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 基础模型（不含输出头）
    
    结构：
    1. 词嵌入
    2. N个解码器层
    3. 最终LayerNorm
    """
    
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        # 词嵌入（词表并行）
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, 
            config.hidden_size
        )
        
        # 解码器层列表
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # 最终LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,           # 输入token IDs
        positions: torch.Tensor,           # 位置索引
    ) -> torch.Tensor:
        """
        模型前向传播
        
        Args:
            input_ids: 输入token IDs [num_tokens]
            positions: 位置索引 [num_tokens]
        
        Returns:
            最终隐藏状态 [num_tokens, hidden_size]
        """
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 残差连接（第一层为None）
        residual = None
        
        # 逐层前向传播
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 最终LayerNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型（完整模型）
    
    包含：
    1. 基础模型
    2. 语言模型输出头
    
    packed_modules_mapping: 权重映射
    - 将HuggingFace的分离权重映射到合并的权重
    """
    
    # 权重映射：HuggingFace名称 -> (本地名称, shard_id)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 权重共享（tie_word_embeddings）
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，返回隐藏状态"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算输出logits"""
        return self.lm_head(hidden_states)
```

**Qwen3 模型结构图：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Qwen3 模型结构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: token IDs                                                       │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Embedding (VocabParallel)                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 0                                                        │   │
│  │  ┌─────────────┐    ┌─────────────┐                            │   │
│  │  │ RMSNorm     │───►│ Attention   │                            │   │
│  │  │             │    │ - QKV Proj  │                            │   │
│  │  └─────────────┘    │ - RoPE      │                            │   │
│  │         │           │ - FlashAttn │                            │   │
│  │         │           │ - Out Proj  │                            │   │
│  │         │           └─────────────┘                            │   │
│  │         │                  │                                    │   │
│  │         │           ┌──────┘                                   │   │
│  │         │           ▼                                          │   │
│  │  ┌─────────────┐    ┌─────────────┐                            │   │
│  │  │ RMSNorm     │───►│ MLP         │                            │   │
│  │  │             │    │ - GateUp    │                            │   │
│  │  └─────────────┘    │ - SwiGLU    │                            │   │
│  │                     │ - Down      │                            │   │
│  │                     └─────────────┘                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│      ... (重复 N 层)                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Final RMSNorm                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LM Head (Parallel)                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  Output: logits                                                         │
│                                                                         │
│  典型配置 (Qwen3-0.6B):                                                  │
│  - 层数: 28                                                             │
│  - 隐藏维度: 1024                                                       │
│  - 注意力头: 16                                                         │
│  - KV头: 8 (GQA)                                                        │
│  - 中间维度: 2816                                                       │
│  - 词表: 151936                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 15. utils/loader.py - 模型权重加载

```python
import os                             # 操作系统接口
from glob import glob                 # 文件路径匹配
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块
from safetensors import safe_open     # SafeTensors格式（安全的tensor序列化）


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认权重加载函数 - 直接复制
    
    Args:
        param: 目标参数
        loaded_weight: 加载的权重
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重
    
    支持：
    1. SafeTensors格式（推荐，安全且高效）
    2. 合并权重自动拆分（如qkv_proj拆分为q_proj,k_proj,v_proj）
    3. 张量并行权重加载
    
    Args:
        model: 要加载权重的模型
        path: 权重文件目录
    """
    # 获取模型的权重映射（如果有）
    # 例如 Qwen3ForCausalLM.packed_modules_mapping
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 遍历所有.safetensors文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重
            for weight_name in f.keys():
                # 检查是否需要映射
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 需要映射的权重
                        # 例如：q_proj -> (qkv_proj, "q")
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        
                        # 使用自定义的weight_loader
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 不需要映射的权重，直接加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

**权重加载流程：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        权重加载流程                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HuggingFace权重文件:                                                   │
│  model-00001-of-00002.safetensors                                       │
│  ├── q_proj.weight                                                      │
│  ├── k_proj.weight                                                      │
│  ├── v_proj.weight                                                      │
│  ├── gate_proj.weight                                                   │
│  ├── up_proj.weight                                                     │
│  └── ...                                                                │
│                                                                         │
│  加载过程：                                                              │
│                                                                         │
│  q_proj.weight ──┐                                                      │
│                  ├──► packed_modules_mapping ──► qkv_proj (shard="q")  │
│  k_proj.weight ──┤                      │                               │
│                  ├──► packed_modules_mapping ──► qkv_proj (shard="k")  │
│  v_proj.weight ──┘                      │                               │
│                                         │                               │
│  gate_proj.weight ──► packed_modules_mapping ──► gate_up_proj (id=0)   │
│  up_proj.weight ────► packed_modules_mapping ──► gate_up_proj (id=1)   │
│                                                                         │
│  张量并行加载：                                                          │
│  qkv_proj.weight = [Q_part, K_part, V_part]                            │
│       │                                                                 │
│       ├──► GPU 0: Q_part[0:Q//2], K_part[0:K//2], V_part[0:V//2]       │
│       └──► GPU 1: Q_part[Q//2:], K_part[K//2:], V_part[V//2:]          │
│                                                                         │
│  weight_loader 的作用：                                                  │
│  1. 切分权重到当前GPU                                                    │
│  2. 处理合并权重的特定部分                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 16. engine/model_runner.py - 模型运行器

```python
import pickle                         # 序列化
import torch                          # PyTorch
import torch.distributed as dist      # 分布式
from multiprocessing.synchronize import Event      # 进程同步
from multiprocessing.shared_memory import SharedMemory  # 共享内存

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型运行器 - 管理模型的加载、KV Cache分配和推理执行
    
    职责：
    1. 模型加载和初始化
    2. KV Cache预分配
    3. Warmup（预热）
    4. CUDA Graph捕获
    5. Prefill/Decode执行
    
    多GPU支持：
    - rank 0: 主进程，执行调度
    - rank > 0: 工作进程，通过共享内存接收指令
    """
    
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器
        
        Args:
            config: 全局配置
            rank: 当前GPU的rank
            event: 进程同步事件（多GPU时使用）
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        
        # ==================== 初始化分布式 ====================
        dist.init_process_group(
            "nccl",                         # NCCL后端（NVIDIA GPU）
            "tcp://localhost:2333",         # 主节点地址
            world_size=self.world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        
        # ==================== 设置默认数据类型和设备 ====================
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # ==================== 创建模型 ====================
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        
        # ==================== 创建采样器 ====================
        self.sampler = Sampler()
        
        # ==================== Warmup和分配 ====================
        self.warmup_model()
        self.allocate_kv_cache()
        
        # ==================== CUDA Graph ====================
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        
        # ==================== 多GPU设置 ====================
        if self.world_size > 1:
            if rank == 0:
                # rank 0创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 其他rank等待并连接共享内存
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 工作进程进入事件循环
                self.loop()

    def exit(self):
        """清理资源"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """工作进程的事件循环"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """从共享内存读取指令"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待rank 0的信号
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """向共享内存写入指令"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()  # 通知所有工作进程

    def call(self, method_name, *args):
        """调用方法（多GPU时通过共享内存同步）"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        模型预热
        
        目的：
        1. 触发CUDA kernel编译
        2. 测量峰值显存使用
        3. 确保后续推理稳定
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 计算warmup的序列数
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        
        # 创建虚拟序列
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        
        # 执行一次推理
        self.run(seqs, True)
        
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配KV Cache
        
        根据可用显存计算可分配的块数
        """
        config = self.config
        hf_config = config.hf_config
        
        # 获取显存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算每个KV块的内存大小
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        
        # KV Cache大小 = 2(K+V) * 层数 * 块数 * 块大小 * KV头数 * 头维度 * 数据类型大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # 计算可分配的块数
        # 可用显存 = 总显存 * 使用率 - 已用 - 峰值 + 当前
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        
        assert config.num_kvcache_blocks > 0
        
        # 创建KV Cache张量
        # 形状: [2(K/V), 层数, 块数, 块大小, KV头数, 头维度]
        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
            self.block_size, num_kv_heads, head_dim
        )
        
        # 将KV Cache分配给每个注意力层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """准备块表张量"""
        max_len = max(len(seq.block_table) for seq in seqs)
        # 填充到相同长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备Prefill阶段的输入
        
        处理变长序列批处理，生成：
        - input_ids: 所有token的ID
        - positions: 每个token的位置
        - cu_seqlens: 累积序列长度（用于FlashAttention）
        - slot_mapping: 每个token在KV Cache中的位置
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            
            # 只取未缓存的token作为Query
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens  # Query长度
            seqlen_k = seqlen                           # Key长度（包含缓存的）
            
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:  # warmup时
                continue
            
            # 计算slot_mapping（未缓存的token）
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        
        # 如果有前缀缓存，准备块表
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """准备Decode阶段的输入"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 新token的存储位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        运行模型
        
        根据条件选择执行方式：
        1. Prefill: 直接执行（计算密集，CUDA Graph收益小）
        2. Decode + eager: 直接执行
        3. Decode + CUDA Graph: 使用捕获的graph
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接执行
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用CUDA Graph
            bs = input_ids.size(0)
            context = get_context()
            
            # 找到合适的graph大小
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 填充输入
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放graph
            graph.replay()
            
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行一次推理迭代
        
        Args:
            seqs: 要处理的序列
            is_prefill: 是否是prefill阶段
        
        Returns:
            生成的token IDs
        """
        # 准备输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # 运行模型
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # 采样（只在rank 0执行）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # 重置上下文
        reset_context()
        
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获CUDA Graph
        
        CUDA Graph可以：
        1. 消除kernel launch开销
        2. 优化内存访问模式
        3. 提高小batch的decode效率
        
        捕获过程：
        1. warmup运行（分配内存，确定执行路径）
        2. 开始捕获
        3. 再次运行（记录操作）
        4. 结束捕获
        """
        config = self.config
        hf_config = config.hf_config
        
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 创建graph输入缓冲区
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 批大小列表（从大到小捕获，可以复用graph pool）
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            
            # 设置上下文
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # Warmup运行
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 开始捕获
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 保存graph pool供后续复用
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        
        # 保存graph变量
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
```

---

### 17. engine/llm_engine.py - LLM 引擎主类

```python
import atexit                         # 程序退出时清理
from dataclasses import fields        # 获取dataclass字段
from time import perf_counter         # 性能计时
from tqdm.auto import tqdm            # 进度条
from transformers import AutoTokenizer  # 分词器
import torch.multiprocessing as mp    # 多进程

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM引擎 - 用户接口层
    
    职责：
    1. 管理配置和初始化
    2. 协调调度器和模型运行器
    3. 提供用户友好的生成接口
    
    多GPU支持：
    - 主进程：调度 + rank 0计算
    - 工作进程：rank 1+计算（通过共享内存通信）
    """
    
    def __init__(self, model, **kwargs):
        """
        初始化LLM引擎
        
        Args:
            model: 模型路径
            **kwargs: 配置参数
        """
        # 从kwargs中提取Config相关的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # 创建工作进程（多GPU时）
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")  # spawn模式避免CUDA fork问题
        
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # 启动工作进程
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 主进程创建rank 0的ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # 创建调度器
        self.scheduler = Scheduler(config)
        
        # 注册退出清理
        atexit.register(self.exit)

    def exit(self):
        """清理资源"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加请求
        
        Args:
            prompt: 提示文本或token IDs
            sampling_params: 采样参数
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一次调度-推理迭代
        
        Returns:
            (完成的输出, token数量)
            token数量 > 0: prefill阶段
            token数量 < 0: decode阶段
        """
        # 调度请求
        seqs, is_prefill = self.scheduler.schedule()
        
        # 执行推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 后处理
        self.scheduler.postprocess(seqs, token_ids)
        
        # 收集完成的输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 计算token数量（用于吞吐量统计）
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已完成"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        生成文本（主接口）
        
        Args:
            prompts: 提示列表（文本或token IDs）
            sampling_params: 采样参数（单个或列表）
            use_tqdm: 是否显示进度条
        
        Returns:
            生成的文本列表
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # 统一sampling_params为列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 添加所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # 主循环
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集完成的输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # 按seq_id排序输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        
        # 解码为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} 
                   for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
        
        return outputs
```

**引擎主循环图解：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM 引擎主循环                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  generate()                                                             │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. 添加所有请求到等待队列                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. 主循环 while not is_finished():                              │   │
│  │                                                                 │   │
│  │     ┌──────────────────────────────────────────────────────┐   │   │
│  │     │  scheduler.schedule()                                  │   │   │
│  │     │  - 优先调度waiting队列（prefill）                      │   │   │
│  │     │  - 然后调度running队列（decode）                       │   │   │
│  │     │  - 资源不足时抢占                                      │   │   │
│  │     └────────────────────┬─────────────────────────────────┘   │   │
│  │                          │                                         │   │
│  │                          ▼                                         │   │
│  │     ┌──────────────────────────────────────────────────────┐   │   │
│  │     │  model_runner.call("run", seqs, is_prefill)          │   │   │
│  │     │                                                        │   │   │
│  │     │  if prefill:                                           │   │   │
│  │     │    - prepare_prefill()                                 │   │   │
│  │     │    - flash_attn_varlen_func()                          │   │   │
│  │     │  else:                                                 │   │   │
│  │     │    - prepare_decode()                                  │   │   │
│  │     │    - flash_attn_with_kvcache()                         │   │   │
│  │     │                                                        │   │   │
│  │     │  - sampler.sample()                                    │   │   │
│  │     └────────────────────┬─────────────────────────────────┘   │   │
│  │                          │                                         │   │
│  │                          ▼                                         │   │
│  │     ┌──────────────────────────────────────────────────────┐   │   │
│  │     │  scheduler.postprocess()                             │   │   │
│  │     │  - 将token添加到序列                                 │   │   │
│  │     │  - 检查是否完成                                      │   │   │
│  │     │  - 释放完成的序列的块                                │   │   │
│  │     └──────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  3. 解码token为文本并返回                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 18. llm.py - 最终接口

```python
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    LLM 类 - 用户直接使用的接口
    
    简单继承自 LLMEngine，保持与 vLLM API 兼容
    
    使用示例：
        from nanovllm import LLM, SamplingParams
        
        llm = LLM("/path/to/model", tensor_parallel_size=2)
        sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
        outputs = llm.generate(["Hello, world!"], sampling_params)
        print(outputs[0]["text"])
    """
    pass
```

---


## 🎯 核心原理总结

### 1. PagedAttention 核心原理

```
问题：传统LLM推理的KV Cache管理
┌─────────────────────────────────────────────────────────────────────────┐
│  传统方式：连续内存分配                                                  │
│                                                                         │
│  请求A: [========]  请求B: [========]  请求C: [========]               │
│         ↑预分配max_len                                                  │
│                                                                         │
│  问题：                                                                  │
│  1. 内部碎片：实际生成长度 < 预分配长度                                   │
│  2. 外部碎片：释放后产生不连续的小块                                      │
│  3. 无法共享：相同prompt的KV Cache重复存储                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  PagedAttention：分页管理                                                │
│                                                                         │
│  物理内存（固定大小的块）：                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ [ ][ ][A][A][B][A][C][B][ ][ ][ ][ ][ ][ ][ ]...               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  请求A块表: [2, 3, 5]  → 物理块 2, 3, 5                                 │
│  请求B块表: [4, 7]     → 物理块 4, 7                                    │
│  请求C块表: [6]        → 物理块 6                                       │
│                                                                         │
│  优势：                                                                  │
│  1. 无内部碎片：按需分配块                                                │
│  2. 无外部碎片：块大小固定，可复用                                        │
│  3. 支持共享：相同块可以多个请求共享                                      │
│  4. 前缀缓存：通过哈希快速匹配前缀                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Continuous Batching 连续批处理

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        连续批处理 vs 静态批处理                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  静态批处理：                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Batch 1: [Req1][Req2][Req3]  ──────► 全部完成后才处理下一批    │   │
│  │  等待: [Req4][Req5][Req6]...                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  问题：短请求需要等长请求完成，GPU空闲时间多                              │
│                                                                         │
│  连续批处理：                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: [Req1][Req2][Req3]  prefill                            │   │
│  │  Step 2: [Req1][Req2][Req3]  decode (Req3完成)                   │   │
│  │  Step 3: [Req1][Req2][Req4]  decode (新请求Req4加入)             │   │
│  │  Step 4: [Req2][Req4][Req5]  decode (Req1完成，Req5加入)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  优势：请求完成后立即释放资源给新请求，GPU利用率高                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. Prefill vs Decode

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Prefill vs Decode 对比                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Prefill（预填充）                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Prompt: "The capital of France is"                             │   │
│  │  Tokens: [The, capital, of, France, is]                         │   │
│  │                     ↓                                           │   │
│  │  一次性处理所有token，计算它们的KV Cache                         │   │
│  │  使用并行计算，计算密集                                           │   │
│  │  输出：最后一个token的logits                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  特点：                                                                  │
│  - 计算量：O(prompt_len²)                                              │
│  - 内存带宽：高（需要读取所有权重）                                     │
│  - 优化：FlashAttention减少内存访问                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Decode（解码）                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  已生成: "The capital of France is"                             │   │
│  │  下一步: 预测下一个token                                          │   │
│  │                     ↓                                           │   │
│  │  每次只处理1个token，复用之前的KV Cache                          │   │
│  │  内存带宽密集（需要读取所有层的KV Cache）                         │   │
│  │  输出：新token                                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  特点：                                                                  │
│  - 计算量：O(context_len) 每步                                         │
│  - 内存带宽：极高（KV Cache读取成为瓶颈）                               │
│  - 优化：PagedAttention、CUDA Graph                                     │
│                                                                         │
│  为什么decode慢？                                                        │
│  - 每次只处理1个token，无法利用矩阵乘的并行性                            │
│  - 需要读取所有层的KV Cache（内存带宽瓶颈）                              │
│  - kernel launch开销相对较大                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4. 张量并行 (Tensor Parallelism)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        张量并行原理                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题：模型太大，单个GPU放不下                                           │
│  例如：70B模型，FP16需要140GB显存                                        │
│                                                                         │
│  解决方案：将权重切分到多个GPU                                           │
│                                                                         │
│  层1（Column Parallel）：                                                │
│  ┌─────────────────┐                                                    │
│  │  输入 X         │                                                    │
│  │  [batch, 4096]  │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│     ┌─────┴─────┐                                                       │
│     ▼           ▼                                                       │
│  ┌───────┐  ┌───────┐                                                   │
│  │GPU 0  │  │GPU 1  │                                                   │
│  │W1     │  │W2     │  W = [W1; W2] 行拼接                              │
│  │[4096, │  │[4096, │                                                   │
│  │ 2048] │  │ 2048] │                                                   │
│  └───┬───┘  └───┬───┘                                                   │
│      │          │                                                       │
│      ▼          ▼                                                       │
│   Y1=X@W1    Y2=X@W2                                                    │
│      │          │                                                       │
│      └────┬─────┘                                                       │
│           ▼                                                             │
│       Y = [Y1, Y2]  输出维度翻倍                                         │
│                                                                         │
│  层2（Row Parallel）：                                                   │
│  ┌─────────────────┐                                                    │
│  │  输入 Y         │                                                    │
│  │  [batch, 4096]  │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│     ┌─────┴─────┐                                                       │
│     ▼           ▼                                                       │
│  ┌───────┐  ┌───────┐                                                   │
│  │GPU 0  │  │GPU 1  │                                                   │
│  │Y1     │  │Y2     │  Y = [Y1, Y2] 列拼接                              │
│  │W1     │  │W2     │  W = [W1, W2] 列拼接                              │
│  │[2048, │  │[2048, │                                                   │
│  │ 4096] │  │ 4096] │                                                   │
│  └───┬───┘  └───┬───┘                                                   │
│      │          │                                                       │
│      ▼          ▼                                                       │
│   Z1=Y1@W1   Z2=Y2@W2                                                   │
│      │          │                                                       │
│      └────┬─────┘                                                       │
│           ▼                                                             │
│       Z = Z1 + Z2  (all-reduce)                                         │
│                                                                         │
│  通信量分析：                                                            │
│  - Column Parallel: 0 通信                                               │
│  - Row Parallel: 1次 all-reduce（数据量 = batch * hidden_size）         │
│  每层需要1次 all-reduce                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5. CUDA Graph 优化

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CUDA Graph 原理                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题：小batch decode的kernel launch开销                                 │
│                                                                         │
│  传统执行：                                                              │
│  CPU:  launch kernel1 ──► launch kernel2 ──► launch kernel3 ...        │
│  GPU:  [kernel1]         [kernel2]         [kernel3]                    │
│           ↑ 空闲时间 ↑ 空闲时间 ↑                                       │
│  每次launch都有CPU-GPU同步开销                                           │
│                                                                         │
│  CUDA Graph：                                                            │
│  1. 录制阶段（一次）：                                                    │
│     CPU:  begin capture ──► run kernels ──► end capture                │
│     GPU:  [k1][k2][k3][k4][k5]...  记录所有操作                         │
│                                                                         │
│  2. 重放阶段（多次）：                                                    │
│     CPU:  graph.replay()  ← 一次调用重放所有kernel                      │
│     GPU:  [k1][k2][k3][k4][k5]...  连续执行                             │
│                                                                         │
│  优势：                                                                  │
│  - 消除CPU launch开销                                                    │
│  - kernel之间无空闲，GPU利用率100%                                        │
│  - 可以优化内存访问模式                                                   │
│                                                                         │
│  限制：                                                                  │
│  - 输入输出大小必须固定                                                   │
│  - 不支持动态控制流                                                       │
│  - 需要为不同batch size分别捕获                                          │
│                                                                         │
│  Nano-vLLM实现：                                                         │
│  - 捕获batch size: 1, 2, 4, 8, 16, 32, ..., 512                          │
│  - 运行时选择最接近的graph                                                │
│  - 超过512使用eager模式                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ❓ 面试核心问题与解答

### 基础概念

**Q1: 什么是KV Cache？为什么需要它？**

```
A: KV Cache是Transformer推理中的关键优化。

原理：
- 在decode阶段，每个新token需要与之前所有token计算注意力
- 如果不缓存，每次都要重新计算之前token的Key和Value
- KV Cache存储了每个token的K和V，避免重复计算

计算：
- 无KV Cache: 生成N个token需要 O(N³) 计算
- 有KV Cache: 只需要 O(N²) 计算

内存开销：
- 每层: 2(K+V) * seq_len * num_kv_heads * head_dim * sizeof(dtype)
- 例如：32层，seq_len=4096，GQA 8头，head_dim=128，fp16
  需要: 32 * 2 * 4096 * 8 * 128 * 2B = 512MB
```

**Q2: PagedAttention解决了什么问题？**

```
A: PagedAttention解决了传统KV Cache管理的三个问题：

1. 内部碎片：
   - 传统：预分配max_len，实际使用可能很少
   - PagedAttention：按需分配固定大小的块

2. 外部碎片：
   - 传统：释放后产生不连续的小块
   - PagedAttention：块大小固定，可以任意复用

3. 无法共享：
   - 传统：每个请求独立的连续内存
   - PagedAttention：相同内容的块可以共享（copy-on-write）

核心设计：
- 物理块：固定大小（如256 tokens）
- 块表：逻辑块到物理块的映射
- 引用计数：支持块共享
- 哈希缓存：前缀匹配加速
```

**Q3: 什么是Continuous Batching？**

```
A: Continuous Batching（连续批处理）是一种动态批处理策略。

传统静态批处理：
- 一批请求一起开始，一起结束
- 短请求需要等长请求完成
- GPU利用率低

连续批处理：
- 每个iteration重新调度
- 请求完成后立即释放资源
- 新请求可以立即加入

实现要点：
1. 区分prefill和decode阶段
2. 资源不足时抢占（preempt）运行中的请求
3. 被抢占的请求放回等待队列头部

优势：
- 提高GPU利用率
- 降低平均延迟
- 支持高并发
```

**Q4: FlashAttention的原理是什么？**

```
A: FlashAttention是一种IO感知的注意力算法。

传统注意力的内存瓶颈：
- 需要存储N×N的注意力矩阵
- 内存访问量是计算量的数倍

FlashAttention的核心思想：
1. 分块计算：将Q、K、V分成小块
2. Softmax稳定化：在线计算softmax的归一化因子
3. 重计算：反向传播时重新计算注意力，不存储中间结果

算法步骤：
for each block of Q:
    for each block of K, V:
        1. 计算 S = Q @ K^T
        2. 计算 P = softmax(S)
        3. 累加 O += P @ V

优势：
- 减少HBM（高带宽内存）访问
- 计算和内存访问平衡
- 支持更长的序列
```

**Q5: 张量并行和流水线并行的区别？**

```
A: 两种都是模型并行策略，但切分维度不同。

张量并行（Tensor Parallelism）：
- 切分单个层的权重
- 例如：将线性层的输出维度切分到2个GPU
- 通信：每层需要1-2次all-reduce
- 适用：单节点多GPU，延迟敏感

流水线并行（Pipeline Parallelism）：
- 切分不同层到不同GPU
- 例如：GPU 0负责层0-3，GPU 1负责层4-7
- 通信：只需要传递激活值
- 适用：多节点，吞吐敏感

Nano-vLLM只实现了张量并行，因为：
1. 代码简洁（约1200行）
2. 单节点场景最常见
3. 张量并行对延迟优化更好
```

**Q6: RoPE（旋转位置编码）的原理？**

```
A: RoPE通过旋转矩阵将位置信息编码到Q和K中。

核心思想：
- 对Q和K的每一维配对进行旋转
- 旋转角度 = position × frequency

数学公式：
对于二维向量 [x1, x2]，旋转θ：
[x1']   [cosθ  -sinθ] [x1]
[x2'] = [sinθ   cosθ] [x2]

扩展到高维：
- 将head_dim维分成head_dim/2对
- 每对应用不同频率的旋转
- 频率：θ_i = base^(-2i/head_dim)

优势：
1. 相对位置：dot(q_m, k_n) 只与(m-n)有关
2. 长序列外推：可以处理超过训练长度的序列
3. 与注意力天然结合
```

**Q7: 为什么decode阶段比prefill慢？**

```
A: Decode阶段慢的原因：

1. 计算并行度低：
   - Prefill：一次处理多个token，矩阵乘可以并行
   - Decode：一次只处理1个token，无法利用矩阵乘优化

2. 内存带宽瓶颈：
   - Prefill：计算密集，主要时间花在矩阵乘
   - Decode：内存带宽密集，需要读取所有层的KV Cache

3. Kernel launch开销：
   - Decode：每个token需要launch多个kernel
   - 小batch时，launch开销占比大

优化方法：
1. PagedAttention：优化KV Cache访问
2. CUDA Graph：减少kernel launch开销
3. 量化：减少内存带宽需求
4. 推测解码：用draft模型加速

数据对比（典型值）：
- Prefill吞吐量：1000-10000 tokens/s
- Decode吞吐量：50-200 tokens/s
```

**Q8: CUDA Graph在LLM推理中的作用？**

```
A: CUDA Graph优化小batch decode的性能。

原理：
1. 录制：记录一次完整的kernel执行序列
2. 重放：后续直接重放录制的序列

优势：
- 消除CPU launch开销
- 减少GPU空闲时间
- 可以优化内存访问

限制：
- 输入输出大小必须固定
- 不支持动态控制流
- 需要为不同batch size分别捕获

Nano-vLLM的实现：
- 捕获batch size: 1, 2, 4, 8, 16, 32, ..., 512
- 运行时选择最接近的graph
- 超过512使用eager模式

效果：
- 小batch decode吞吐量提升20-50%
- 对prefill效果不明显（计算密集）
```

---

## 📖 学习建议

### 1. 代码阅读顺序

```
第一阶段（建立认知）：
├── sampling_params.py    # 5分钟
├── config.py             # 10分钟
└── utils/context.py      # 10分钟

第二阶段（核心数据结构）：
├── engine/sequence.py    # 20分钟
└── engine/block_manager.py  # 30分钟 ⭐重点

第三阶段（调度系统）：
└── engine/scheduler.py   # 30分钟 ⭐重点

第四阶段（模型层）：
├── layers/linear.py      # 30分钟 ⭐重点（张量并行）
├── layers/layernorm.py   # 15分钟
├── layers/activation.py  # 10分钟
├── layers/rotary_embedding.py  # 20分钟
├── layers/attention.py   # 30分钟 ⭐重点
├── layers/embed_head.py  # 20分钟
└── layers/sampler.py     # 10分钟

第五阶段（模型架构）：
├── models/qwen3.py       # 30分钟
└── utils/loader.py       # 15分钟

第六阶段（引擎核心）：
├── engine/model_runner.py  # 40分钟 ⭐重点
├── engine/llm_engine.py    # 20分钟
└── llm.py                # 5分钟
```

### 2. 动手实践建议

1. **单步调试**：在关键函数打断点，观察数据流动
2. **修改参数**：改变block_size、max_num_seqs等，观察影响
3. **添加日志**：在调度、内存分配处打印状态
4. **性能分析**：使用nvprof分析kernel执行时间

### 3. 深入学习方向

1. **FlashAttention**：阅读原始论文和Triton实现
2. **Triton编程**：学习GPU kernel开发
3. **量化推理**：INT8/INT4量化实现
4. **推测解码**：Draft-then-verify机制
5. **多模态**：扩展到视觉-语言模型

---

## 🔗 参考资料

1. **vLLM论文**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
2. **FlashAttention论文**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
3. **RoPE论文**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
4. **Tensor Parallelism**: [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
5. **Nano-vLLM GitHub**: https://github.com/GeeeekExplorer/nano-vllm

---

*本文档基于 Nano-vLLM 项目（约1200行代码）编写，是学习VLLM推理引擎的完整指南。*
