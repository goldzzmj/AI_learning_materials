# SGLang-Diffusion 快速执行脚本

本目录包含 SGLang-Diffusion 框架的快速执行脚本，帮助您快速上手图像生成和编辑任务。

## 脚本列表

| 脚本 | 描述 | 用途 |
|------|------|------|
| `install_sglang_diffusion.sh` | 安装脚本 | 自动安装SGLang-Diffusion及其依赖 |
| `generate_qwen_image.sh` | 文生图脚本 | 使用Qwen-Image生成图像 |
| `edit_qwen_image.sh` | 图像编辑脚本 | 使用Qwen-Image-Edit编辑图像 |
| `python_api_example.py` | Python API示例 | 展示Python API的各种用法 |

## 快速开始

### 1. 安装 SGLang-Diffusion

```bash
# 基础安装
./install_sglang_diffusion.sh

# 从源码安装（开发版本）
./install_sglang_diffusion.sh --source

# 跳过可选依赖
./install_sglang_diffusion.sh --skip-optional
```

### 2. 生成图像

```bash
# 基础生成
./generate_qwen_image.sh -p "一只可爱的猫咪"

# 指定尺寸
./generate_qwen_image.sh -p "美丽风景" -W 1920 -H 1080

# 启用Cache-DiT加速
./generate_qwen_image.sh -p "科幻场景" --cache-dit

# 多GPU并行
./generate_qwen_image.sh -p "复杂场景" --gpus 2 --cfg-parallel

# 完整加速配置
./generate_qwen_image.sh -p "高质量图像" \
    --cache-dit --torch-compile --gpus 4 --cfg-parallel
```

### 3. 编辑图像

```bash
# 基础编辑
./edit_qwen_image.sh -i input.jpg -p "将背景改为海滩"

# 启用加速
./edit_qwen_image.sh -i input.jpg -p "添加彩虹" --cache-dit

# 多GPU加速
./edit_qwen_image.sh -i large_image.jpg -p "复杂编辑" --gpus 2
```

### 4. Python API

```bash
# 运行所有示例
python3 python_api_example.py

# 查看帮助
python3 python_api_example.py --help
```

## 环境变量

### Cache-DiT 加速

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=3              # 前Fn个block计算
export SGLANG_CACHE_DIT_BN=3              # 后Bn个block计算
export SGLANG_CACHE_DIT_RDT=0.12          # 残差差异阈值
export SGLANG_CACHE_DIT_SCM_PRESET=fast   # SCM预设 (fast/balanced/quality)
export SGLANG_CACHE_DIT_TAYLORSEER=true   # 启用TaylorSeer
```

### torch.compile 加速

```bash
export SGLANG_ENABLE_TORCH_COMPILE=true
export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
```

### 注意力后端

```bash
# 选择注意力后端
export SGLANG_ATTENTION_BACKEND=flashattention  # flashattention/sageattention/sla
```

## 常用参数

### 生成参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p, --prompt` | 生成提示词 | 必需 |
| `-o, --output` | 输出路径 | ./output.png |
| `-W, --width` | 图像宽度 | 1024 |
| `-H, --height` | 图像高度 | 1024 |
| `-s, --steps` | 推理步数 | 30 |
| `-g, --guidance` | 引导比例 | 2.5 |
| `--seed` | 随机种子 | 随机 |

### 性能参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpus` | GPU数量 | 1 |
| `--cfg-parallel` | 启用CFG并行 | false |
| `--cache-dit` | 启用Cache-DiT | false |
| `--torch-compile` | 启用torch.compile | false |

## 示例场景

### 场景1: 快速原型

```bash
# 快速生成草图
./generate_qwen_image.sh -p "设计草图" -s 10
```

### 场景2: 高质量生成

```bash
# 高质量图像
./generate_qwen_image.sh -p "高质量艺术作品" \
    -s 50 -g 3.0 --cache-dit --seed 42
```

### 场景3: 批量生产

```bash
# 批量生成（使用Python API）
python3 -c "
from sglang.multimodal_gen import generate_image
prompts = ['猫咪', '狗狗', '兔子']
results = generate_image(model_path='Qwen/Qwen-Image', prompt=prompts)
for i, r in enumerate(results):
    r.save(f'batch_{i}.png')
"
```

### 场景4: 图像编辑工作流

```bash
# 步骤1: 生成基础图像
./generate_qwen_image.sh -p "人物肖像" -o portrait.png

# 步骤2: 编辑图像
./edit_qwen_image.sh -i portrait.png -p "添加太阳镜" -o portrait_glasses.png

# 步骤3: 进一步编辑
./edit_qwen_image.sh -i portrait_glasses.png -p "改变背景为海滩" -o final.png
```

## 故障排除

### OOM (显存不足)

```bash
# 减小图像尺寸
./generate_qwen_image.sh -p "测试" -W 512 -H 512

# 启用layer-wise offload
export SGLANG_ENABLE_LAYERWISE_OFFLOAD=true
```

### 生成速度慢

```bash
# 启用所有加速选项
./generate_qwen_image.sh -p "测试" \
    --cache-dit --torch-compile --gpus 2 --cfg-parallel
```

### 图像质量差

```bash
# 增加推理步数
./generate_qwen_image.sh -p "高质量" -s 50 -g 3.0
```

## 更多信息

- [SGLang-Diffusion 详细文档](../sglang_diffusion_detailed_guide.md)
- [SGLang 官方文档](https://sglang.io/)
- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
