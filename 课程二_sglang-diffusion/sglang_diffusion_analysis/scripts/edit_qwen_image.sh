#!/bin/bash
# =============================================================================
# Qwen-Image-Edit 图像编辑快速脚本
# 描述: 使用SGLang-Diffusion进行Qwen-Image图像编辑
# 作者: AI Assistant
# 日期: 2026-02-03
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 打印函数
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_highlight() { echo -e "${CYAN}$1${NC}"; }

# 默认配置
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen-Image-Edit-2511}"
IMAGE_PATH="${IMAGE_PATH:-}"
PROMPT="${PROMPT:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./edited_output.png}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
SEED="${SEED:-}"
NUM_GPUS="${NUM_GPUS:-1}"

# 加速选项
ENABLE_CACHE_DIT="${ENABLE_CACHE_DIT:-false}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"

# 显示帮助
show_help() {
    cat << 'EOF'
Qwen-Image-Edit 图像编辑脚本

用法: ./edit_qwen_image.sh [选项]

基本选项:
  -i, --image PATH            输入图像路径 (必需)
  -p, --prompt TEXT           编辑指令 (必需)
  -o, --output PATH           输出路径 (默认: ./edited_output.png)
  -s, --steps NUM             推理步数 (默认: 30)
  -g, --guidance NUM          引导比例 (默认: 2.5)
  --seed NUM                  随机种子
  
模型选项:
  -m, --model PATH            模型路径 (默认: Qwen/Qwen-Image-Edit-2511)
  
性能选项:
  --gpus NUM                  GPU数量 (默认: 1)
  --cache-dit                 启用Cache-DiT加速
  --torch-compile             启用torch.compile
  
其他选项:
  -h, --help                  显示帮助
  --examples                  显示示例

示例:
  ./edit_qwen_image.sh -i input.jpg -p "将背景改为海滩"
  ./edit_qwen_image.sh -i photo.png -p "添加彩虹" --cache-dit
  ./edit_qwen_image.sh -i portrait.jpg -p "改变发型为短发" --gpus 2

EOF
}

# 显示示例
show_examples() {
    cat << 'EOF'
使用示例:
=========

1. 基础编辑:
   ./edit_qwen_image.sh -i input.jpg -p "将背景改为海滩"

2. 添加元素:
   ./edit_qwen_image.sh -i photo.png -p "添加一只鸟在天空中"

3. 风格转换:
   ./edit_qwen_image.sh -i portrait.jpg -p "转换为油画风格"

4. 颜色调整:
   ./edit_qwen_image.sh -i landscape.jpg -p "将天空变成紫色"

5. 物体替换:
   ./edit_qwen_image.sh -i car.jpg -p "将汽车变成红色跑车"

6. 启用加速:
   ./edit_qwen_image.sh -i input.jpg -p "高质量编辑" --cache-dit

7. 多GPU加速:
   ./edit_qwen_image.sh -i large_image.jpg -p "复杂编辑" --gpus 2

编辑指令技巧:
=============

• 具体明确: "将背景改为日落时分的海滩"
• 使用形容词: "添加一只可爱的小狗"
• 指定风格: "转换为赛博朋克风格"
• 描述变化: "将衣服颜色从蓝色改为红色"
• 添加细节: "在背景中添加山脉和湖泊"

EOF
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--image)
                IMAGE_PATH="$2"
                shift 2
                ;;
            -p|--prompt)
                PROMPT="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_PATH="$2"
                shift 2
                ;;
            -s|--steps)
                NUM_INFERENCE_STEPS="$2"
                shift 2
                ;;
            -g|--guidance)
                GUIDANCE_SCALE="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            -m|--model)
                MODEL_PATH="$2"
                shift 2
                ;;
            --gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            --cache-dit)
                ENABLE_CACHE_DIT="true"
                shift
                ;;
            --torch-compile)
                ENABLE_TORCH_COMPILE="true"
                shift
                ;;
            --examples)
                show_examples
                exit 0
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证必需参数
    if [ -z "$IMAGE_PATH" ]; then
        print_error "请提供输入图像路径 (-i)"
        show_help
        exit 1
    fi
    
    if [ -z "$PROMPT" ]; then
        print_error "请提供编辑指令 (-p)"
        show_help
        exit 1
    fi
    
    # 检查输入文件是否存在
    if [ ! -f "$IMAGE_PATH" ]; then
        print_error "输入图像不存在: $IMAGE_PATH"
        exit 1
    fi
}

# 设置加速环境变量
setup_acceleration() {
    print_info "配置加速选项..."
    
    if [ "$ENABLE_CACHE_DIT" = "true" ]; then
        export SGLANG_CACHE_DIT_ENABLED=true
        export SGLANG_CACHE_DIT_FN=3
        export SGLANG_CACHE_DIT_BN=3
        export SGLANG_CACHE_DIT_RDT=0.12
        export SGLANG_CACHE_DIT_SCM_PRESET=fast
        export SGLANG_CACHE_DIT_TAYLORSEER=true
        print_success "Cache-DiT 已启用"
    fi
    
    if [ "$ENABLE_TORCH_COMPILE" = "true" ]; then
        export SGLANG_ENABLE_TORCH_COMPILE=true
        export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
        print_success "torch.compile 已启用"
    fi
}

# 打印配置
print_config() {
    echo ""
    print_highlight "======================================"
    print_highlight "Qwen-Image-Edit 图像编辑"
    print_highlight "======================================"
    echo ""
    print_info "模型: $MODEL_PATH"
    print_info "输入图像: $IMAGE_PATH"
    print_info "编辑指令: $PROMPT"
    print_info "步数: $NUM_INFERENCE_STEPS"
    print_info "引导比例: $GUIDANCE_SCALE"
    [ -n "$SEED" ] && print_info "随机种子: $SEED"
    print_info "输出: $OUTPUT_PATH"
    echo ""
    print_info "GPU数量: $NUM_GPUS"
    print_info "Cache-DiT: $ENABLE_CACHE_DIT"
    print_info "torch.compile: $ENABLE_TORCH_COMPILE"
    echo ""
    print_highlight "======================================"
    echo ""
}

# 构建命令
build_command() {
    local cmd="sglang generate"
    
    cmd="$cmd --model-path \"$MODEL_PATH\""
    cmd="$cmd --prompt \"$PROMPT\""
    cmd="$cmd --image-path \"$IMAGE_PATH\""
    cmd="$cmd --num-inference-steps $NUM_INFERENCE_STEPS"
    cmd="$cmd --guidance-scale $GUIDANCE_SCALE"
    cmd="$cmd --save-output \"$OUTPUT_PATH\""
    
    [ -n "$SEED" ] && cmd="$cmd --seed $SEED"
    [ "$NUM_GPUS" -gt 1 ] && cmd="$cmd --num-gpus $NUM_GPUS"
    
    echo "$cmd"
}

# 执行编辑
run_edit() {
    local cmd=$(build_command)
    
    print_info "执行命令:"
    echo "$cmd"
    echo ""
    
    # 执行命令
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "编辑成功！"
        print_success "输出文件: $OUTPUT_PATH"
        
        # 显示文件信息
        if [ -f "$OUTPUT_PATH" ]; then
            local size=$(du -h "$OUTPUT_PATH" | cut -f1)
            print_info "文件大小: $size"
        fi
    else
        print_error "编辑失败！"
        exit 1
    fi
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    # 设置加速
    setup_acceleration
    
    # 打印配置
    print_config
    
    # 执行编辑
    run_edit
}

# 运行
main "$@"
