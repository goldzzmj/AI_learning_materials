#!/bin/bash
# =============================================================================
# Qwen-Image 文生图快速生成脚本
# 描述: 使用SGLang-Diffusion生成Qwen-Image图像
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
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen-Image}"
PROMPT="${PROMPT:-一只可爱的猫咪在草地上玩耍，阳光明媚，高清细节}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
SEED="${SEED:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./output.png}"
NUM_GPUS="${NUM_GPUS:-1}"
ENABLE_CFG_PARALLEL="${ENABLE_CFG_PARALLEL:-false}"

# 加速选项
ENABLE_CACHE_DIT="${ENABLE_CACHE_DIT:-false}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"

# 显示帮助
show_help() {
    cat << 'EOF'
Qwen-Image 文生图生成脚本

用法: ./generate_qwen_image.sh [选项]

基本选项:
  -p, --prompt TEXT           生成提示词 (必需)
  -o, --output PATH           输出路径 (默认: ./output.png)
  -W, --width NUM             图像宽度 (默认: 1024)
  -H, --height NUM            图像高度 (默认: 1024)
  -s, --steps NUM             推理步数 (默认: 30)
  -g, --guidance NUM          引导比例 (默认: 2.5)
  --seed NUM                  随机种子
  
模型选项:
  -m, --model PATH            模型路径 (默认: Qwen/Qwen-Image)
  
性能选项:
  --gpus NUM                  GPU数量 (默认: 1)
  --cfg-parallel              启用CFG并行
  --cache-dit                 启用Cache-DiT加速
  --torch-compile             启用torch.compile
  
其他选项:
  -h, --help                  显示帮助
  --examples                  显示示例

示例:
  ./generate_qwen_image.sh -p "一只可爱的猫咪"
  ./generate_qwen_image.sh -p "日落海滩" -W 1920 -H 1080 --cache-dit
  ./generate_qwen_image.sh -p "未来城市" --gpus 2 --cfg-parallel

EOF
}

# 显示示例
show_examples() {
    cat << 'EOF'
使用示例:
=========

1. 基础生成:
   ./generate_qwen_image.sh -p "一只可爱的猫咪"

2. 指定图像尺寸:
   ./generate_qwen_image.sh -p "美丽风景" -W 1920 -H 1080

3. 启用Cache-DiT加速:
   ./generate_qwen_image.sh -p "科幻场景" --cache-dit

4. 多GPU并行:
   ./generate_qwen_image.sh -p "复杂场景" --gpus 2 --cfg-parallel

5. 完整加速配置:
   ./generate_qwen_image.sh -p "高质量图像" \
       --cache-dit --torch-compile --gpus 4 --cfg-parallel

6. 使用随机种子（可复现）:
   ./generate_qwen_image.sh -p "特定风格" --seed 42

7. 自定义输出路径:
   ./generate_qwen_image.sh -p "艺术作品" -o ./my_art.png

EOF
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--prompt)
                PROMPT="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_PATH="$2"
                shift 2
                ;;
            -W|--width)
                WIDTH="$2"
                shift 2
                ;;
            -H|--height)
                HEIGHT="$2"
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
            --cfg-parallel)
                ENABLE_CFG_PARALLEL="true"
                shift
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
    if [ -z "$PROMPT" ]; then
        print_error "请提供提示词 (-p)"
        show_help
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
    print_highlight "Qwen-Image 文生图生成"
    print_highlight "======================================"
    echo ""
    print_info "模型: $MODEL_PATH"
    print_info "提示词: $PROMPT"
    print_info "尺寸: ${WIDTH}x${HEIGHT}"
    print_info "步数: $NUM_INFERENCE_STEPS"
    print_info "引导比例: $GUIDANCE_SCALE"
    [ -n "$SEED" ] && print_info "随机种子: $SEED"
    print_info "输出: $OUTPUT_PATH"
    echo ""
    print_info "GPU数量: $NUM_GPUS"
    print_info "CFG并行: $ENABLE_CFG_PARALLEL"
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
    cmd="$cmd --width $WIDTH"
    cmd="$cmd --height $HEIGHT"
    cmd="$cmd --num-inference-steps $NUM_INFERENCE_STEPS"
    cmd="$cmd --guidance-scale $GUIDANCE_SCALE"
    cmd="$cmd --save-output \"$OUTPUT_PATH\""
    
    [ -n "$SEED" ] && cmd="$cmd --seed $SEED"
    [ "$NUM_GPUS" -gt 1 ] && cmd="$cmd --num-gpus $NUM_GPUS"
    [ "$ENABLE_CFG_PARALLEL" = "true" ] && cmd="$cmd --enable-cfg-parallel"
    
    echo "$cmd"
}

# 执行生成
run_generation() {
    local cmd=$(build_command)
    
    print_info "执行命令:"
    echo "$cmd"
    echo ""
    
    # 执行命令
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "生成成功！"
        print_success "输出文件: $OUTPUT_PATH"
        
        # 显示文件信息
        if [ -f "$OUTPUT_PATH" ]; then
            local size=$(du -h "$OUTPUT_PATH" | cut -f1)
            print_info "文件大小: $size"
        fi
    else
        print_error "生成失败！"
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
    
    # 执行生成
    run_generation
}

# 运行
main "$@"
