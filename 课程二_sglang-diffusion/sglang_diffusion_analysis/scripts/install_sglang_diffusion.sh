#!/bin/bash
# =============================================================================
# SGLang-Diffusion 安装脚本
# 描述: 自动安装SGLang-Diffusion框架及其依赖
# 作者: AI Assistant
# 日期: 2026-02-03
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印标题
print_header() {
    echo ""
    echo "======================================"
    echo "$1"
    echo "======================================"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 获取Python版本
get_python_version() {
    python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'
}

# 检查Python版本是否满足要求
check_python_version() {
    local version=$(get_python_version)
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 9 ]); then
        print_error "Python版本 $version 不满足要求，需要 >= 3.9"
        exit 1
    fi
    
    print_success "Python版本检查通过: $version"
}

# 检查CUDA是否可用
check_cuda() {
    if command_exists nvidia-smi; then
        print_info "检测到NVIDIA GPU:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    else
        print_warning "未检测到NVIDIA GPU，将安装CPU版本"
        return 1
    fi
}

# 安装uv包管理器
install_uv() {
    print_header "安装 uv 包管理器"
    
    if command_exists uv; then
        print_success "uv 已安装: $(uv --version)"
        return 0
    fi
    
    print_info "正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # 添加uv到PATH
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if command_exists uv; then
        print_success "uv 安装成功: $(uv --version)"
    else
        print_error "uv 安装失败"
        exit 1
    fi
}

# 安装SGLang-Diffusion
install_sglang_diffusion() {
    print_header "安装 SGLang-Diffusion"
    
    print_info "正在安装 SGLang-Diffusion（这可能需要几分钟）..."
    
    # 使用uv安装（更快）
    uv pip install 'sglang[diffusion]' --prerelease=allow
    
    print_success "SGLang-Diffusion 安装完成"
}

# 从源码安装（开发版本）
install_from_source() {
    print_header "从源码安装 SGLang-Diffusion"
    
    local install_dir=${1:-"$HOME/sglang"}
    
    print_info "克隆仓库到 $install_dir..."
    
    if [ -d "$install_dir" ]; then
        print_warning "目录已存在，更新代码..."
        cd "$install_dir"
        git pull
    else
        git clone https://github.com/sgl-project/sglang.git "$install_dir"
        cd "$install_dir"
    fi
    
    print_info "安装依赖..."
    uv pip install -e "python[diffusion]" --prerelease=allow
    
    print_success "源码安装完成"
}

# 验证安装
verify_installation() {
    print_header "验证安装"
    
    print_info "检查 SGLang 版本..."
    python3 -c "import sglang; print(f'SGLang 版本: {sglang.__version__}')"
    
    print_info "检查扩散模型支持..."
    python3 -c "from sglang.multimodal_gen import generate_image; print('扩散模型支持: OK')"
    
    print_success "安装验证通过！"
}

# 安装可选依赖
install_optional_deps() {
    print_header "安装可选依赖"
    
    print_info "安装 SageAttention（推荐用于长序列）..."
    uv pip install sageattention || print_warning "SageAttention 安装失败，将使用默认注意力后端"
    
    print_info "安装 flash-attn（推荐用于高性能）..."
    uv pip install flash-attn --no-build-isolation || print_warning "flash-attn 安装失败"
    
    print_success "可选依赖安装完成"
}

# 打印使用说明
print_usage() {
    cat << 'EOF'

======================================
SGLang-Diffusion 安装完成！
======================================

快速开始:
---------
1. 文生图:
   sglang generate --model-path Qwen/Qwen-Image \
       --prompt "一只可爱的猫咪" \
       --save-output output.png

2. 图像编辑:
   sglang generate --model-path Qwen/Qwen-Image-Edit-2511 \
       --prompt "将背景改为海滩" \
       --image-path input.jpg \
       --save-output edited.png

3. 启用加速:
   export SGLANG_CACHE_DIT_ENABLED=true
   export SGLANG_CACHE_DIT_SCM_PRESET=fast
   sglang generate --model-path Qwen/Qwen-Image \
       --prompt "一只可爱的猫咪"

更多选项:
---------
  --num-inference-steps: 推理步数 (默认: 30)
  --guidance-scale: 引导比例 (默认: 2.5)
  --width, --height: 图像尺寸 (默认: 1024x1024)
  --seed: 随机种子
  --num-gpus: GPU数量
  --enable-cfg-parallel: 启用CFG并行

查看详细文档:
-------------
  请查看 sglang_diffusion_detailed_guide.md

EOF
}

# 主函数
main() {
    print_header "SGLang-Diffusion 安装脚本"
    
    # 解析参数
    local from_source=false
    local install_dir="$HOME/sglang"
    local skip_optional=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --source)
                from_source=true
                shift
                ;;
            --dir)
                install_dir="$2"
                shift 2
                ;;
            --skip-optional)
                skip_optional=true
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --source          从源码安装"
                echo "  --dir <路径>      源码安装目录 (默认: ~/sglang)"
                echo "  --skip-optional   跳过可选依赖安装"
                echo "  --help            显示帮助"
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                exit 1
                ;;
        esac
    done
    
    # 检查Python版本
    check_python_version
    
    # 检查CUDA
    check_cuda
    
    # 安装uv
    install_uv
    
    # 安装SGLang-Diffusion
    if [ "$from_source" = true ]; then
        install_from_source "$install_dir"
    else
        install_sglang_diffusion
    fi
    
    # 安装可选依赖
    if [ "$skip_optional" = false ]; then
        install_optional_deps
    fi
    
    # 验证安装
    verify_installation
    
    # 打印使用说明
    print_usage
}

# 运行主函数
main "$@"
