#!/usr/bin/env python3
# =============================================================================
# SGLang-Diffusion Python API 使用示例
# 描述: 展示如何使用Python API进行图像生成和编辑
# 作者: AI Assistant
# 日期: 2026-02-03
# =============================================================================

"""
SGLang-Diffusion Python API 使用示例

本脚本展示了如何使用SGLang-Diffusion的Python API进行:
1. 文生图 (Text-to-Image)
2. 图像编辑 (Image-to-Image Editing)
3. 批量生成 (Batch Generation)
4. 高级配置 (Advanced Configuration)
"""

import os
import sys
from typing import List, Optional, Union
import time

# 确保SGLang已安装
try:
    import sglang
    from sglang.multimodal_gen import generate_image
except ImportError:
    print("错误: 请先安装SGLang-Diffusion")
    print("运行: pip install 'sglang[diffusion]' --prerelease=allow")
    sys.exit(1)


# =============================================================================
# 示例 1: 基础文生图
# =============================================================================
def example_basic_text_to_image():
    """
    基础文生图示例
    
    最简单的使用方式，只需要提供模型路径和提示词
    """
    print("\n" + "="*60)
    print("示例 1: 基础文生图")
    print("="*60)
    
    start_time = time.time()
    
    # 生成图像
    result = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt="一只可爱的猫咪在草地上玩耍，阳光明媚，高清细节",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=2.5,
        seed=42,  # 设置随机种子以获得可复现的结果
    )
    
    # 保存图像
    output_path = "output_basic.png"
    result.save(output_path)
    
    elapsed_time = time.time() - start_time
    print(f"✓ 图像已保存到: {output_path}")
    print(f"✓ 生成时间: {elapsed_time:.2f}秒")


# =============================================================================
# 示例 2: 图像编辑
# =============================================================================
def example_image_editing():
    """
    图像编辑示例
    
    使用Qwen-Image-Edit模型对现有图像进行编辑
    """
    print("\n" + "="*60)
    print("示例 2: 图像编辑")
    print("="*60)
    
    # 检查输入图像是否存在
    input_image = "input.jpg"
    if not os.path.exists(input_image):
        print(f"⚠ 输入图像不存在: {input_image}")
        print("  跳过此示例")
        return
    
    start_time = time.time()
    
    # 编辑图像
    result = generate_image(
        model_path="Qwen/Qwen-Image-Edit-2511",
        prompt="将背景改为海滩，添加日落",
        image_path=input_image,
        num_inference_steps=30,
        guidance_scale=2.5,
    )
    
    # 保存编辑后的图像
    output_path = "output_edited.png"
    result.save(output_path)
    
    elapsed_time = time.time() - start_time
    print(f"✓ 编辑后的图像已保存到: {output_path}")
    print(f"✓ 编辑时间: {elapsed_time:.2f}秒")


# =============================================================================
# 示例 3: 批量生成
# =============================================================================
def example_batch_generation():
    """
    批量生成示例
    
    一次生成多张图像，提高效率
    """
    print("\n" + "="*60)
    print("示例 3: 批量生成")
    print("="*60)
    
    prompts = [
        "一只可爱的猫咪",
        "一只可爱的狗狗",
        "一只可爱的兔子",
    ]
    
    start_time = time.time()
    
    # 批量生成
    results = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt=prompts,  # 传入列表进行批量生成
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=2.5,
    )
    
    # 保存所有图像
    for i, result in enumerate(results):
        output_path = f"output_batch_{i}.png"
        result.save(output_path)
        print(f"✓ 图像 {i+1} 已保存到: {output_path}")
    
    elapsed_time = time.time() - start_time
    print(f"✓ 批量生成完成，平均每张: {elapsed_time/len(prompts):.2f}秒")


# =============================================================================
# 示例 4: 使用Cache-DiT加速
# =============================================================================
def example_with_cache_dit():
    """
    使用Cache-DiT加速示例
    
    Cache-DiT可以显著加速生成过程，最高可达169%的加速
    """
    print("\n" + "="*60)
    print("示例 4: 使用Cache-DiT加速")
    print("="*60)
    
    # 设置Cache-DiT环境变量
    os.environ["SGLANG_CACHE_DIT_ENABLED"] = "true"
    os.environ["SGLANG_CACHE_DIT_FN"] = "3"  # 前3个block计算
    os.environ["SGLANG_CACHE_DIT_BN"] = "3"  # 后3个block计算
    os.environ["SGLANG_CACHE_DIT_RDT"] = "0.12"  # 残差差异阈值
    os.environ["SGLANG_CACHE_DIT_SCM_PRESET"] = "fast"  # 快速预设
    os.environ["SGLANG_CACHE_DIT_TAYLORSEER"] = "true"  # 启用TaylorSeer
    
    print("Cache-DiT配置:")
    print(f"  - Fn: {os.environ['SGLANG_CACHE_DIT_FN']}")
    print(f"  - Bn: {os.environ['SGLANG_CACHE_DIT_BN']}")
    print(f"  - RDT: {os.environ['SGLANG_CACHE_DIT_RDT']}")
    print(f"  - SCM Preset: {os.environ['SGLANG_CACHE_DIT_SCM_PRESET']}")
    print(f"  - TaylorSeer: {os.environ['SGLANG_CACHE_DIT_TAYLORSEER']}")
    
    start_time = time.time()
    
    result = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt="未来城市，科幻风格，高清细节",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=2.5,
    )
    
    output_path = "output_cache_dit.png"
    result.save(output_path)
    
    elapsed_time = time.time() - start_time
    print(f"✓ 图像已保存到: {output_path}")
    print(f"✓ 生成时间 (with Cache-DiT): {elapsed_time:.2f}秒")


# =============================================================================
# 示例 5: 不同尺寸的图像生成
# =============================================================================
def example_different_sizes():
    """
    不同尺寸的图像生成示例
    
    展示如何生成不同宽高比的图像
    """
    print("\n" + "="*60)
    print("示例 5: 不同尺寸的图像生成")
    print("="*60)
    
    # 定义不同的尺寸配置
    size_configs = [
        ("正方形", 1024, 1024),
        ("横屏", 1920, 1080),
        ("竖屏", 1080, 1920),
    ]
    
    for name, width, height in size_configs:
        print(f"\n生成 {name} 图像 ({width}x{height})...")
        
        start_time = time.time()
        
        result = generate_image(
            model_path="Qwen/Qwen-Image",
            prompt=f"美丽的风景，{name}构图",
            width=width,
            height=height,
            num_inference_steps=20,  # 减少步数以加快演示
            guidance_scale=2.5,
        )
        
        output_path = f"output_{name}.png"
        result.save(output_path)
        
        elapsed_time = time.time() - start_time
        print(f"  ✓ 已保存: {output_path} ({elapsed_time:.2f}秒)")


# =============================================================================
# 示例 6: 高级配置
# =============================================================================
def example_advanced_configuration():
    """
    高级配置示例
    
    展示更多高级选项的使用
    """
    print("\n" + "="*60)
    print("示例 6: 高级配置")
    print("="*60)
    
    # 设置多种加速选项
    os.environ["SGLANG_CACHE_DIT_ENABLED"] = "true"
    os.environ["SGLANG_CACHE_DIT_SCM_PRESET"] = "fast"
    os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "true"
    os.environ["SGLANG_TORCH_COMPILE_MODE"] = "max-autotune-no-cudagraphs"
    
    print("高级配置:")
    print("  - Cache-DiT: 启用")
    print("  - torch.compile: 启用")
    print("  - 模式: max-autotune-no-cudagraphs")
    
    start_time = time.time()
    
    result = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt="高质量艺术作品，精细细节，专业摄影",
        width=1024,
        height=1024,
        num_inference_steps=50,  # 更多步数以获得更高质量
        guidance_scale=3.0,  # 更高的引导比例
        seed=12345,  # 固定种子
    )
    
    output_path = "output_advanced.png"
    result.save(output_path)
    
    elapsed_time = time.time() - start_time
    print(f"✓ 高质量图像已保存到: {output_path}")
    print(f"✓ 生成时间: {elapsed_time:.2f}秒")


# =============================================================================
# 示例 7: 性能对比
# =============================================================================
def example_performance_comparison():
    """
    性能对比示例
    
    对比不同配置下的生成速度
    """
    print("\n" + "="*60)
    print("示例 7: 性能对比")
    print("="*60)
    
    prompt = "一只可爱的猫咪在草地上玩耍"
    
    # 测试1: 基础配置
    print("\n测试 1: 基础配置")
    start = time.time()
    result1 = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt=prompt,
        num_inference_steps=30,
    )
    time1 = time.time() - start
    print(f"  时间: {time1:.2f}秒")
    
    # 测试2: 启用Cache-DiT
    print("\n测试 2: 启用Cache-DiT")
    os.environ["SGLANG_CACHE_DIT_ENABLED"] = "true"
    os.environ["SGLANG_CACHE_DIT_SCM_PRESET"] = "fast"
    
    start = time.time()
    result2 = generate_image(
        model_path="Qwen/Qwen-Image",
        prompt=prompt,
        num_inference_steps=30,
    )
    time2 = time.time() - start
    speedup = time1 / time2
    print(f"  时间: {time2:.2f}秒")
    print(f"  加速比: {speedup:.2f}x")
    
    # 保存结果
    result1.save("output_perf_baseline.png")
    result2.save("output_perf_accelerated.png")
    print("\n✓ 结果已保存:")
    print("  - output_perf_baseline.png (基础配置)")
    print("  - output_perf_accelerated.png (加速配置)")


# =============================================================================
# 主函数
# =============================================================================
def main():
    """主函数：运行所有示例"""
    
    print("\n" + "="*60)
    print("SGLang-Diffusion Python API 使用示例")
    print("="*60)
    print("\n本脚本展示了SGLang-Diffusion的各种使用方式")
    print("每个示例都可以单独运行")
    
    # 运行示例
    examples = [
        ("基础文生图", example_basic_text_to_image),
        ("图像编辑", example_image_editing),
        ("批量生成", example_batch_generation),
        ("Cache-DiT加速", example_with_cache_dit),
        ("不同尺寸", example_different_sizes),
        ("高级配置", example_advanced_configuration),
        ("性能对比", example_performance_comparison),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    # 询问用户要运行哪个示例
    print("\n请输入要运行的示例编号 (1-7)，或输入 'all' 运行所有示例:")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n✗ 示例 '{name}' 失败: {e}")
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            try:
                examples[idx][1]()
            except Exception as e:
                print(f"\n✗ 示例失败: {e}")
        else:
            print("无效的示例编号")
    else:
        print("无效的选择")
    
    print("\n" + "="*60)
    print("示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
