# -----------------------------------------------------------------------------
# denoise_video.py
#
# 描述:
#   使用预训练的 SwinIR 深度学习模型对视频文件进行降噪。
#   此脚本设计为在 NVIDIA GPU 上通过 PyTorch 运行。
#
# 作者: Gemini (基于与用户的交流)
# 日期: 2025年6月5日
#
# 用法:
#   python denoise_video.py \
#       --input <输入视频路径> \
#       --output <输出视频路径> \
#       --model <SwinIR模型权重路径.pth> \
#       --gpu <要使用的GPU ID>
#
# 示例:
#   python denoise_video.py \
#       --input noisy_video.mp4 \
#       --output denoised_video.mp4 \
#       --model 005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth \
#       --gpu 0
# -----------------------------------------------------------------------------

import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm

def denoise_video(input_path, output_path, model_path, gpu_id, noise_level):
    """
    使用指定的SwinIR模型在GPU上对视频进行降噪。

    参数:
    input_path (str): 输入的带噪声视频文件路径。
    output_path (str): 处理后输出的降噪视频文件路径。
    model_path (str): 预训练的SwinIR模型 (.pth) 文件路径。
    gpu_id (int): 要使用的GPU设备ID。
    noise_level (int): 模型训练时对应的噪声水平 (例如 15, 25, 50)。
    """
    # --- 1. 环境和设备设置 ---
    if not torch.cuda.is_available():
        print("错误: PyTorch 未检测到可用的 CUDA 设备。请检查您的安装。")
        return
    
    if gpu_id >= torch.cuda.device_count():
        print(f"错误: 无效的 GPU ID {gpu_id}。检测到的GPU数量为 {torch.cuda.device_count()}。")
        return

    device = torch.device(f'cuda:{gpu_id}')
    print(f"使用的设备: {device} ({torch.cuda.get_device_name(gpu_id)})")

    # --- 2. 视频 I/O 设置 ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开输入视频: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用 'mp4v' 编码器，它具有较好的兼容性
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {frame_count} 帧。")

    # --- 3. 加载 SwinIR 模型 ---
    try:
        # 动态导入模型架构。确保 model_architecture.py 在同一目录下。
        from model_architecture import SwinIR as ModelClass
        
        # 为彩色图像降噪任务定义SwinIR模型。
        # 参数需要与SwinIR-M模型的定义相匹配。
        model = ModelClass(upscale=1, img_size=(height, width),
                           window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                           embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                           mlp_ratio=2, upsampler='', resi_connection='1conv',
                           task='color_dn', noise=noise_level)

        # 加载预训练的权重文件
        pretrained_model = torch.load(model_path, map_location='cpu')
        
        # 自动处理不同的权重保存格式（例如 'params' 或 'params_ema'）
        param_key = 'params'
        if param_key not in pretrained_model:
            param_key = next(iter(pretrained_model))
        
        model.load_state_dict(pretrained_model[param_key], strict=True)
        
        # 将模型移动到指定的GPU
        model.to(device)
        # 设置为评估模式（这会禁用dropout等训练特有的层）
        model.eval()
        print("SwinIR 模型已成功加载到GPU。")

    except ImportError:
        print("错误: 无法导入模型架构。请确保 'model_architecture.py' 文件与此脚本在同一目录下。")
        return
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # --- 4. 逐帧处理视频 ---
    print(f"开始处理视频 '{input_path}'...")
    
    # torch.no_grad() 上下文管理器可以禁用梯度计算，从而减少内存消耗并加速推理
    with torch.no_grad():
        # tqdm 用于创建一个漂亮的进度条
        for _ in tqdm(range(frame_count), desc="降噪进度", unit="帧"):
            ret, frame = cap.read()
            if not ret:
                print("\n视频帧读取完毕或发生错误。")
                break

            # 图像预处理:
            # 1. OpenCV 默认使用 BGR, 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2. 从 NumPy 数组 (H, W, C) 转换为 PyTorch 张量 (C, H, W)
            # 3. 将像素值从 [0, 255] 归一化到 [0.0, 1.0]
            # 4. 增加一个批次维度 (1, C, H, W)
            # 5. 将张量移动到GPU
            img_tensor = (torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
            
            # 在GPU上执行降噪
            denoised_tensor = model(img_tensor)

            # 图像后处理:
            # 1. 移除批次维度
            # 2. 将张量移回CPU
            # 3. 限制像素值在 [0, 1] 范围内
            # 4. 反归一化到 [0, 255] 并转换为8位整数
            # 5. 从 (C, H, W) 转回 (H, W, C)
            # 6. 转换为NumPy数组
            denoised_frame_rgb = (denoised_tensor.squeeze(0).cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            
            # 从 RGB 转回 BGR 以便 OpenCV 写入
            denoised_frame_bgr = cv2.cvtColor(denoised_frame_rgb, cv2.COLOR_RGB2BGR)

            # 将处理后的帧写入输出视频文件
            out.write(denoised_frame_bgr)

    # --- 5. 清理和收尾 ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("-" * 40)
    print(f"处理完成！降噪后的视频已保存到: '{output_path}'")
    print("-" * 40)


if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="使用 SwinIR 模型对视频进行高性能降噪")
    parser.add_argument('--input', type=str, required=True, help="输入的带噪声视频文件路径。")
    parser.add_argument('--output', type=str, required=True, help="降噪后视频的保存路径。")
    parser.add_argument('--model', type=str, required=True, help="预训练的 SwinIR (.pth) 模型权重文件路径。")
    parser.add_argument('--gpu', type=int, default=0, help="要使用的GPU的ID (例如 0, 1, 2, ...)。默认为0。")
    parser.add_argument('--noise', type=int, default=25, help="模型对应的噪声水平 (15, 25, 50)。必须与模型文件匹配。默认为25。")
    
    args = parser.parse_args()

    # 调用主函数执行降噪任务
    denoise_video(args.input, args.output, args.model, args.gpu, args.noise)