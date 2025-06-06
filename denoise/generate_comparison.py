# generate_comparison.py
#
# 描述:
#   读取一个原始视频和一个处理后的视频，生成一个左右分屏的对比视频，
#   并计算处理前后视频的平均PSNR和SSIM值。
#
# 用法:
#   python generate_comparison.py --original <原始视频> --denoised <降噪视频> --output <对比视频>

import cv2
import numpy as np
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def create_comparison_and_metrics(original_path, denoised_path, output_path):
    """
    读取原始视频和降噪后的视频，计算PSNR/SSIM，并生成左右对比视频。
    """
    # 打开两个视频文件
    cap_orig = cv2.VideoCapture(original_path)
    cap_denoised = cv2.VideoCapture(denoised_path)

    # 检查文件是否成功打开
    if not cap_orig.isOpened():
        print(f"错误: 无法打开原始视频文件: {original_path}")
        return
    if not cap_denoised.isOpened():
        print(f"错误: 无法打开降噪视频文件: {denoised_path}")
        return

    # 获取视频属性 (以原始视频为准)
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    # 检查帧数是否为0
    if frame_count == 0:
        print("错误: 视频文件没有帧。")
        cap_orig.release()
        cap_denoised.release()
        return

    # 创建一个新的VideoWriter，宽度是原来的两倍
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    total_psnr = 0
    total_ssim = 0
    
    print("正在生成对比视频并计算PSNR/SSIM...")
    # 使用tqdm创建进度条
    for _ in tqdm(range(frame_count), desc="对比视频生成中", unit="帧"):
        ret_orig, frame_orig = cap_orig.read()
        ret_denoised, frame_denoised = cap_denoised.read()

        # 如果任何一个视频提前结束，则停止处理
        if not ret_orig or not ret_denoised:
            print("\n警告: 其中一个视频提前结束，处理中断。")
            break

        # --- 计算指标 ---
        # PSNR: 峰值信噪比, 数值越高越好
        total_psnr += psnr(frame_orig, frame_denoised, data_range=255)
        
        # SSIM: 结构相似性, 数值越接近1越好
        total_ssim += ssim(frame_orig, frame_denoised, multichannel=True, channel_axis=2)

        # --- 在帧上添加标签 ---
        # 设置字体、大小、颜色和厚度
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (255, 255, 255) # 白色
        thickness = 3
        
        cv2.putText(frame_orig, 'Original', (30, 70), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(frame_denoised, 'Denoised', (30, 70), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # --- 将两帧水平拼接 ---
        comparison_frame = np.hstack((frame_orig, frame_denoised))
        
        # 写入拼接后的帧
        out.write(comparison_frame)

    # --- 打印最终结果 ---
    avg_psnr = total_psnr / frame_count
    avg_ssim = total_ssim / frame_count
    
    print("\n" + "-" * 40)
    print("客观指标评估完成:")
    print(f"  平均峰值信噪比 (PSNR): {avg_psnr:.2f} dB")
    print(f"  平均结构相似性 (SSIM): {avg_ssim:.4f}")
    print("-" * 40)
    print(f"对比视频已成功生成: {output_path}")

    # 释放所有资源
    cap_orig.release()
    cap_denoised.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="生成降噪前后对比视频及质量评估指标")
    parser.add_argument('--original', type=str, required=True, help="原始带噪声的视频文件。")
    parser.add_argument('--denoised', type=str, required=True, help="已经降噪处理后的视频文件。")
    parser.add_argument('--output', type=str, required=True, help="输出的对比视频文件路径。")
    args = parser.parse_args()

    # 运行主函数
    create_comparison_and_metrics(args.original, args.denoised, args.output)