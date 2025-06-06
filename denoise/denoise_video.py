# -----------------------------------------------------------------------------
# denoise_video_final_patch_based.py
#
# 描述:
#   使用SwinIR模型对高分辨率视频进行降噪。
#   采用分块推理(patch-based inference)策略来处理大尺寸视频帧，
#   以解决模型尺寸不匹配的问题，并有效管理GPU内存。
#
# 作者: Gemini (基于与用户的交流)
# 日期: 2025年6月5日
#
# 用法:
#   python denoise_video_final_patch_based.py \
#       --input <输入视频路径> \
#       --output <输出视频路径> \
#       --model <SwinIR模型权重路径.pth>
# -----------------------------------------------------------------------------

import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
import math

# ---------------------------------
# Helper Functions for Tiling
# ---------------------------------
def tile_image(img, tile_size=128, overlap=16):
    """将图像分割成重叠的小块"""
    h, w, c = img.shape
    tiles = []
    for i in range(0, h, tile_size - overlap):
        for j in range(0, w, tile_size - overlap):
            # 确保切片不越界
            h_end = min(i + tile_size, h)
            w_end = min(j + tile_size, w)
            tile = img[i:h_end, j:w_end, :]
            # 如果切片尺寸小于目标尺寸，进行填充
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, c), dtype=np.uint8)
                padded_tile[0:tile.shape[0], 0:tile.shape[1], :] = tile
                tile = padded_tile
            tiles.append(tile)
    return tiles, (h, w)

def untile_image(tiles, original_size, tile_size=128, overlap=16):
    """将处理后的小块重新拼接成完整图像"""
    h, w = original_size
    c = tiles[0].shape[2]
    output_img = np.zeros((h, w, c), dtype=np.float32)
    count_map = np.zeros((h, w, c), dtype=np.float32)
    
    idx = 0
    for i in range(0, h, tile_size - overlap):
        for j in range(0, w, tile_size - overlap):
            h_end = min(i + tile_size, h)
            w_end = min(j + tile_size, w)
            
            tile = tiles[idx]
            # 去掉填充部分
            tile_h, tile_w, _ = tile.shape
            original_tile_h = h_end - i
            original_tile_w = w_end - j
            
            output_img[i:h_end, j:w_end, :] += tile[0:original_tile_h, 0:original_tile_w, :]
            count_map[i:h_end, j:w_end, :] += 1
            idx += 1
            
    # 对重叠区域进行平均
    output_img /= count_map
    return np.clip(output_img, 0, 255).astype(np.uint8)

def denoise_video_patch_based(input_path, output_path, model_path, gpu_id, noise_level):
    """
    使用分块推理策略对视频进行降噪。
    """
    # --- 1. 环境和设备设置 ---
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # --- 2. 视频 I/O 设置 ---
    cap = cv2.VideoCapture(input_path)
    # ... (与之前版本相同的视频I/O设置) ...
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {frame_count} 帧。")

    # --- 3. 加载SwinIR模型 ---
    try:
        from model_architecture import SwinIR as ModelClass
        
        # 关键修复: 使用模型训练时的固定图像尺寸 (例如128x128) 来创建模型
        # 而不是视频的完整尺寸
        PATCH_SIZE = 128
        model = ModelClass(upscale=1, img_size=(PATCH_SIZE, PATCH_SIZE),
                           window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                           embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                           mlp_ratio=2, upsampler='', resi_connection='1conv',
                           task='color_dn', noise=noise_level)
                           
        pretrained_model = torch.load(model_path, map_location='cpu')
        param_key = 'params'
        if param_key not in pretrained_model:
            param_key = next(iter(pretrained_model))
        model.load_state_dict(pretrained_model[param_key], strict=True)
        model.to(device)
        model.eval()
        print("SwinIR模型已成功加载到GPU。")

    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # --- 4. 逐帧进行分块处理 ---
    print(f"开始使用分块推理处理视频 '{input_path}'...")
    with torch.no_grad():
        for _ in tqdm(range(frame_count), desc="降噪进度", unit="帧"):
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. 将帧分割成小块
            tiles, original_size = tile_image(frame_rgb, tile_size=PATCH_SIZE, overlap=32)
            denoised_tiles = []
            
            # 2. 逐块进行降噪
            for tile in tiles:
                img_tensor = (torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
                denoised_tensor = model(img_tensor)
                denoised_tile_rgb = (denoised_tensor.squeeze(0).cpu().clamp(0, 1) * 255.0).permute(1, 2, 0).numpy()
                denoised_tiles.append(denoised_tile_rgb)

            # 3. 将降噪后的小块拼接回完整图像
            denoised_frame_rgb = untile_image(denoised_tiles, original_size, tile_size=PATCH_SIZE, overlap=32)
            denoised_frame_bgr = cv2.cvtColor(denoised_frame_rgb, cv2.COLOR_RGB2BGR)
            
            out.write(denoised_frame_bgr)

    # --- 5. 清理和收尾 ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("-" * 40)
    print(f"处理完成！降噪后的视频已保存到: '{output_path}'")
    print("-" * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用SwinIR模型对视频进行高性能降噪 (分块处理版)")
    parser.add_argument('--input', type=str, required=True, help="输入的带噪声视频文件路径。")
    parser.add_argument('--output', type=str, required=True, help="降噪后视频的保存路径。")
    parser.add_argument('--model', type=str, required=True, help="预训练的SwinIR (.pth) 模型权重文件路径。")
    parser.add_argument('--gpu', type=int, default=0, help="要使用的GPU的ID。默认为0。")
    parser.add_argument('--noise', type=int, default=25, help="模型对应的噪声水平(15, 25, 50)。必须与模型文件匹配。默认为25。")
    
    args = parser.parse_args()
    denoise_video_patch_based(args.input, args.output, args.model, args.gpu, args.noise)