# -----------------------------------------------------------------------------
# denoise_video_final_fixed.py
#
# 描述:
#   修复了分块逻辑中的边界条件，确保所有处理块尺寸一致，
#   解决了 `torch.stack` 的尺寸不匹配错误。
# -----------------------------------------------------------------------------

import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
import time

# ---------------------------------
# Helper Functions for Tiling (Corrected)
# ---------------------------------
def tile_image(img, tile_size=128, overlap=32):
    """将图像分割成重叠的小块，确保所有块尺寸相同"""
    h, w, c = img.shape
    stride = tile_size - overlap
    
    # 计算需要填充的尺寸，以确保图像尺寸是步长的整数倍
    pad_h = (stride - (h - overlap) % stride) % stride
    pad_w = (stride - (w - overlap) % stride) % stride
    
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    h_pad, w_pad, _ = img_padded.shape
    tiles = []
    
    # 修正循环范围，确保不会切出碎块
    for i in range(0, h_pad - tile_size + 1, stride):
        for j in range(0, w_pad - tile_size + 1, stride):
            tiles.append(img_padded[i:i+tile_size, j:j+tile_size, :])
            
    return tiles, (h, w), (h_pad, w_pad)

def untile_image(tiles, original_size, padded_size, tile_size=128, overlap=32):
    """将处理后的小块重新拼接成完整图像"""
    h, w = original_size
    h_pad, w_pad = padded_size
    c = tiles[0].shape[2]
    stride = tile_size - overlap
    
    output_img_padded = np.zeros((h_pad, w_pad, c), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad, c), dtype=np.float32)
    
    idx = 0
    # 修正循环范围，与tile_image保持一致
    for i in range(0, h_pad - tile_size + 1, stride):
        for j in range(0, w_pad - tile_size + 1, stride):
            output_img_padded[i:i+tile_size, j:j+tile_size, :] += tiles[idx]
            count_map[i:i+tile_size, j:j+tile_size, :] += 1
            idx += 1
            
    # 处理重叠区域的平均值
    output_img_padded /= np.maximum(count_map, 1) # 避免除以零
    
    # 返回裁剪掉填充部分的原始尺寸图像
    return np.clip(output_img_padded[0:h, 0:w, :], 0, 255).astype(np.uint8)


def denoise_video_optimized(input_path, output_path, model_path, gpu_id, noise_level, batch_size):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {frame_count} 帧。")

    try:
        from model_architecture import SwinIR as ModelClass
        PATCH_SIZE = 128
        model = ModelClass(upscale=1, img_size=(PATCH_SIZE, PATCH_SIZE),
                           window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                           embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                           mlp_ratio=2, upsampler='', resi_connection='1conv',
                           task='color_dn', noise=noise_level)
        pretrained_model = torch.load(model_path, map_location='cpu')
        param_key = 'params'
        if param_key not in pretrained_model: param_key = next(iter(pretrained_model))
        model.load_state_dict(pretrained_model[param_key], strict=True)
        model.to(device)
        model.eval()
        print("SwinIR模型已成功加载到GPU。")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    print(f"开始使用批处理(Batch Size={batch_size})和AMP高速处理视频...")
    total_time = 0
    with torch.no_grad():
        pbar = tqdm(range(frame_count), desc="降噪进度", unit="帧")
        for frame_idx in pbar:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tiles, original_size, padded_size = tile_image(frame_rgb, tile_size=PATCH_SIZE, overlap=32)
            denoised_tiles = []
            
            for i in range(0, len(tiles), batch_size):
                batch_tiles = tiles[i:i+batch_size]
                batch_tensors_list = [torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0 for tile in batch_tiles]
                batch_tensors = torch.stack(batch_tensors_list).to(device)
                
                with torch.cuda.amp.autocast():
                    denoised_batch = model(batch_tensors)
                
                denoised_batch_np = (denoised_batch.cpu().clamp(0, 1) * 255.0).permute(0, 2, 3, 1).numpy()
                denoised_tiles.extend([denoised_batch_np[j] for j in range(denoised_batch_np.shape[0])])

            denoised_frame_rgb = untile_image(denoised_tiles, original_size, padded_size, tile_size=PATCH_SIZE, overlap=32)
            denoised_frame_bgr = cv2.cvtColor(denoised_frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(denoised_frame_bgr)
            
            end_time = time.time()
            frame_time = end_time - start_time
            total_time += frame_time
            if frame_time > 0:
                pbar.set_postfix_str(f"{1/frame_time:.2f} 帧/秒")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("-" * 40)
    if total_time > 0:
      print(f"处理完成！总耗时: {total_time:.2f}秒, 平均速度: {frame_count/total_time:.2f} 帧/秒。")
    print(f"降噪后的视频已保存到: '{output_path}'")
    print("-" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用SwinIR模型对视频进行高性能降噪 (最终修复版)")
    # ... (命令行参数部分与之前完全相同) ...
    parser.add_argument('--input', type=str, required=True, help="输入的带噪声视频文件路径。")
    parser.add_argument('--output', type=str, required=True, help="降噪后视频的保存路径。")
    parser.add_argument('--model', type=str, required=True, help="预训练的SwinIR (.pth) 模型权重文件路径。")
    parser.add_argument('--gpu', type=int, default=0, help="要使用的GPU的ID。默认为0。")
    parser.add_argument('--noise', type=int, default=25, help="模型对应的噪声水平(15, 25, 50)。默认为25。")
    parser.add_argument('--batch_size', type=int, default=32, help="一次性送入GPU处理的分块数量。根据显存调整。默认为32。")
    
    args = parser.parse_args()
    denoise_video_optimized(args.input, args.output, args.model, args.gpu, args.noise, args.batch_size)