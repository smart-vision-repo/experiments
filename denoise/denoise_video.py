# -----------------------------------------------------------------------------
# denoise_video_optimized.py
#
# 描述:
#   使用批处理(Batching)和自动混合精度(AMP)来最大化GPU性能，
#   高速处理高分辨率视频的降噪任务。
# -----------------------------------------------------------------------------

import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
import math
import time

# --- Helper Functions (No changes needed here) ---
def tile_image(img, tile_size=128, overlap=32):
    h, w, c = img.shape
    tiles = []
    # Calculate padding
    pad_h = (tile_size - (h - overlap) % (tile_size - overlap)) % (tile_size - overlap)
    pad_w = (tile_size - (w - overlap) % (tile_size - overlap)) % (tile_size - overlap)
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    h_pad, w_pad, _ = img_padded.shape
    for i in range(0, h_pad - overlap, tile_size - overlap):
        for j in range(0, w_pad - overlap, tile_size - overlap):
            tiles.append(img_padded[i:i+tile_size, j:j+tile_size, :])
    return tiles, (h, w)

def untile_image(tiles, original_size, tile_size=128, overlap=32):
    h, w = original_size
    c = tiles[0].shape[2]
    
    # Calculate padding used in tiling
    pad_h = (tile_size - (h - overlap) % (tile_size - overlap)) % (tile_size - overlap)
    pad_w = (tile_size - (w - overlap) % (tile_size - overlap)) % (tile_size - overlap)
    h_pad, w_pad = h + pad_h, w + pad_w
    
    output_img_padded = np.zeros((h_pad, w_pad, c), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad, c), dtype=np.float32)
    
    idx = 0
    for i in range(0, h_pad - overlap, tile_size - overlap):
        for j in range(0, w_pad - overlap, tile_size - overlap):
            output_img_padded[i:i+tile_size, j:j+tile_size, :] += tiles[idx]
            count_map[i:i+tile_size, j:j+tile_size, :] += 1
            idx += 1
            
    output_img_padded /= count_map
    return np.clip(output_img_padded[0:h, 0:w, :], 0, 255).astype(np.uint8)


def denoise_video_optimized(input_path, output_path, model_path, gpu_id, noise_level, batch_size):
    """
    使用批处理和AMP对视频进行高速降噪。
    """
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # --- 视频I/O设置 (与之前相同) ---
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {frame_count} 帧。")

    # --- 加载SwinIR模型 (与之前相同) ---
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

    # --- 核心优化：逐帧进行批处理化分块推理 ---
    print(f"开始使用批处理(Batch Size={batch_size})和AMP高速处理视频...")
    total_time = 0
    with torch.no_grad():
        pbar = tqdm(range(frame_count), desc="降噪进度", unit="帧")
        for frame_idx in pbar:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tiles, original_size = tile_image(frame_rgb, tile_size=PATCH_SIZE, overlap=32)
            denoised_tiles = []
            
            # --- 关键优化点：将所有小块分批次处理 ---
            for i in range(0, len(tiles), batch_size):
                batch_tiles = tiles[i:i+batch_size]
                
                # 预处理批次
                batch_tensors = [torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0 for tile in batch_tiles]
                batch_tensors = torch.stack(batch_tensors).to(device)
                
                # --- 关键优化点：使用AMP (Automatic Mixed Precision) ---
                with torch.cuda.amp.autocast():
                    denoised_batch = model(batch_tensors)
                
                # 后处理批次
                denoised_batch_np = (denoised_batch.cpu().clamp(0, 1) * 255.0).permute(0, 2, 3, 1).numpy()
                denoised_tiles.extend([denoised_batch_np[j] for j in range(denoised_batch_np.shape[0])])

            denoised_frame_rgb = untile_image(denoised_tiles, original_size, tile_size=PATCH_SIZE, overlap=32)
            denoised_frame_bgr = cv2.cvtColor(denoised_frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(denoised_frame_bgr)
            
            end_time = time.time()
            frame_time = end_time - start_time
            total_time += frame_time
            # 更新进度条，显示实时速度
            pbar.set_postfix_str(f"{1/frame_time:.2f} 帧/秒")

    # --- 清理和收尾 ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("-" * 40)
    print(f"处理完成！总耗时: {total_time:.2f}秒, 平均速度: {frame_count/total_time:.2f} 帧/秒。")
    print(f"降噪后的视频已保存到: '{output_path}'")
    print("-" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用SwinIR模型对视频进行高性能降噪 (优化版)")
    parser.add_argument('--input', type=str, required=True, help="输入的带噪声视频文件路径。")
    parser.add_argument('--output', type=str, required=True, help="降噪后视频的保存路径。")
    parser.add_argument('--model', type=str, required=True, help="预训练的SwinIR (.pth) 模型权重文件路径。")
    parser.add_argument('--gpu', type=int, default=0, help="要使用的GPU的ID。默认为0。")
    parser.add_argument('--noise', type=int, default=25, help="模型对应的噪声水平(15, 25, 50)。默认为25。")
    parser.add_argument('--batch_size', type=int, default=32, help="一次性送入GPU处理的分块数量。根据显存调整。默认为32。")
    
    args = parser.parse_args()
    denoise_video_optimized(args.input, args.output, args.model, args.gpu, args.noise, args.batch_size)