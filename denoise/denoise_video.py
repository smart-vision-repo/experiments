# denoise_video_debug.py
# (这是一个单GPU的脚本，专门用于调试黑屏问题)
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
import time

# --- Helper Functions (No changes) ---
def tile_image(img, tile_size=128, overlap=32):
    h, w, c = img.shape
    stride = tile_size - overlap
    pad_h = (stride - (h - overlap) % stride) % stride
    pad_w = (stride - (w - overlap) % stride) % stride
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h_pad, w_pad, _ = img_padded.shape
    tiles = []
    for i in range(0, h_pad - tile_size + 1, stride):
        for j in range(0, w_pad - tile_size + 1, stride):
            tiles.append(img_padded[i:i+tile_size, j:j+tile_size, :])
    return tiles, (h, w), (h_pad, w_pad)

def untile_image(tiles, original_size, padded_size, tile_size=128, overlap=32):
    h, w = original_size
    h_pad, w_pad = padded_size
    c = tiles[0].shape[2]
    stride = tile_size - overlap
    output_img_padded = np.zeros((h_pad, w_pad, c), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad, c), dtype=np.float32)
    idx = 0
    for i in range(0, h_pad - tile_size + 1, stride):
        for j in range(0, w_pad - tile_size + 1, stride):
            output_img_padded[i:i+tile_size, j:j+tile_size, :] += tiles[idx]
            count_map[i:i+tile_size, j:j+tile_size, :] += 1
            idx += 1
    output_img_padded /= np.maximum(count_map, 1)
    return np.clip(output_img_padded[0:h, 0:w, :], 0, 255).astype(np.uint8)


def denoise_video_debug(input_path, output_path, model_path, gpu_id, noise_level, batch_size):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # ... (视频I/O设置部分与之前相同) ...
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {frame_count} 帧。")

    # ... (模型加载部分与之前相同) ...
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

    print(f"开始调试处理，只处理前5帧...")
    with torch.no_grad():
        for frame_idx in range(5): # 只处理5帧用于调试
            ret, frame = cap.read()
            if not ret: break

            print(f"\n--- 正在处理第 {frame_idx + 1} 帧 ---")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tiles, original_size, padded_size = tile_image(frame_rgb, tile_size=PATCH_SIZE, overlap=32)
            denoised_tiles = []
            
            # 只处理第一批小块用于调试
            batch_tiles = tiles[0:batch_size]
            batch_tensors_list = [torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0 for tile in batch_tiles]
            batch_tensors = torch.stack(batch_tensors_list).to(device)
            
            with torch.cuda.amp.autocast():
                denoised_batch = model(batch_tensors)
            
            # ======================= DEBUG PROBE 1 =======================
            # 检查模型直接输出的数值范围
            print(f"Debug Probe 1: 模型输出张量的 Min={denoised_batch.min():.6f}, Max={denoised_batch.max():.6f}, Mean={denoised_batch.mean():.6f}")
            # =============================================================

            denoised_batch_np = (denoised_batch.cpu().clamp(0, 1) * 255.0).permute(0, 2, 3, 1).numpy()
            
            # ======================= DEBUG PROBE 2 =======================
            # 保存第一个降噪后的小块图像，看看它是不是黑的
            debug_tile_to_save = denoised_batch_np[0].astype(np.uint8)
            debug_tile_to_save_bgr = cv2.cvtColor(debug_tile_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"debug_tile_frame_{frame_idx + 1}.png", debug_tile_to_save_bgr)
            print(f"Debug Probe 2: 已保存第一个降噪小块到 'debug_tile_frame_{frame_idx + 1}.png'")
            # =============================================================

            # 为了快速调试，我们这里不再完整处理，直接跳出
            # 如果需要完整检查，请注释掉下面的 break
            # break 

            # (下面是完整的处理逻辑，暂时可以不用看)
            denoised_tiles.extend([denoised_batch_np[j] for j in range(denoised_batch_np.shape[0])])
            # 处理剩余的批次
            for i in range(batch_size, len(tiles), batch_size):
                batch_tiles = tiles[i:i+batch_size]
                batch_tensors_list = [torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0 for tile in batch_tiles]
                batch_tensors = torch.stack(batch_tensors_list).to(device)
                with torch.cuda.amp.autocast():
                    denoised_batch = model(batch_tensors)
                denoised_batch_np = (denoised_batch.cpu().clamp(0, 1) * 255.0).permute(0, 2, 3, 1).numpy()
                denoised_tiles.extend([denoised_batch_np[j] for j in range(denoised_batch_np.shape[0])])

            denoised_frame_rgb = untile_image(denoised_tiles, original_size, padded_size, tile_size=PATCH_SIZE, overlap=32)
            denoised_frame_bgr = cv2.cvtColor(denoised_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # ======================= DEBUG PROBE 3 =======================
            cv2.imwrite(f"debug_full_frame_{frame_idx + 1}.png", denoised_frame_bgr)
            print(f"Debug Probe 3: 已保存完整降噪帧到 'debug_full_frame_{frame_idx + 1}.png'")
            # =============================================================
            
            out.write(denoised_frame_bgr)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\n调试脚本运行结束。")

if __name__ == '__main__':
    # ... (命令行参数部分与之前相同) ...
    parser = argparse.ArgumentParser(description="用于调试黑屏问题的单GPU降噪脚本")
    parser.add_argument('--input', type=str, required=True, help="输入的带噪声视频文件路径。")
    parser.add_argument('--output', type=str, required=True, help="降噪后视频的保存路径。")
    parser.add_argument('--model', type=str, required=True, help="预训练的SwinIR (.pth) 模型权重文件路径。")
    parser.add_argument('--gpu', type=int, default=0, help="要使用的GPU的ID。默认为0。")
    parser.add__argument('--noise', type=int, default=25, help="模型对应的噪声水平(15, 25, 50)。默认为25。")
    parser.add_argument('--batch_size', type=int, default=64, help="一次性送入GPU处理的分块数量。默认为64。")
    args = parser.parse_args()
    denoise_video_debug(args.input, args.output, args.model, args.gpu, args.noise, args.batch_size)