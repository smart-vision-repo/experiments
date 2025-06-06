# -----------------------------------------------------------------------------
# denoise_video_pipeline_fixed.py
#
# 描述:
#   修复了多进程中 'tqdm' 未定义的NameError。
#   采用多进程并行流水线架构，最大化CPU与GPU的并行度，
#   实现对视频降噪任务的极致性能优化。
# -----------------------------------------------------------------------------
import cv2
import torch
import numpy as np
import argparse
import time
import multiprocessing as mp
from queue import Empty

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

# --- Process Functions ---
def reader_process(input_path, frame_queue, frame_count):
    cap = cv2.VideoCapture(input_path)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret: break
        frame_queue.put((i, frame))
    frame_queue.put(None)
    cap.release()

def worker_process(frame_queue, result_queue, model_path, gpu_id, noise_level, batch_size):
    device = torch.device(f'cuda:{gpu_id}')
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
    except Exception as e:
        print(f"[Worker] Model loading failed: {e}")
        return

    with torch.no_grad():
        while True:
            try:
                task = frame_queue.get(timeout=10)
                if task is None:
                    result_queue.put(None)
                    break
                
                idx, frame = task
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
                result_queue.put((idx, denoised_frame_bgr))

            except Empty:
                print("[Worker] Frame queue is empty. Exiting.")
                break
    
def writer_process(result_queue, output_path, fps, width, height, frame_count):
    """从结果队列获取帧并按顺序写入视频"""
    from tqdm import tqdm  # <--- 在这里添加导入
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    results_buffer = {}
    next_frame_idx = 0
    
    with tqdm(total=frame_count, desc="写入进度", unit="帧") as pbar:
        while next_frame_idx < frame_count:
            try:
                task = result_queue.get(timeout=20)
                if task is None: break
                
                idx, frame = task
                results_buffer[idx] = frame
                
                while next_frame_idx in results_buffer:
                    out.write(results_buffer.pop(next_frame_idx))
                    next_frame_idx += 1
                    pbar.update(1)

            except Empty:
                print("[Writer] Result queue timed out. Assuming processing is done.")
                break
                
    out.release()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="使用多进程流水线对视频进行极致性能降噪")
    parser.add_argument('--input', type=str, required=True, help="输入的带噪声视频文件路径。")
    parser.add_argument('--output', type=str, required=True, help="降噪后视频的保存路径。")
    parser.add_argument('--model', type=str, required=True, help="预训练的SwinIR (.pth) 模型权重文件路径。")
    parser.add_argument('--gpu', type=int, default=0, help="要使用的GPU的ID。默认为0。")
    parser.add_argument('--noise', type=int, default=25, help="模型对应的噪声水平(15, 25, 50)。默认为25。")
    parser.add_argument('--batch_size', type=int, default=64, help="一次性送入GPU处理的分块数量。默认为64。")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise ValueError("Cannot open input video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_queue = mp.Queue(maxsize=30)
    result_queue = mp.Queue(maxsize=30)

    reader = mp.Process(target=reader_process, args=(args.input, frame_queue, frame_count))
    worker = mp.Process(target=worker_process, args=(frame_queue, result_queue, args.model, args.gpu, args.noise, args.batch_size))
    writer = mp.Process(target=writer_process, args=(result_queue, args.output, fps, width, height, frame_count))
    
    print("启动多进程流水线...")
    start_time = time.time()
    
    reader.start()
    worker.start()
    writer.start()

    reader.join()
    worker.join()
    writer.join()
    
    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 40)
    if total_time > 0:
        print(f"全部处理完成！总耗时: {total_time:.2f}秒, 平均速度: {frame_count/total_time:.2f} 帧/秒。")
    print(f"降噪后的视频已保存到: '{args.output}'")
    print("-" * 40)