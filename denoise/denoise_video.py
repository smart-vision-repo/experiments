# denoise_video_optimized.py
# 优化点：使用多GPU + autocast 混合精度（修复 DataParallel 多进程冲突）

import cv2
import torch
import numpy as np
import argparse
import time
import multiprocessing as mp
from queue import Empty
from torch.cuda.amp import autocast

# --- Helper Functions ---
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

# --- Frame Reader ---
def reader_process(input_path, frame_queue, frame_count):
    cap = cv2.VideoCapture(input_path)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((i, frame))
    cap.release()
    frame_queue.put(None)

# --- Worker with Model ---
def worker_process(frame_queue, result_queue, model_path, noise_level):
    device = torch.device('cuda:0')
    from model_architecture import SwinIR
    model = SwinIR(
        upscale=1, img_size=(128, 128), window_size=8, img_range=1.,
        depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6]*6,
        mlp_ratio=2, upsampler='', resi_connection='1conv',
        task='color_dn', noise=noise_level
    )
    pretrained = torch.load(model_path, map_location='cpu')
    param_key = 'params' if 'params' in pretrained else next(iter(pretrained))
    model.load_state_dict(pretrained[param_key], strict=True)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    while True:
        item = frame_queue.get()
        if item is None:
            break
        idx, frame = item
        frame = frame.astype(np.float32) / 255.
        tiles, orig_size, pad_size = tile_image(frame)
        input_tiles = torch.stack([
            torch.from_numpy(t.transpose(2, 0, 1)).float() for t in tiles
        ])
        input_tiles = input_tiles.to(device)

        with torch.no_grad():
            with autocast():
                output_tiles = model(input_tiles)

        output_tiles = output_tiles.cpu().numpy()
        output_tiles = [
            (np.clip(t.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            for t in output_tiles
        ]
        result = untile_image(output_tiles, orig_size, pad_size)
        result_queue.put((idx, result))

    result_queue.put(None)

# --- Writer ---
from tqdm import tqdm

def writer_process(output_path, result_queue, frame_count, fps, resolution):
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
    results = {}
    received = 0
    pbar = tqdm(total=frame_count, desc="Writing frames", unit="frame")
    while received < frame_count:
        item = result_queue.get()
        if item is None:
            break
        idx, frame = item
        results[idx] = frame
        while received in results:
                        writer.write(results.pop(received))
            received += 1
            pbar.update(1)
        pbar.close()
    writer.release()

# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--noise_level', type=int, default=25)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_queue = mp.Queue(maxsize=20)
    result_queue = mp.Queue(maxsize=20)

    reader = mp.Process(target=reader_process, args=(args.input, frame_queue, frame_count))
    reader.start()

    worker = mp.Process(target=worker_process, args=(frame_queue, result_queue, args.model_path, args.noise_level))
    worker.start()

    writer = mp.Process(target=writer_process, args=(args.output, result_queue, frame_count, fps, (width, height)))
    writer.start()

    reader.join()
    worker.join()
    writer.join()
