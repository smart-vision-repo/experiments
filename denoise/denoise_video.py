import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor
from model_architecture import SwinIR

# ------------------------ Configuration ------------------------
input_video_path = "input.mp4"
output_video_path = "output_denoised.mp4"
model_path = "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
device = "cuda"
batch_size = 8
num_gpus = torch.cuda.device_count()
num_workers = 4
tile_pad = 0
half_precision = True
# ---------------------------------------------------------------

# ------------------------ Model Setup --------------------------
model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8, img_range=1., 
               depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], 
               mlp_ratio=2, upsampler='', resi_connection='1conv')
pretrained_model = torch.load(model_path)
model.load_state_dict(pretrained_model, strict=True)
model.eval()

if num_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)
# ---------------------------------------------------------------

def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps

def write_video(frames, fps, path):
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

def preprocess(frames):
    return torch.stack([to_tensor(f).unsqueeze(0) for f in frames]).squeeze(1)

def denoise_batch(batch):
    with torch.no_grad():
        batch = batch.to(device)
        with autocast(enabled=half_precision):
            output = model(batch)
        return output.clamp_(0, 1).cpu()

def process_all_frames(frames):
    processed = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(0, len(frames), batch_size), desc="Denoising"):
            batch_frames = frames[i:i+batch_size]
            batch_tensor = preprocess(batch_frames)
            output_tensor = denoise_batch(batch_tensor)
            outputs = [to_pil_image(t) for t in output_tensor]
            results = [np.array(img) for img in outputs]
            processed.extend(results)
    return processed

def main():
    assert Path(input_video_path).exists(), f"Input video {input_video_path} not found."
    print(f"Reading video: {input_video_path}")
    frames, fps = read_video_frames(input_video_path)
    print(f"Total {len(frames)} frames. FPS: {fps}")
    results = process_all_frames(frames)
    print(f"Writing output video: {output_video_path}")
    write_video(results, fps, output_video_path)
    print("Done.")

if __name__ == "__main__":
    main()