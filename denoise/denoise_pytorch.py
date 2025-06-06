import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm 

# This is a placeholder. You will get this from the model's repository.
from model_architecture import DenoiseModel #<-- IMPORTANT: REPLACE THIS

def denoise_video(input_path, output_path, model_path, gpu_id):
    """
    Denoises a video using a PyTorch model on a specified GPU.
    """
    # 1. Setup Device and Model
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Using device: {device}")

    try:
        # Load the model architecture and weights
        model = DenoiseModel()  #<-- REPLACE with the actual model class
        model.load_state_dict(torch.load(model_path, map_location='cpu')['params']) # Adjust '.['params']' based on the model file structure
        model.to(device)
        model.eval() # Set to evaluation mode (important!)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'model_architecture.py' and your model class name are correct.")
        return

    # 2. Setup Video I/O
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {frame_count} frames from '{input_path}'...")

    # 3. Process Frames
    with torch.no_grad(): # Speeds up inference, saves memory
        for _ in tqdm(range(frame_count), desc="Denoising"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from OpenCV's BGR (H, W, C) to PyTorch's RGB Tensor (1, C, H, W)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = (torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
            
            # Denoise on the GPU
            denoised_tensor = model(img_tensor)

            # Convert back to OpenCV format
            denoised_frame_rgb = (denoised_tensor.squeeze(0).cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            denoised_frame_bgr = cv2.cvtColor(denoised_frame_rgb, cv2.COLOR_RGB2BGR)

            out.write(denoised_frame_bgr)

    # 4. Release resources
    cap.release()
    out.release()
    print(f"Denoising complete. Output saved to '{output_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Denoising using PyTorch")
    parser.add_argument('--input', type=str, required=True, help="Path to the noisy input video.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the denoised output video.")
    parser.add_argument('--model', type=str, required=True, help="Path to the pre-trained .pth model file.")
    parser.add_argument('--gpu', type=int, default=0, help="ID of the GPU to use (e.g., 0, 1, 2, 3).")
    
    args = parser.parse_args()

    denoise_video(args.input, args.output, args.model, args.gpu)
