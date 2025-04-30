"""
Functional script to take in a video and output the affordance map for each frame.
"""

import os.path as osp
import cv2
import numpy as np
import torch

from src.model import AffordanceModel
from src.utils.img_utils import *
from src.utils.argument_utils import get_yaml_config

import cv2
import numpy as np
from tqdm import tqdm 

def center_crop_square(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    return img[start_h:start_h+min_dim, start_w:start_w+min_dim]

def process_video(video_path, output_path, model, prompt_text, sample_interval=10, fps=30):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # List to store visualization frames
    vis_frames = []
    
    try:
        # Process frames
        frame_idx = 0
        with tqdm(total=total_frames) as pbar:
            while frame_idx < total_frames:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_idx % sample_interval == 0:
                    # Center crop to square
                    frame_square = center_crop_square(frame)
                    frame_square = resize_image(frame_square, target_shape=672) # resize to largest supported size
                    
                    # Convert BGR to RGB and normalize
                    frame_rgb = cv2.cvtColor(frame_square, cv2.COLOR_BGR2RGB)
                    frame_normalized = frame_rgb.astype(float) / 255.0
                    
                    # Get affordance map from model
                    affordance_map = model.inference(frame_normalized, prompt_text)
                    
                    # Create visualization
                    vis_frame = overlay_heatmap(frame_normalized, affordance_map)
                    vis_frames.append(vis_frame)
                
                frame_idx += 1
                pbar.update(1)

                # for debugging, only do 10 frames
                if frame_idx > 300:
                    break
        
        # Create video from collected frames
        if vis_frames:
            print(f"Creating video with {len(vis_frames)} frames")
            h, w = vis_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            for frame in vis_frames:
                # Convert RGB to BGR for writing
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
    
    finally:
        # Clean up
        cap.release()

# Example usage
if __name__ == "__main__":
    video_path = "/viscam/u/yihetang/unsup-affordance-inference/data/sample_data/sample_input_video.mp4"
    output_path = "/viscam/u/yihetang/unsup-affordance-inference/data/sample_data/sample_output_video.mp4"
    prompt_text = "handle"

    # Load model
    config = get_yaml_config("/viscam/u/yihetang/unsup-affordance-inference/checkpoints/gemini/config.yaml")
    model = AffordanceModel(config)
    
    process_video(video_path, output_path, model, prompt_text, sample_interval=10)