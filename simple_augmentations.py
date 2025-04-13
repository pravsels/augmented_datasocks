import os
from pathlib import Path
import cv2
import torch
import kornia.augmentation as K
from tqdm import tqdm
import numpy as np

# --- Setup Directories ---
original_videos_dir = Path("camera_2")
original_videos_dir.mkdir(exist_ok=True)
augmented_videos_dir = Path("augmented_videos_camera_2")
augmented_videos_dir.mkdir(exist_ok=True)

# --- Select Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU.")

# --- Define the Augmentation Pipeline ---
# Use intensity-based (non-cropping) transforms; same_on_batch=True forces every frame to use the same parameters.
augmentation_pipeline = K.AugmentationSequential(
    K.RandomHorizontalFlip(same_on_batch=True, p=0.5),
    K.ColorJiggle(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5,
        same_on_batch=True,
        p=1.0
    ),
    # K.RandomClahe(
    #     same_on_batch=True,
    #     p=0.3,
    # ),
    # K.RandomEqualize(
    #     same_on_batch=True,
    #     p=0.5,
    # ),
    # K.RandomDissolving(
    #     same_on_batch=True,
    #     p=0.5
    # ),
    K.RandomGamma(
        same_on_batch=True,
        p=0.1,
    ),
    # K.RandomMotionBlur(
    #     3, 35., 0.5, p=0.1
    # ),
    K.RandomPlanckianJitter(
        mode='CIED',
        same_on_batch=True,
        p=0.5
    ),
    # K.RandomPlasmaBrightness(
    #     same_on_batch=True,
    #     p=0.3,
    # ),
    K.RandomPlasmaShadow(
        same_on_batch=True,
        p=0.3,
    ),
    # K.RandomPlasmaContrast(
    #     same_on_batch=True,
    #     p=0.3,
    # ),
    # K.RandomPosterize(
    #     same_on_batch=True,
    #     p=0.3,
    # ),
    # K.RandomRain(
    #     drop_height=(1,2),drop_width=(1,2),number_of_drops=(1,1),
    #     same_on_batch=True,
    #     p=1.,
    # ),
    K.RandomPerspective(
        same_on_batch=True,
        p=0.1
    ),
    K.RandomThinPlateSpline(
        same_on_batch=True,
        p=0.1
    ),
    K.RandomGaussianBlur(
        kernel_size=(3, 3),
        sigma=(0.1, 1.0),
        same_on_batch=True,
        p=0.5
    ),
    data_keys=["input"]
).to(device)

# --- Function to Process a Single Video ---
def process_video(input_path: Path, output_path: Path, batch_size: int = 16):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing '{input_path.name}': {width}x{height} @ {fps} fps, {total_frames} frames total.")
    
    # Define VideoWriter.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # --- Lock the Transformation Parameters on the First Frame ---
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Error reading first frame from: {input_path}")
    
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # (C, H, W)
    frame_tensor = frame_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    
    augmented_first = augmentation_pipeline(frame_tensor)
    params = augmentation_pipeline._params  # Lock these parameters for the entire video.
    
    # Write the first frame.
    aug_first_cpu = augmented_first.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    aug_first_bgr = cv2.cvtColor((aug_first_cpu * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.write(aug_first_bgr)
    
    frames_processed = 1
    frame_batch = []
    
    with tqdm(total=total_frames - 1, desc=f"Processing {input_path.name}") as pbar:
        while frames_processed < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame BGR -> RGB and to tensor.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_batch.append(frame_tensor)
            frames_processed += 1
            
            if len(frame_batch) == batch_size or frames_processed == total_frames:
                batch_tensor = torch.stack(frame_batch, dim=0).to(device)
                augmented_batch = augmentation_pipeline(batch_tensor, params=params)
                for img_tensor in augmented_batch:
                    img_np = (img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                frame_batch = []
                pbar.update(batch_size)
    
    cap.release()
    out.release()
    print(f"Augmented video saved as: {output_path.name}")


import argparse

# --- Parse Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Batch video augmentation using Kornia")
parser.add_argument("--runs_per_vid", type=int, default=1,
                    help="Number of augmentation runs per video (each run samples new augmentation parameters)")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Number of frames to process simultaneously")
args = parser.parse_args()

for video_file in original_videos_dir.glob("*.mp4"):
    for run in range(args.runs_per_vid):
        output_filename = f"{video_file.stem}_augmented_run{run + 1}.mp4"
        output_file = augmented_videos_dir / output_filename
        process_video(video_file, output_file, batch_size=args.batch_size)

