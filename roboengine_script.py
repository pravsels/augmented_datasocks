import os
import imageio
import numpy as np
from PIL import Image
import cv2
from robo_engine.infer_engine import RoboEngineRobotSegmentation
from robo_engine.infer_engine import RoboEngineObjectSegmentation
from robo_engine.infer_engine import RoboEngineAugmentation

def fix_segmentation_masks(frames, threshold=127):
    fixed_masks = []
    last_good_mask = None

    for frame in frames:
        # Convert to grayscale if it's BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Already single channel
            gray = frame
        
        # Threshold to ensure a clean binary mask
        _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Check if the mask has non-zero pixels
        if np.count_nonzero(binary_mask) > 0:
            # Non-empty mask
            final_mask = binary_mask
            last_good_mask = binary_mask.copy()
        else:
            # The mask is empty for this frame
            if last_good_mask is not None:
                # Reuse the last good mask
                final_mask = last_good_mask
            else:
                # We don't have a previous mask; keep the original frame 
                final_mask = np.zeros_like(binary_mask) 
        
        fixed_masks.append(final_mask)
    
    return fixed_masks

# Path to input video
video_filepath = "./original_videos/episode_000029 2.mp4"

# Initialize the segmentation and augmentation engines
engine_robo_seg = RoboEngineRobotSegmentation()
engine_obj_seg = RoboEngineObjectSegmentation()
engine_bg_aug = RoboEngineAugmentation(aug_method='engine')

# Read in the video frames
video = imageio.get_reader(video_filepath)
instruction = "pick the yellow sock and place it into the circular brown container"
image_np_list = [frame for frame in video]

# Generate segmentation masks for robot and object segmentation
robo_masks = engine_robo_seg.gen_video(image_np_list)
obj_masks = engine_obj_seg.gen_video(image_np_list, instruction)

obj_masks_fixed = fix_segmentation_masks(obj_masks)

# Combine the masks to create a full segmentation mask used for augmentation
masks = ((robo_masks + obj_masks) > 0).astype(np.float32)
masks_np_list = [mask for mask in masks]

# Generate an augmented video using the computed masks
aug_video = engine_bg_aug.gen_image_batch(image_np_list, masks_np_list)

# Function to write a list of frames to a video file
def save_video(frames, output_filepath, fps=24):
    writer = imageio.get_writer(output_filepath, fps=fps)
    for frame in frames:
        # Optionally, convert frames to uint8 if needed:
        if frame.dtype != np.uint8:
            # Scale the values to 0-255 and cast to uint8 (assuming frame values in [0,1] or similar)
            frame = (255 * frame).astype(np.uint8)
        writer.append_data(frame)
    writer.close()
    print("Video saved to:", output_filepath)

# Save the augmented video
aug_video_filepath = "./augmented_video.mp4"
save_video(aug_video, aug_video_filepath, fps=24)

# Save the robot segmentation video
robo_seg_video_filepath = "./robo_seg_video.mp4"
save_video(robo_masks, robo_seg_video_filepath, fps=24)

# Save the object segmentation video
obj_seg_video_filepath = "./object_seg_video.mp4"
save_video(obj_masks, obj_seg_video_filepath, fps=24)

obj_seg_fixed_video_filepath = "./object_seg_fixed_video.mp4"
save_video(obj_masks_fixed, obj_seg_fixed_video_filepath, fps=24)
