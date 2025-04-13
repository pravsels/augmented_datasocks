import os
import imageio
import numpy as np
from PIL import Image
import cv2
from robo_engine.infer_engine import RoboEngineAugmentation

def load_and_convert_masks(mask_video_path):
    """Load masks from a video file and convert to grayscale if needed."""
    reader = imageio.get_reader(mask_video_path)
    masks = []
    for frame in reader:
        # If the frame has 3 channels (color), convert it to grayscale.
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        masks.append(gray)
    return masks

def fix_segmentation_masks(frames, threshold=127):
    fixed_masks = []
    last_good_mask = None

    for frame in frames:
        # The mask should be single channel at this point.
        # Threshold to ensure a clean binary mask.
        _, binary_mask = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)

        # Check if the mask has non-zero pixels
        if np.count_nonzero(binary_mask) > 0:
            final_mask = binary_mask
            last_good_mask = binary_mask.copy()
        else:
            # Reuse the last good mask if available, or use an empty mask.
            final_mask = last_good_mask if last_good_mask is not None else np.zeros_like(binary_mask)
        
        fixed_masks.append(final_mask)
    
    return fixed_masks

# -------------------------------
# Paths to input videos:
# -------------------------------
video_filepath = "./original_videos/episode_000029 2.mp4"
robo_seg_mask_filepath = "./robo_seg_video.mp4"
obj_seg_mask_filepath = "./mask_video_fixed.mp4"

# -------------------------------
# Load the original video frames
# -------------------------------
video_reader = imageio.get_reader(video_filepath)
image_np_list = [frame for frame in video_reader]

# -------------------------------
# Load and convert the pre‑generated mask videos to grayscale
# -------------------------------
robo_masks = load_and_convert_masks(robo_seg_mask_filepath)
obj_masks = load_and_convert_masks(obj_seg_mask_filepath)

# -------------------------------
# Optionally fix the object masks so that any empty frames are replaced with the last valid mask.
# -------------------------------
obj_masks_fixed = fix_segmentation_masks(obj_masks)

# -------------------------------
# Combine the robot and fixed object masks to create a full segmentation mask.
# At this point, the masks are 2D arrays, so stacking them later in the augmentation engine will work correctly.
# -------------------------------
# Combine frame-by-frame using numpy arrays:
combined_masks = ((np.array(robo_masks) + np.array(obj_masks_fixed)) > 0).astype(np.float32)
masks_np_list = [mask for mask in combined_masks]

# -------------------------------
# Initialize the augmentation engine.
# -------------------------------
engine_bg_aug = RoboEngineAugmentation(aug_method='engine')

# Generate an augmented video using the original frames and the combined mask.
aug_video = engine_bg_aug.gen_image_batch(image_np_list, masks_np_list)

# -------------------------------
# Function to write frames to a video file.
# -------------------------------
def save_video(frames, output_filepath, fps=24):
    writer = imageio.get_writer(output_filepath, fps=fps)
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (255 * frame).astype(np.uint8)
        writer.append_data(frame)
    writer.close()
    print("Video saved to:", output_filepath)

# Save the augmented video.
aug_video_filepath = "./augmented_video.mp4"
save_video(aug_video, aug_video_filepath, fps=24)
