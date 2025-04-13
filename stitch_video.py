import numpy as np
import cv2

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
                # We don't have a previous mask; keep it empty or zero
                final_mask = np.zeros_like(binary_mask) 
        
        fixed_masks.append(final_mask)
    
    return fixed_masks


import cv2

cap = cv2.VideoCapture("object_seg_video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

fixed = fix_segmentation_masks(frames)

height, width = fixed[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('mask_video_fixed.mp4', fourcc, 30, (width, height), isColor=False)

for mask_frame in fixed:
    out.write(mask_frame)
out.release()


