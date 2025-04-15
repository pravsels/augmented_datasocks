import os
import shutil
import re
import json
from pathlib import Path

# User-configurable parameters
# Note: Update these paths to match your actual setup
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = current_dir  # Using the current directory as source
augmented_dir_camera_1 = os.path.join(current_dir, "augmented_videos_camera_1")
augmented_dir_camera_2 = os.path.join(current_dir, "augmented_videos_camera_2")
output_dir = os.path.join(current_dir, "NewDataset")
num_copies = 10  # Number of augmented copies per original video

# Map camera folders to their augmentation directories
camera_mapping = {
    "observation.images.main": augmented_dir_camera_1,
    "observation.images.secondary_0": augmented_dir_camera_2
}

def create_directory_structure():
    """Create the necessary directory structure for the new dataset."""
    print(f"Creating output directories in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    
    # Create proper video directory structure
    for camera_key in camera_mapping.keys():
        os.makedirs(os.path.join(output_dir, "videos", "chunk-000", camera_key), exist_ok=True)
    
    print("Directory structure created successfully.")

def copy_metadata_files():
    """Copy all metadata files from source to destination."""
    meta_files = ["info.json", "stats.json", "tasks.jsonl"]
    for file in meta_files:
        src_file = os.path.join(source_dir, "meta", file)
        if os.path.exists(src_file):
            dst_file = os.path.join(output_dir, "meta", file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied metadata file: {file}")

def get_episode_files():
    """Get all episode files from the dataset sorted by index."""
    data_dir = os.path.join(source_dir, "data", "chunk-000")
    episode_files = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and filename.startswith("episode_"):
            # Extract episode number from filename
            match = re.search(r"episode_(\d+)\.parquet", filename)
            if match:
                episode_idx = int(match.group(1))
                episode_files.append((episode_idx, filename))
    
    # Sort episodes by index
    episode_files.sort()
    print(f"Found {len(episode_files)} episodes in the dataset")
    return episode_files

def get_episode_length(episode_idx):
    """Get the length of an episode from episodes.jsonl."""
    src_episodes_jsonl = os.path.join(source_dir, "meta", "episodes.jsonl")
    if os.path.exists(src_episodes_jsonl):
        with open(src_episodes_jsonl, 'r') as f:
            for line in f:
                episode_data = json.loads(line.strip())
                if episode_data["episode_index"] == episode_idx:
                    return episode_data["length"]
    return 0  # Default if not found

def process_dataset():
    """Process the dataset by copying original and augmented videos with their parquet files."""
    episode_files = get_episode_files()
    processed_episodes = []
    next_idx = 0
    
    for original_idx, parquet_file in episode_files:
        print(f"Processing episode {original_idx:06d}")
        episode_length = get_episode_length(original_idx)
        
        # 1. First copy the original episode
        src_parquet = os.path.join(source_dir, "data", "chunk-000", parquet_file)
        dst_parquet = os.path.join(output_dir, "data", "chunk-000", f"episode_{next_idx:06d}.parquet")
        
        # Copy the parquet file
        shutil.copy2(src_parquet, dst_parquet)
        
        # Add to processed episodes list
        processed_episodes.append({
            "episode_index": next_idx,
            "tasks": ["pick up the socks and put it in the bowl"],
            "length": episode_length
        })
        
        # Copy the original videos
        for camera_key in camera_mapping.keys():
            src_video = os.path.join(source_dir, "videos", "chunk-000", camera_key, f"episode_{original_idx:06d}.mp4")
            dst_video = os.path.join(output_dir, "videos", "chunk-000", camera_key, f"episode_{next_idx:06d}.mp4")
            
            if os.path.exists(src_video):
                shutil.copy2(src_video, dst_video)
                print(f"  Copied original video for {camera_key}")
            else:
                print(f"  Warning: Original video not found at {src_video}")
        
        orig_episode_idx = next_idx
        next_idx += 1
        
        # 2. Now add the augmented copies
        print(f"  Adding {num_copies} augmented versions for episode {original_idx:06d}")
        
        for aug_idx in range(1, num_copies + 1):
            # Use a new episode index for each augmented version
            new_idx = next_idx
            
            # Copy the same parquet file with a new index
            dst_parquet = os.path.join(output_dir, "data", "chunk-000", f"episode_{new_idx:06d}.parquet")
            shutil.copy2(src_parquet, dst_parquet)
            
            # Add to processed episodes
            processed_episodes.append({
                "episode_index": new_idx,
                "tasks": ["pick up the socks and put it in the bowl"],
                "length": episode_length
            })
            
            # Copy the augmented videos
            for camera_key, aug_dir in camera_mapping.items():
                aug_video = os.path.join(aug_dir, f"episode_{original_idx:06d}_augmented_run{aug_idx}.mp4")
                dst_video = os.path.join(output_dir, "videos", "chunk-000", camera_key, f"episode_{new_idx:06d}.mp4")
                
                if os.path.exists(aug_video):
                    shutil.copy2(aug_video, dst_video)
                    print(f"    Copied augmented video {aug_idx} for {camera_key}")
                else:
                    print(f"    Warning: Augmented video not found at {aug_video}")
                    # Fall back to original if augmented not found
                    src_video = os.path.join(source_dir, "videos", "chunk-000", camera_key, f"episode_{original_idx:06d}.mp4")
                    if os.path.exists(src_video):
                        shutil.copy2(src_video, dst_video)
                        print(f"      Using original video as fallback for {camera_key}")
            
            next_idx += 1
    
    # Write updated episodes.jsonl
    with open(os.path.join(output_dir, "meta", "episodes.jsonl"), 'w') as f:
        for episode in processed_episodes:
            json.dump(episode, f)
            f.write('\n')
    
    # Update info.json
    update_info_json(len(processed_episodes))
    
    print(f"\nProcessing complete! Created dataset with {next_idx} total episodes.")
    print(f"Final dataset is at: {output_dir}")
    
    # Print summary
    orig_count = len(episode_files)
    aug_count = next_idx - orig_count
    print(f"\nDataset Summary:")
    print(f"  Original episodes: {orig_count}")
    print(f"  Augmented episodes: {aug_count}")
    print(f"  Total episodes: {next_idx}")

def update_info_json(total_episodes):
    """Update the info.json file with the new episode count."""
    info_path = os.path.join(output_dir, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info_data = json.load(f)
        
        info_data["total_episodes"] = total_episodes
        info_data["splits"] = {"train": f"0:{total_episodes}"}
        
        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=4)
        
        print(f"Updated info.json with total_episodes={total_episodes}")

def main():
    print("Starting dataset processing...")
    create_directory_structure()
    copy_metadata_files()
    process_dataset()

if __name__ == "__main__":
    main()
