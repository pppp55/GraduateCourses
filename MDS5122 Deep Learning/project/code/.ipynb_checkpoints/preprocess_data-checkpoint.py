import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def select_videos_by_classes(train_json_path, target_classes_dict, videos_per_class=100):
    
    """Select and validate videos by class, ensuring adequate valid samples per category"""
    
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    selected_videos = {folder_name: [] for folder_name in target_classes_dict.values()}
    template_to_folder = target_classes_dict
    
    print("Selecting videos...")
    
    # Collect all videos with at least 21 frames
    all_candidate_videos = {folder_name: [] for folder_name in target_classes_dict.values()}
    
    for item in tqdm(train_data, desc="Collecting candidates"):
        if item['template'] in template_to_folder:
            folder_name = template_to_folder[item['template']]
            all_candidate_videos[folder_name].append({
                'id': item['id'],
                'label': item['label'],
                'template': item['template'],
                'folder': folder_name
            })
    
    # Collect sufficient valid videos for each specified class
    for folder_name, candidate_list in all_candidate_videos.items():
        print(f"\nValidating videos for {folder_name}...")
        valid_count = 0
        
        for video_info in tqdm(candidate_list, desc=folder_name):
            if valid_count >= videos_per_class:
                break
                
            video_id = video_info['id']
            video_path = f'../data/20bn-something-something-v2/{video_id}.webm'
            
            # Check if video file exists
            if not os.path.exists(video_path):
                continue
            
            # Check if video has sufficient frames
            frames = extract_frames_from_video(video_path, 21)
            if len(frames) >= 21:
                selected_videos[folder_name].append(video_info)
                valid_count += 1
        
        print(f"Found {valid_count} valid videos for {folder_name}")
        
        # Issue warning if any class lacks sufficient valid videos
        if valid_count < videos_per_class:
            print(f"Warning: Only found {valid_count} valid videos for {folder_name}, requested {videos_per_class}")
    
    return selected_videos

def extract_frames_from_video(video_path, num_frames=21):
    
    """Extract first N (21) frames from video"""
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (96, 96))
        frames.append(frame_resized)
    
    cap.release()
    return frames

def preprocess_dataset():
    # Specified class
    target_classes_dict = {
        'Pushing [something] from right to left': 'Pushing',
        'Dropping [something] onto [something]': 'Dropping', 
        'Covering [something] with [something]': 'Covering'
    }
    
    videos_per_class = 500  # 500 videos each class
    output_dir = '../data/processed_dataset'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select videos
    selected_videos_dict = select_videos_by_classes('../data/labels/train.json', target_classes_dict, videos_per_class)
    
    # Save selected video info by class
    with open(os.path.join(output_dir, 'selected_videos_by_class.json'), 'w') as f:
        json.dump(selected_videos_dict, f, indent=2)
    
    # Create folders for each class and process videos
    print("\nExtracting frames by class...")
    
    for folder_name, videos_list in selected_videos_dict.items():
        class_dir = os.path.join(output_dir, folder_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Get corresponding template name
        template_name = [k for k, v in target_classes_dict.items() if v == folder_name][0]
        print(f"Processing class: {template_name} -> {folder_name}")
        
        # Process all videos for current class
        processed_count = 0
        for video_info in tqdm(videos_list, desc=f"{folder_name}"):
            video_id = video_info['id']
            video_path = f'../data/20bn-something-something-v2/{video_id}.webm'
            
            # Re-extract frames to ensure consistency
            frames = extract_frames_from_video(video_path, 21)
            
            if len(frames) < 21:
                print(f"Warning: Video {video_id} now has only {len(frames)} frames, skipping...")
                continue
            
            # Create directory for each video under its class folder
            video_dir = os.path.join(class_dir, video_id)
            os.makedirs(video_dir, exist_ok=True)
            
            # Save frames
            for i, frame in enumerate(frames):
                np.save(os.path.join(video_dir, f'frame_{i:02d}.npy'), frame)
            
            # Save video metadata
            video_info_path = os.path.join(video_dir, 'metadata.json')
            with open(video_info_path, 'w') as f:
                json.dump(video_info, f)
            
            processed_count += 1
        
        print(f"Completed {folder_name}: {processed_count} videos processed")
    
    # Generate processing summary
    print("\n=== Processing Summary ===")
    total_processed = 0
    for folder_name, videos_list in selected_videos_dict.items():
        class_dir = os.path.join(output_dir, folder_name)
        if os.path.exists(class_dir):
            processed_count = len([name for name in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, name))])
            total_processed += processed_count
            print(f"{folder_name}: {processed_count} videos")
    
    print(f"Total videos processed: {total_processed}")
    print("Preprocessing completed!")

if __name__ == '__main__':
    preprocess_dataset()