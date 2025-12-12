import os
import json
from typing import List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class _BaseVideoFrameDataset(Dataset):
    """Utility base class that loads metadata shared across train/eval splits."""

    def __init__(self, processed_data_dir, transform=None, return_tensor=True):
        self.processed_data_dir = processed_data_dir
        self.transform = transform
        self.return_tensor = return_tensor

        with open(os.path.join(processed_data_dir, 'selected_videos_by_class.json'), 'r') as f:
            self.video_dict = json.load(f)

        self.selected_samples: List[Dict] = []
        for folder_name, videos_list in self.video_dict.items():
            for video_info in videos_list:
                video_dir = os.path.join(processed_data_dir, folder_name, video_info['id'])
                if os.path.isdir(video_dir):
                    self.selected_samples.append({
                        'video_info': video_info,
                        'video_dir': video_dir
                    })

    def _load_frame(self, frame_path):
        """Load a single frame, honoring optional transforms and output format."""
        with Image.open(frame_path) as img:
            img = img.convert('RGB')
            if self.transform is not None:
                transformed = self.transform(img)
            else:
                transformed = img

        if isinstance(transformed, torch.Tensor):
            frame = transformed
        else:
            frame = np.array(transformed)

        if self.return_tensor:
            if isinstance(frame, torch.Tensor):
                return frame
            return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        if isinstance(frame, torch.Tensor):
            return frame.permute(1, 2, 0).cpu().numpy()
        return frame


class VideoFrameTrainDataset(_BaseVideoFrameDataset):
    """Pairs frames 1-20 so each item predicts the next frame (1->2 ... 19->20)."""

    def __init__(self, processed_data_dir, transform=None, return_tensor=True):
        super().__init__(processed_data_dir, transform, return_tensor)
        self.frame_pairs: List[Dict] = []

        for sample in self.selected_samples:
            video_info = sample['video_info']
            video_dir = sample['video_dir']
            for frame_idx in range(19):  # 0->1 up to 18->19 (frames 1-20)
                input_path = os.path.join(video_dir, f'frame_{frame_idx:02d}.jpg')
                target_path = os.path.join(video_dir, f'frame_{frame_idx + 1:02d}.jpg')
                if os.path.exists(input_path) and os.path.exists(target_path):
                    self.frame_pairs.append({
                        'video_info': video_info,
                        'input_path': input_path,
                        'target_path': target_path,
                        'frame_idx': frame_idx,
                        'video_dir': video_dir
                    })

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        pair = self.frame_pairs[idx]
        input_frame = self._load_frame(pair['input_path'])
        target_frame = self._load_frame(pair['target_path'])

        return {
            'input': input_frame,
            'target': target_frame,
            'video_id': pair['video_info']['id'],
            'folder': pair['video_info']['folder'],
            'instruction': pair['video_info']['label'],
            'template': pair['video_info']['template'],
            'frame_index': pair['frame_idx'],
        }


class VideoFrameEvalDataset(_BaseVideoFrameDataset):
    """Uses frame 20 to predict frame 21 for each video clip."""

    def __init__(self, processed_data_dir, transform=None, return_tensor=True):
        super().__init__(processed_data_dir, transform, return_tensor)
        self.eval_pairs: List[Dict] = []

        for sample in self.selected_samples:
            video_info = sample['video_info']
            video_dir = sample['video_dir']
            input_path = os.path.join(video_dir, 'frame_19.jpg')
            target_path = os.path.join(video_dir, 'frame_20.jpg')
            if os.path.exists(input_path) and os.path.exists(target_path):
                self.eval_pairs.append({
                    'video_info': video_info,
                    'input_path': input_path,
                    'target_path': target_path,
                    'video_dir': video_dir
                })

    def __len__(self):
        return len(self.eval_pairs)

    def __getitem__(self, idx):
        pair = self.eval_pairs[idx]
        input_frame = self._load_frame(pair['input_path'])
        target_frame = self._load_frame(pair['target_path'])

        return {
            'input': input_frame,
            'target': target_frame,
            'video_id': pair['video_info']['id'],
            'folder': pair['video_info']['folder'],
            'instruction': pair['video_info']['label'],
            'template': pair['video_info']['template'],
            'frame_index': 19,
        }


if __name__ == '__main__':
    train_dataset = VideoFrameTrainDataset('../data/processed_dataset', return_tensor=False)
    eval_dataset = VideoFrameEvalDataset('../data/processed_dataset', return_tensor=True)

    train_sample = train_dataset[0]
    eval_sample = eval_dataset[0]

    print('=== Train Sample ===')
    print(f"Input shape : {np.array(train_sample['input']).shape}")
    print(f"Target shape: {np.array(train_sample['target']).shape}")
    print(f"Video ID    : {train_sample['video_id']}")
    print(f"Frame index : {train_sample['frame_index']}")

    print('\n=== Eval Sample ===')
    print(f"Input tensor shape : {eval_sample['input'].shape}")
    print(f"Target tensor shape: {eval_sample['target'].shape}")
    print(f"Video ID           : {eval_sample['video_id']}")