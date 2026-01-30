import os
import cv2
from matplotlib import transforms
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from PIL import Image
from torch.utils.data import Dataset
import torch
import os

class UCF101Dataset(Dataset):
    def __init__(self, data_root, annotation_file, class_ind_file, resize_height=224, resize_width=224, frames_per_clip=16, transform=None):
        """
        Args:
            data_root (string): Path to the folder with videos (UCF-101).
            annotation_file (string): Path to the split .txt file.
            class_ind_file (string): Path to the classInd.txt file.
            frames_per_clip (int): Number of frames to extract per video (default 16).
            transform (callable, optional): Transformations.
        """
        self.data_root = data_root
        self.frames_per_clip = frames_per_clip
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transform = transform 
        
        # Load class map (from string to number)
        self.class_to_idx = self._load_class_ind(class_ind_file)
        
        # Load list of videos and labels
        self.video_paths, self.labels = self._load_annotation(annotation_file)

    def _load_class_ind(self, path):
        
        class_to_idx = {}
        with open(path, 'r') as f:
            for line in f:
                idx, class_name = line.strip().split()
                # resize index to be zero-based
                class_to_idx[class_name] = int(idx) - 1 
        return class_to_idx

    def _load_annotation(self, path):
        video_paths = []
        labels = []
        with open(path, 'r') as f:
            for line in f:
                # EXAMPLE: "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1"
                # extract relative path and label
                parts = line.strip().split()
                rel_path = parts[0]
                
                # Extract class name from path
                class_name = rel_path.split('/')[0]
                
                # safety check
                if class_name in self.class_to_idx:
                    # Store full path and label in the respective lists
                    video_paths.append(os.path.join(self.data_root, rel_path))
                    labels.append(self.class_to_idx[class_name])
                    
        return video_paths, labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Read the video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame for CNN (ResNet)
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            frames.append(frame)
            
        cap.release()
        
        # Handle corrupted/empty videos
        if len(frames) == 0:
            # Return Black Video
            print(f"Warning: Video {video_path} is corrupted or empty. Returning black video.")
            return torch.zeros((3, self.frames_per_clip, 112, 112)), label

        # Convert the list into a numpy array for advanced indexing
        frames = np.array(frames) # Shape: (Total_Frames_Original, H, W, C)
        total_frames = len(frames)

        # RESAMPLING
        # 
        # CASE A: Longer Video
        # If the video has more frames than needed, we can sample uniformly
        
        # CASE B: Shorter Video
        # Stretching (repeats frames)
        
        indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)
        
        # FRAME SELECTION (numpy i love you)
        frames = frames[indices] # Shape: (frames_per_clip, H, W, C)

        # FINAL TRANSFORMATIONS
        if self.transform:
            frames = self.transform(frames)
        else:
             frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0

        return frames, label
    
def get_datasets(config):
    

    full_train_dataset = UCF101Dataset(
        data_root=config["DATA_ROOT"],
        annotation_file=config["TRAIN_ANNOTATION_FILE"],
        class_ind_file=config["CLASS_IND_FILE"],
        frames_per_clip=config["FRAMES_PER_CLIP"],
        resize_height=config["RESIZE_HEIGHT"],
        resize_width=config["RESIZE_WIDTH"]
    )
    

    test_dataset = UCF101Dataset(
        data_root=config["DATA_ROOT"],
        annotation_file=config["TEST_ANNOTATION_FILE"],
        class_ind_file=config["CLASS_IND_FILE"],
        frames_per_clip=config["FRAMES_PER_CLIP"],
        resize_height=config["RESIZE_HEIGHT"],
        resize_width=config["RESIZE_WIDTH"]
    )
    
 
    targets = full_train_dataset.labels
    

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train_dataset)),
        test_size=0.2,          # 20% validation
        shuffle=True,
        stratify=targets,      
        random_state=42         
    )
    
    
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)
    
    print("-" * 30)
    print(f"Dataset Split Summary:")
    print(f"Train: {len(train_subset)}")
    print(f"Val:   {len(val_subset)}")
    print(f"Test:  {len(test_dataset)}")
    print("-" * 30)
    
    return train_subset, val_subset, test_dataset


