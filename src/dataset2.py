import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms

class UCF101DatasetFast(Dataset):
    """
    Custom dataset for loading UCF101 clips.
    """
    def __init__(self, annotation_file, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir 
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples = []
        
        # Loading paths and labels
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                # parts[0] is the video path, parts[1] is the label
                path_clean = os.path.splitext(parts[0])[0] 
                label = int(parts[1]) - 1 if len(parts) > 1 else 0
                full_path = os.path.join(self.root_dir, path_clean)
                
                if os.path.exists(full_path):
                    self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        
        # Reading files in the folder and sorting
        all_files = sorted(os.listdir(folder_path))
        file_names = all_files[:self.frames_per_clip] # Take the first N frames
        
        seed = torch.seed()

        frames = []
        for file_name in file_names:
            img_path = os.path.join(folder_path, file_name)
            img = Image.open(img_path).convert('RGB')

            torch.manual_seed(seed)
            
            # If a transform is defined, apply it to the single frame (PIL Image)
            if self.transform:
                img = self.transform(img)
            else:
                # Minimal fallback: convert to tensor if no transforms are provided
                img = transforms.ToTensor()(img)
            
            frames.append(img)
            
        # Handling empty folders (zero padding)
        if len(frames) == 0:
            return torch.zeros((3, self.frames_per_clip, 224, 224)), label

        # Stack frames: (T, C, H, W)
        video_tensor = torch.stack(frames)
        
        # Permute to PyTorch Video format: (C, T, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor, label


import os
from sklearn.model_selection import train_test_split
from torchvision import transforms

def get_datasets(config):
    """
    Creates and returns Train, Validation, and Test datasets.
    Implements group-based splitting to prevent data leakage and 
    includes a final integrity check for overlapping samples.
    """
    
    # DEFINE TRANSFORMATIONS
    # Training: Augmentation + Preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test: Preprocessing only (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # LOAD BASE SAMPLES
    # Load all training metadata to perform the split
    base_train_dataset = UCF101DatasetFast(
        annotation_file=config["TRAIN_ANNOTATION_FILE"],
        root_dir=config["DATA_ROOT"],
        frames_per_clip=config["FRAMES_PER_CLIP"]
    )

    # GROUP-BASED SPLITTING LOGIC
    # UCF101 videos are grouped (g01, g02, ...).  
    # videos from the same group stay together in either Train or Val.
    samples_by_group = {}
    for path, label in base_train_dataset.samples:
        # Extract folder name: v_ActionName_gXX_cYY
        folder_name = os.path.basename(path)
        parts = folder_name.split('_')
        
        # Unique Group ID = ActionName + GroupNumber (e.g., 'Archery_g01')
        # This handles cases where different actions might have the same group number
        group_id = f"{parts[1]}_{parts[2]}" 
        
        
        if group_id not in samples_by_group:
            samples_by_group[group_id] = {'samples': [], 'label': label}
        samples_by_group[group_id]['samples'].append((path, label))

    # Prepare unique groups and labels for stratified split
    unique_group_names = list(samples_by_group.keys())
    group_labels = [samples_by_group[g]['label'] for g in unique_group_names]

    # Split unique GROUPS instead of individual videos
    train_group_names, val_group_names = train_test_split(
        unique_group_names,
        test_size=0.2,
        shuffle=True,
        stratify=group_labels,
        random_state=42
    )


    # Reconstruct final sample lists from the selected groups
    train_samples = []
    for g_name in train_group_names:
        train_samples.extend(samples_by_group[g_name]['samples'])

    val_samples = []
    for g_name in val_group_names:
        val_samples.extend(samples_by_group[g_name]['samples'])

    # INITIALIZE DATASET INSTANCES
    train_ds = UCF101DatasetFast(
        annotation_file=config["TRAIN_ANNOTATION_FILE"],
        root_dir=config["DATA_ROOT"],
        transform=train_transform,
        frames_per_clip=config["FRAMES_PER_CLIP"]
    )
    train_ds.samples = train_samples

    val_ds = UCF101DatasetFast(
        annotation_file=config["TRAIN_ANNOTATION_FILE"],
        root_dir=config["DATA_ROOT"],
        transform=val_test_transform,
        frames_per_clip=config["FRAMES_PER_CLIP"]
    )
    val_ds.samples = val_samples

    test_ds = UCF101DatasetFast(
        annotation_file=config["TEST_ANNOTATION_FILE"],
        root_dir=config["DATA_ROOT"],
        transform=val_test_transform,
        frames_per_clip=config["FRAMES_PER_CLIP"]
    )

    # INTEGRITY CHECK (NON-OVERLAPPING VERIFICATION)
    train_paths = set([s[0] for s in train_ds.samples])
    val_paths   = set([s[0] for s in val_ds.samples])
    test_paths  = set([s[0] for s in test_ds.samples])

    overlap_train_val  = train_paths.intersection(val_paths)
    overlap_train_test = train_paths.intersection(test_paths)
    overlap_val_test   = val_paths.intersection(test_paths)

    print("-" * 50)
    print(f"GROUP-BASED SPLIT SUMMARY:")
    print(f"Train samples: {len(train_ds)}, {train_group_names} groups")
    print(f"Val samples:   {len(val_ds)}, {val_group_names} groups")
    print(f"Test samples:  {len(test_ds)}")
    print("-" * 50)
    
    # Verify overlaps
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("SUCCESS: No overlapping samples found between datasets.")
    else:
        if len(overlap_train_val) > 0:
            print(f"ERROR: {len(overlap_train_val)} samples overlap between Train and Val!")
        if len(overlap_train_test) > 0:
            print(f"ERROR: {len(overlap_train_test)} samples overlap between Train and Test!")
        if len(overlap_val_test) > 0:
            print(f"ERROR: {len(overlap_val_test)} samples overlap between Val and Test!")
    print("-" * 50)

    return train_ds, val_ds, test_ds