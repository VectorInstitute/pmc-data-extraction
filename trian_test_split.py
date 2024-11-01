import os
import shutil
from glob import glob
import numpy as np

# Define the root directory of your datasets and the target directory for train/val/test
root_dir = '/projects/multimodal/datasets/MedSAM'
target_dir = '/projects/DeepLesion/datasets/MedSAM'
train_ratio, val_ratio = 0.7, 0.15  # 70% train, 20% val, 10% test


# Ensure the target directory exists and create train, val, test subdirectories
os.makedirs(target_dir, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

# Process each modality
for modality in os.listdir(root_dir):
    modality_path = os.path.join(root_dir, modality)
    if os.path.isdir(modality_path):
        # Collect all npz files across datasets within the modality
        npz_files = glob(os.path.join(modality_path, '**/*.npz'), recursive=True)
        # Shuffle files to mix them up
        np.random.shuffle(npz_files)
        
        # Calculate split indices
        num_files = len(npz_files)
        train_end = int(train_ratio * num_files)
        val_end = train_end + int(val_ratio * num_files)
        
        # Split files
        train_files = npz_files[:train_end]
        val_files = npz_files[train_end:val_end]
        test_files = npz_files[val_end:]
        
        # Function to copy files to the target directory
        def copy_files(files, type):
            dest_folder = os.path.join(target_dir, type, modality)
            os.makedirs(dest_folder, exist_ok=True)
            for file in files:
                shutil.copy(file, dest_folder)
        
        # Copy files to respective folders
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

print("Files have been organized into train, val, and test directories under their respective modalities.")
