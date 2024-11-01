import os
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
import argparse
import random

def convert_npz_to_npy(file_info):
    npz_path, npy_base_dir = file_info
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    imgs = npz["imgs"]
    gts = npz["gts"]
    
    if len(gts.shape) > 2: ## 3D image
        return

    # Determine the output path based on the input path and base directory
    relative_path = os.path.relpath(npz_path, start=args.npz_dir)
    base_name = os.path.splitext(relative_path)[0]
    img_npy_path = os.path.join(npy_base_dir, base_name + ".npy")
    # gt_npy_path = os.path.join(npy_base_dir, "gts", base_name + ".npy")

    # Ensure directories exist
    os.makedirs(os.path.dirname(img_npy_path), exist_ok=True)
    # os.makedirs(os.path.dirname(gt_npy_path), exist_ok=True)

    # Process and save the images and ground truths
    np.save(img_npy_path, imgs)
    # np.save(gt_npy_path, gts)

def main(args):
    # List all npz files recursively
    npz_files = glob(os.path.join(args.npz_dir, '**', '*.npz'), recursive=True)
    
    
    # # Calculate 80% of the total number of files
    # num_files_to_select = int(len(npz_files) * 0.8)

    # # Randomly sample 80% of the files
    # npz_files = random.sample(npz_files, num_files_to_select)
    

    # Prepare file info tuples (file path, target base directory)
    file_info_list = [(npz_file, args.npy_dir) for npz_file in npz_files]

    # Process files using multiprocessing
    with mp.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap(convert_npz_to_npy, file_info_list), total=len(file_info_list), desc="Converting npz to npy"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert npz files to npy while maintaining directory structure.")
    parser.add_argument("-npz_dir", type=str, required=True, help="Path to the directory containing .npz files.")
    parser.add_argument("-npy_dir", type=str, required=True, help="Target directory to store .npy files.")
    parser.add_argument("-num_workers", type=int, default=4, help="Number of workers for parallel processing.")
    args = parser.parse_args()
    print("Processing ...")
    main(args)
