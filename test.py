# import os
# import pyarrow as pa
# import pyarrow.ipc as ipc

# # Define the directory containing your .arrow files
# directory_path = '/projects/multimodal/datasets/lc25000_lung/cache/lc25000_lung_train.arrow'

# # Initialize a variable to keep track of the total number of rows
# total_rows = 0

# # Loop over each file in the directory
# for filename in os.listdir(directory_path):
#     file_path = os.path.join(directory_path, filename)
#     # Check if the file is an Arrow file
#     if file_path.endswith('.arrow'):
#         # Open the Arrow file
#         with open(file_path, 'rb') as f:
#             reader = ipc.open_file(f)
#             table = reader.read_all()
#             # Add the number of rows in this table to the total
#             total_rows += table.num_rows
#             print(f"Number of rows in {filename}: {table.num_rows}")

# # Print the total number of rows in the dataset
# print("Total number of rows in the dataset:", total_rows)




# import torch

# # Path to your .ckpt file
# checkpoint_path = '/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch=31-step=104672.ckpt'

# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# # Print all keys in the state dictionary
# if 'state_dict' in checkpoint:
#     # For models saved with PyTorch Lightning, the keys might be in checkpoint['state_dict']
#     state_dict = checkpoint['state_dict']
# else:
#     # For standard PyTorch models, the keys might be directly in the checkpoint
#     state_dict = checkpoint

# # List all keys
# for key in state_dict.keys():
#     print(key)



# import pickle
# import datasets

# # Path to your .pkl file
# file_path = '/projects/multimodal/datasets/pcam/cache/pcam_test.pkl'

# # Load the .pkl file
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Check if the data is an instance of datasets.arrow_dataset.Dataset
# if isinstance(data, datasets.arrow_dataset.Dataset):
#     print("Number of instances in the dataset:", data.num_rows)
# else:
#     print("Data type:", type(data))
#     print("Review the loaded data structure to determine how to count instances.")
    


import pandas as pd

# Path to your CSV file
csv_file_path = '/projects/multimodal/datasets/PMC-OA-2_labels/pmcoa_2_valid_imagenet_5_labels.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path, error_bad_lines=False)

# Print the DataFrame column names
# print("Column names:", df["caption"][10])
print(f"{df.head(5)}")
print(f"{len(df)}")




# --------------------------------------------------------- split outputs in train valid and test
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load the original CSV file
# file_path = 'output_medsam.csv'
# data = pd.read_csv(file_path)

# # Define split sizes
# train_size = 0.7  # 70% for training
# val_size = 0.15   # 15% for validation
# test_size = 0.15  # 15% for testing

# # First split: train and temp (temp will be split into val and test)
# train_data, temp_data = train_test_split(data, test_size=(1 - train_size), random_state=42)

# # Second split: split temp data into val and test
# val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=42)

# # Save the splits to new CSV files
# train_data.to_csv('train.csv', index=False)
# val_data.to_csv('val.csv', index=False)
# test_data.to_csv('test.csv', index=False)

# print("Data split and saved successfully!")





# --------------------------------------------------- get PMC-OA split ratio
# import os

# # Define the file paths
# train_file = "/projects/multimodal/datasets/pmc_oa/train.jsonl"
# valid_file = "/projects/multimodal/datasets/pmc_oa/valid.jsonl"
# test_file = "/projects/multimodal/datasets/pmc_oa/test.jsonl"

# # Function to count lines in a file
# def count_lines(file_path):
#     with open(file_path, 'r') as f:
#         return sum(1 for line in f)

# # Count lines in each file
# train_count = count_lines(train_file)
# valid_count = count_lines(valid_file)
# test_count = count_lines(test_file)

# # Calculate total and split ratios
# total = train_count + valid_count + test_count
# train_ratio = train_count / total
# valid_ratio = valid_count / total
# test_ratio = test_count / total

# # Print results
# print(f"Train Count: {train_count}, Ratio: {train_ratio:.2%}")
# print(f"Valid Count: {valid_count}, Ratio: {valid_ratio:.2%}")
# print(f"Test Count: {test_count}, Ratio: {test_ratio:.2%}")



# ------------ split pmcoa-2 into train valid and test ----------------------------------
# import os
# import json
# import random

# # Define the path to the original .jsonl file
# input_file = "/projects/multimodal/datasets/pmc_oa/pmc_oa2.jsonl"
# output_dir = os.path.dirname(input_file)

# # Load all lines from the original .jsonl file
# with open(input_file, 'r') as f:
#     data = [json.loads(line) for line in f]

# # Shuffle the data
# random.shuffle(data)

# # Calculate split indices based on the 80/10/10 ratio
# total = len(data)
# train_end = int(0.8 * total)
# valid_end = train_end + int(0.1 * total)

# # Split the data
# train_data = data[:train_end]
# valid_data = data[train_end:valid_end]
# test_data = data[valid_end:]

# # Define output file paths
# train_file = os.path.join(output_dir, "pmc_oa2_train.jsonl")
# valid_file = os.path.join(output_dir, "pmc_oa2_valid.jsonl")
# test_file = os.path.join(output_dir, "pmc_oa2_test.jsonl")

# # Function to write a split to a .jsonl file
# def write_jsonl(data, file_path):
#     with open(file_path, 'w') as f:
#         for item in data:
#             f.write(json.dumps(item) + '\n')

# # Write each split to its respective file
# write_jsonl(train_data, train_file)
# write_jsonl(valid_data, valid_file)
# write_jsonl(test_data, test_file)

# print(f"Data split and saved to:\nTrain: {train_file}\nValid: {valid_file}\nTest: {test_file}")


