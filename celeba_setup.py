import os
import shutil
import kagglehub
from tqdm import tqdm

# Download the dataset
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

# Define the target directory
target_dir = './dataset/celeba'

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Move the downloaded dataset to the target directory
for item in tqdm(os.listdir(path)):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)

print(f"Dataset moved to {target_dir}")
