import kagglehub
import os
# Download latest version
"""
path = kagglehub.dataset_download("anushabellam/trending-videos-on-youtube")

print("Path to dataset files:", path)
"""
from datasets import load_dataset

path = os.path.join("dataset")
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset(
    "AdoCleanCode/Youtube8M_real_train_data_v4_0.8",
    cache_dir=path              
)