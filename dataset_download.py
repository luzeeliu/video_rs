import kagglehub
import os
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_dataset
import shutil

path = os.path.join('..',"dataset")
# Download latest version
os.makedirs(path, exist_ok=True)



dataset = kagglehub.dataset_download("anushabellam/trending-videos-on-youtube")
# chaneg the download location
#csv_path = os.path.join(dataset,"Trending videos on youtube dataset.csv")
shutil.move(dataset,path)
original = os.path.join(path,"1","Trending videos on youtube dataset.csv")
new = os.path.join(path,"Trending_video.csv")
os.rename(original,new)
print("Path to dataset files:", path)




# Login using e.g. `huggingface-cli login` to access this dataset
"""
ds = load_dataset(
    "AdoCleanCode/Youtube8M_real_train_data_v4_0.8",
    split="train"          
)
"""
# save to own path file
#ds.save_to_disk(path)
"""
original = os.path.join("dataset","train","data-00000-of-00001.arrow")
name = os.path.join("dataset", "Youtube_train.arrow")
# rename
os.rename(original, name)
"""
# write own arrow file to have more control
"""
table = ds.to_arrow()
outpath = os.path.join("dataset","YouTube_train.arrow")
with pa.OSFile(outpath, "wb") as sink:
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
"""
# download pandas version

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/AdoCleanCode/Youtube8M_real_train_data_v4_0.8/data/train-00000-of-00001.parquet")
out_path = os.path.join(path, "train_original.parquet")
df.to_parquet(out_path)

print(f"save in {out_path}")

for sub in ("train", "test","validation"):
    os.makedirs(os.path.join(path, sub), exist_ok=True)

df2 = pd.read_parquet("hf://datasets/AdoCleanCode/Youtube8M_general_test_data/data/train-00000-of-00001.parquet")
out_path3 = os.path.join(path,"test","test.parquet")
df2.to_parquet(out_path3)

dp = load_dataset("parquet", data_files={"train":out_path}, split="train")

splits = dp.train_test_split(test_size=0.2, seed=25)

train_path = os.path.join(path,"train","train.parquet")
valid_path = os.path.join(path,"validation","val.parquet")
splits["train"].to_parquet(train_path)
splits["test"].to_parquet(valid_path)
print("successful get dataset")