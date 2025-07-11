import kagglehub
import os
import pyarrow as pa
import pyarrow.ipc as ipc
# Download latest version
path = os.path.join("dataset")

out_path2 = os.path.join("dataset","Trending_video.csv")
dataset = kagglehub.dataset_download("anushabellam/trending-videos-on-youtube")
# chaneg the download location
os.replace(dataset,out_path2)

print("Path to dataset files:", out_path2)

from datasets import load_dataset


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
out_path = os.path.join("dataset", "train.parquet")
df.to_parquet(out_path)

print(f"save in {out_path}")