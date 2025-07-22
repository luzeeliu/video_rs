import os
import pandas as pd

dataset_path = os.path.join('..',"dataset")

train_path = os.path.join(dataset_path,"train","train.parquet")

df = pd.read_parquet(train_path, columns=['coarse_label'])

print(df.head())
