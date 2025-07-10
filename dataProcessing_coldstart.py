import os
import pandas as pd


data_path = os.path.join("dataset", "Trending_video.csv")

df = pd.read_csv(data_path)
print(df.head())

