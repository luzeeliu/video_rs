import re
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

## attention in dataprocessing we dont need use dataset 
# this data will be used in item embadding train so it is static not streaming
#file_path = os.path.join("dataset", "train.parquet")

# .arrow file manipulate
"""
# use memory_map for zero-copy reads
source = pa.memory_map(file_path, "r")
# build reader to load the data
reader = ipc.RecordBatchFileReader(source)
table = reader.read_all()

df = table.to_pandas()
print(df.head())
"""
#df = pd.read_parquet(file_path, columns=("caption","coarse_label"))

# dataset processing step
# lower-case, special character processing tokenize
# MPNet have their own tokenizer
"""
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]','',text)
    return text.split()
"""
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# model class
class rs_MPNet(Dataset):
    def __init__(self, dataframe, tokenizer, max_len = 128):
        self.texts = dataframe['caption'].tolist()
        self.labels = dataframe['coarse_label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    # in torch train must have len and getitem in model data processig calss
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = str(self.labels[index])
        encoding = self.tokenizer(
            text,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_len,
            return_tensors = "pt"
        )
        return{
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def dataprocessing_MPNet(df):
    return rs_MPNet(df,tokenizer)
        