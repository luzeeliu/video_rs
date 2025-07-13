import re
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

## attention in dataprocessing we dont need use dataset 
########################because in the dataset label it is string so we need convert it in to idx just like vocb ############
# this data will be used in item embadding train so it is static not streaming
"""
dataset_base = os.path.join('..',"dataset")
file_path = os.path.join(dataset_base, "train","train.parquet")

df = pd.read_parquet(file_path)
print(df.head())
"""
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

def label_voc(dataframe):
    label = dataframe['coarse_label'].tolist()
    unique_label = sorted(set(label))
    label_id = {l: idx for idx, l in enumerate(unique_label)}
    labels = [label_id[l] for l in label]
    return labels, label_id

# dataset processing class
class rs_MPNet(Dataset):
    def __init__(self, dataframe, labels, tokenizer, max_len = 128):
        self.texts = dataframe['caption'].tolist()
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        
    
    # in torch train must have len and getitem in model data processig calss
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
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

def dataprocessing_MPNet(df, labels = None, label_id = None):
    if labels is None:
        labels, label_id = label_voc(df)
    return labels, rs_MPNet(df,labels, tokenizer), label_id
        