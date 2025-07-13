import torch
import os
import pickle
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataProcessing_MPNet import dataprocessing_MPNet
from MPNet import vanilla_MPNet

dataset_path = os.path.join('..',"dataset")
model_save_base = os.path.join('..',"nn")
train_path = os.path.join(dataset_path,"train","train.parquet")
val_path = os.path.join(dataset_path,"validation","val.parquet")
save_path = os.path.join(model_save_base,"MPNet.pth")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # in this mode TypeError: SentenceTransformer.forward() takes 2 positional arguments but 3 were given
        features = {
            'input_ids': inputs,
            'attention_mask': attention_mask
        }
        
        # TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not dict
        # SentenceTransformer returns a dict; our Dense head overwrote
        # 'sentence_embedding' with out_dim scores → use that as logits
        outputs = model(features)
        logits = outputs['sentence_embedding']
        # common training step
        # return model output
        # calculate loss 
        # lossfunction backtrack
        # optimize
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()  
         
        # evaluate
        total_loss += loss.item()
        preds = torch.argmax(logits, dim= 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    # save model
    
    
    return avg_loss, acc

@torch.no_grad()
def validation(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    tot = 0
    # get batch from loader
    for batch in val_loader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        features = {
            'input_ids': inputs,
            'attention_mask': attention_mask
        }
        
        output = model(features)
        logits = output['sentence_embedding']
        loss = criterion(logits, labels)
        # different with train is no backtrack and optimize
        total_loss += loss.item()
        correct += (torch.argmax(logits, dim = 1) == labels).sum().item()
        
        tot += labels.size(0)
    acc = correct / tot
    avg_loss = total_loss / len(val_loader)
    
    return acc, avg_loss

def main():
    # common setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = vanilla_MPNet()
    model = wrapper.model.to(device)
    optimize = Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    # parpare for training
    df_train = pd.read_parquet(train_path)
    label_ids,train_data = dataprocessing_MPNet(df_train)
    train_loader = DataLoader(train_data, batch_size= 32, shuffle= True, num_workers=4,pin_memory=True)
    # save the labels
    with open(os.path.join(model_save_base,"label.kpl"), "wb") as f:
        pickle.dump(label_ids, f)
    
    # parpare for validation
    df_val = pd.read_parquet(val_path)
    _,val_data = dataprocessing_MPNet(df_val, label_ids)
    val_loader = DataLoader(val_data, batch_size= 32, shuffle= True, num_workers= 4,pin_memory=True)
    
    best = 0
    for epoch in range(epochs):
        train_avgloss, train_acc = train_epoch(model, train_loader,criterion, optimize, device)
        val_avgloss, val_acc = validation(model, val_loader, criterion, device)
        print(f"in {epoch} epoch, the training accuracy is {train_acc}, average training loss is {train_avgloss}")
        print(f"in {epoch} epoch, the validation accuracy is {val_acc}, average validation loss is {val_avgloss}")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(),save_path)    
    

if __name__ == '__main__':
    main()