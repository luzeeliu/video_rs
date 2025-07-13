import os
import torch
import pickle
import time
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm


from dataProcessing_MPNet import dataprocessing_MPNet
from MPNet import vanilla_MPNet



dataset_path = os.path.join('..',"dataset")
model_save_base = os.path.join('..',"nn")
train_path = os.path.join(dataset_path,"train","train.parquet")
val_path = os.path.join(dataset_path,"validation","val.parquet")
save_path = os.path.join(model_save_base,"MPNet.pth")

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    t0 = time.perf_counter()
    pbar = tqdm(loader, desc=f"training Epoch {epoch} ▶", unit="batch")
    for batch in pbar:

        inputs = batch['input_ids'].to(device, non_blocking = True)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # in this mode TypeError: SentenceTransformer.forward() takes 2 positional arguments but 3 were given

        features = {
            'input_ids': inputs,
            'attention_mask': attention_mask
        }
        
        # implement AMP
        with autocast('cuda'):
            embedding = model(features)['sentence_embedding']
            loss = criterion(embedding, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
         
        
        # evaluate
        total_loss += loss.item()
        preds = embedding.argmax(dim = 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(
          loss=f"{total_loss/total:.4f}",
          acc=f"{correct/total:.4f}"
        )
    
    t1 = time.perf_counter()
    print(f"[TIMING] 1 epoch fwd+back: {t1-t0:.2f}s")
    avg_loss = total_loss / len(loader)
    acc = correct / total
    # save model
    
    
    return avg_loss, acc

@torch.no_grad()
def validation(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    tot = 0
    pbar = tqdm(val_loader, desc=f"validation Epoch {epoch} ▶", unit="batch")
    # get batch from loader
    for batch in pbar:
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
        preds = logits.argmax(dim = 1)
        correct += (preds == labels).sum().item()
        
        tot += labels.size(0)
        pbar.set_postfix(
          loss=f"{total_loss/tot:.4f}",
          acc=f"{correct/tot:.4f}"
        )
    acc = correct / tot
    avg_loss = total_loss / len(val_loader)
    
    return acc, avg_loss

def main():
    
    # parpare for training
    df_train = pd.read_parquet(train_path)
    label_ids,train_data, label_num = dataprocessing_MPNet(df_train)
    train_loader = DataLoader(train_data, batch_size= 32, shuffle= True, num_workers= 4, pin_memory=True)
    """
    t0 = time.perf_counter()
    for _ in train_loader:
        pass
    t1 = time.perf_counter()
    print(f"[TIMING] Full epoch load time: {t1-t0:.2f}s")
    """
    # save the labels
    with open(os.path.join(model_save_base,"label.kpl"), "wb") as f:
        pickle.dump(label_ids, f)
    
    # parpare for validation
    df_val = pd.read_parquet(val_path)
    _,val_data,_ = dataprocessing_MPNet(df_val, label_ids)
    val_loader = DataLoader(val_data, batch_size= 32, shuffle= True, num_workers= 4, pin_memory=True)
    
    
    # common setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on ", device)
    scaler = GradScaler()
    
    wrapper = vanilla_MPNet(out_dim= len(label_num))
    model = wrapper.model.to(device)
    # model = torch.compile(model)
    #model = torch.compile(model, backend="nvfuser")
    trunk = model._modules['0']
    head = model._modules['2']
    
    #model.load_state_dict(torch.load(save_path, weights_only= True))
    optimize = AdamW([
        {"params": trunk.parameters(), "lr": 1e-5},
        {"params": head.parameters(), "lr":1e-3},
    ], weight_decay= 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing= 0.1)
    #scheduler = optim.lr_scheduler.StepLR(optimize, step_size=30, gamma=0.8)
    epochs = 6

    best = 0
    for epoch in range(epochs):
        train_avgloss, train_acc = train_epoch(model, train_loader,criterion, optimize, scaler,device, epoch)
        val_acc, val_avgloss = validation(model, val_loader, criterion, device, epoch)
        print(f"in {epoch} epoch, the training accuracy is {train_acc}, average training loss is {train_avgloss}")
        print(f"in {epoch} epoch, the validation accuracy is {val_acc}, average validation loss is {val_avgloss}")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(),save_path)    
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("training crashed:", e)
        raise