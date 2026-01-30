import torch
from tqdm import tqdm 
import os
import re   

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()  
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training")
    
    for batch_videos, batch_labels in loop:
        # Shape videos: [Batch, 3, 16, 224, 224]
        videos = batch_videos.to(device)
        labels = batch_labels.to(device)
        outputs = model(videos) # Shape: [Batch, 101]
        loss = criterion(outputs, labels)
        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
    avg_loss = running_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    
    model.eval()  
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_videos, batch_labels in tqdm(loader, desc="Validation"):
            videos = batch_videos.to(device)
            labels = batch_labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

def get_next_run_id(save_dir):
    """
    Checks existing folders (Run_01, Run_02...) and returns the next available ID.
    """
    # If the folder doesn't exist yet, start from Run_01
    if not os.path.exists(save_dir):
        return "Run_01"
    
    
    existing_folders = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    
    # Extract numbers from folders named "Run_XX"
    run_numbers = []
    pattern = re.compile(r"^Run_(\d+)$") # Looks for "Run_" followed by digits
    
    for folder in existing_folders:
        match = pattern.match(folder)
        if match:
            run_numbers.append(int(match.group(1)))
            
    if not run_numbers:
        next_id = 1
    else:
        next_id = max(run_numbers) + 1
        
    return f"Run_{next_id:02d}" 