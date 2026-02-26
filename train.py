import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import MultilabelDataset, get_class_weights
from model import get_model

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def masked_bce_loss(outputs, targets, pos_weight):
    """
    Computes BCEWithLogitsLoss but ignores targets with NaN values.
    """
    # Create mask for valid targets
    mask = ~torch.isnan(targets)
    
    # Replace NaN with 0 temporarily so loss function doesn't crash
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    
    # Compute loss component-wise
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    loss = criterion(outputs, safe_targets)
    
    # Apply mask
    masked_loss = loss * mask.float()
    
    # Safe guard against division by zero if all targets are NaN in a batch
    num_valid = mask.float().sum()
    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        return torch.tensor(0.0, requires_grad=True).to(outputs.device)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and DataLoader setup
    train_tfms, val_tfms = get_transforms()
    
    # Load entire dataset
    full_dataset = MultilabelDataset(args.data_dir, args.label_file, transform=train_tfms)
    
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Check data_dir and label_file.")
        
    pos_weights = get_class_weights(full_dataset).to(device)
    print(f"Class positive weights for imbalance handling: {pos_weights}")
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # We should ideally apply val_tfms to val_dataset, but random_split shares the transform.
    # For a simple setup, this is acceptable, but could be customized by building subset classes.
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model setup
    model = get_model(num_classes=4, pretrained=True)
    model = model.to(device)
    
    # Optimizer setup - training all layers (fine-tuning) with a small learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Track metrics
    iteration_losses = []
    
    print("Starting training...")
    iteration_num = 0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = masked_bce_loss(outputs, targets, pos_weights)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            iteration_losses.append(loss.item())
            iteration_num += 1
            
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        # Simple Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = masked_bce_loss(outputs, targets, pos_weights)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] Validation Loss: {val_loss:.4f}")
        
    # Save the model weights
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model weights saved to {args.model_save_path}")
    
    # Plot the loss curve
    plt.figure()
    plt.plot(range(1, iteration_num + 1), iteration_losses)
    plt.xlabel('iteration_number')
    plt.ylabel('training_loss')
    plt.title('Aimonk_multilabel_problem')
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multilabel Classification Model')
    parser.add_argument('--data_dir', type=str, default='images', help='Directory containing images')
    parser.add_argument('--label_file', type=str, default='labels.txt', help='Path to labels descriptor')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='model_weights.pth', help='Path to save trained weights')
    
    args = parser.parse_args()
    
    # Create output dir if needed
    os.makedirs(os.path.dirname(args.model_save_path) or '.', exist_ok=True)
    
    train(args)
