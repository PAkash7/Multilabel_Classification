import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MultilabelDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        self._parse_labels(label_file)
        
    def _parse_labels(self, label_file):
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # First 5 lines are likely headers: Image Name, Attr1, Attr2, Attr3, Attr4
        if len(lines) >= 1 and "Image Name" in lines[0]:
            lines = lines[1:] # if there is a header, drop the first line
            
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                img_name = parts[0]
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    attr1 = self._parse_val(parts[1])
                    attr2 = self._parse_val(parts[2])
                    attr3 = self._parse_val(parts[3])
                    attr4 = self._parse_val(parts[4])
                    
                    self.images.append(img_name)
                    self.labels.append([attr1, attr2, attr3, attr4])
                
    def _parse_val(self, val):
        val = str(val).lower()
        if 'na' in val:
            return float('nan')
        elif '1' in val:
            return 1.0
        elif '0' in val:
            return 0.0
        return float('nan')
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a blank image if missing/corrupt
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return image, label_tensor

def get_class_weights(dataset):
    """
    Computes pos_weight for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = negative_samples / positive_samples
    """
    labels = np.array(dataset.labels)
    num_classes = labels.shape[1]
    pos_weights = []
    
    for i in range(num_classes):
        col = labels[:, i]
        valid_indices = ~np.isnan(col)
        valid_col = col[valid_indices]
        
        pos_count = np.sum(valid_col == 1.0)
        neg_count = np.sum(valid_col == 0.0)
        
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0
        pos_weights.append(weight)
        
    return torch.tensor(pos_weights, dtype=torch.float32)
