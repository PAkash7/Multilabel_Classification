import torch.nn as nn
from torchvision import models

def get_model(num_classes=4, pretrained=True):
    """
    Returns a ResNet50 model modified for multilabel classification.
    """
    # Load pretrained ResNet50 on ImageNet
    model = models.resnet50(pretrained=pretrained)
    
    # Replace final fully connected layer for num_classes outputs
    num_ftrs = model.fc.in_features
    # We do not use sigmoid here because we will use BCEWithLogitsLoss during training
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
