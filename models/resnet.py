import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Transfer(nn.Module):
    """
    ResNet50 model adapted for Transfer Learning on DR detection.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50Transfer, self).__init__()
        
        # Load pre-trained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.base_model = resnet50(weights=weights)
        
        # We extract the number of features of the last layer
        num_ftrs = self.base_model.fc.in_features
        
        # Replace the classifier layer with a new one for our num_classes
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.base_model(x)
