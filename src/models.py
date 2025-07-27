import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    """Initializes a ResNet-18 model with a modified final layer."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model