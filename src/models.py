import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet18(num_classes=num_classes)
        
        # Modify the first convolutional layer for small images
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Re-initialize the weights of the new layer
        nn.init.kaiming_normal_(
            self.resnet.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        return self.resnet(x)