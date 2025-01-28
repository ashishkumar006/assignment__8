import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()

        # C1: Initial Conv Layer
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, padding=1)  # RF: 3

        # C2: Depthwise Separable Convolution with dilation
        self.depthwise = nn.Conv2d(24, 24, kernel_size=3, groups=24, padding=2, dilation=2)  # RF: 7
        self.pointwise = nn.Conv2d(24, 48, kernel_size=1)  # RF: 7

        # C3: Strided Convolution (Reduce spatial dims)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)  # RF: 15

        # C4: Dilated Convolution
        self.conv3 = nn.Conv2d(64, 80, kernel_size=3, padding=4, dilation=4)  # RF: 31

        # C5: Final Dilated Convolution with stride 2 (to reduce spatial size)
        self.conv4 = nn.Conv2d(80, 96, kernel_size=3, stride=2, padding=6, dilation=6)  # RF: 49

        # GAP Layer
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(96, 10)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(80)
        self.bn5 = nn.BatchNorm2d(96)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.pointwise(self.depthwise(x))))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = F.relu(self.bn5(self.conv4(x)))
        x = self.gap(x)
        x = x.view(-1, 96)
        x = self.fc(x)
        return x

# Initialize the model
model = CIFAR10Net()

# Print model summary for input size (3, 32, 32) corresponding to CIFAR-10 images
summary(model, (3, 32, 32))
