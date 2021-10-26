import torch
import torch.nn as nn


class InternalClassifier(nn.Module):
  def __init__(self, n_channels, num_classes, block_expansion=1):
    super().__init__()
    
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(n_channels * block_expansion, num_classes)
  
  def forward(self, x):
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x