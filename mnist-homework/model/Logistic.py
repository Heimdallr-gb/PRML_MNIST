import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# Logistic Regression方法

class logisticRg(nn.Module):
    def __init__(self):
        super(logisticRg, self).__init__()
        self.lr = nn.Sequential(
            nn.Linear(28*28,10)
        )

    def forward(self, x):
        output = self.lr(x)
        return output, x   