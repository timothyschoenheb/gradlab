import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    # 3-block convnet for 28x28 grayscale inputs
    def __init__(self, num_classes=10, width=32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(1, width), block(width, width * 2), block(width * 2, width * 4))
        self.head = nn.Linear(width * 4 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.head(x.flatten(1))
