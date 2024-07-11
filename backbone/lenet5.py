import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8464, 256), # 100 * 100
            nn.Linear(256, 64),
            nn.Linear(64, 25)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.Flatten()(x)
        out = self.fc(x)
        return out

if __name__ == '__main__':
    x = torch.randn(64, 3, 100, 100)
    model = LeNet5()
    print(model(x).size())