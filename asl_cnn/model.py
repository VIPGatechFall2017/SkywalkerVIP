import torch 
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(1,8), stride=(1,4), padding=2),
            nn.BatchNorm2d(5),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=(1,4), stride=(1,2), padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU())
        self.fc = nn.Linear(10*13*261, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out