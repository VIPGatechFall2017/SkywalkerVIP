import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(1,32), stride=(1,4), padding=2),
            nn.BatchNorm2d(5),
            nn.Dropout2d(p=0.3),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=(1,16), stride=(1,2), padding=2),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=0.3),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(10, 15, kernel_size=(1,8), stride=(1,2), padding=2),
            nn.BatchNorm2d(15),
            nn.Dropout2d(p=0.3),
            nn.ReLU())
        self.fc1 = nn.Linear(32760+3, (32760+3)//2)
        self.fc2 = nn.Linear((32760+3)//2, 32)
        
    def forward(self, x, angle):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out,angle), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out