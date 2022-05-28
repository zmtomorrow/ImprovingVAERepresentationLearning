  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tools import *
from torch.distributions import Normal


class ClassNet(nn.Module):
    def __init__(self,device, z_dim):
        super(ClassNet, self).__init__()
        self.device=device
        self.fc1= nn.Linear(z_dim, 200)
        self.bn=nn.BatchNorm1d(200)
        self.dropout=nn.Dropout(0,1)
        self.fc2 = nn.Linear(200, 10)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self,x):
        h=F.relu(self.fc1(x))
        h=self.fc2(self.bn(self.dropout(h)))
        return torch.log_softmax(h,dim=-1)

    def predict(self,x):
        p=self.forward(x)
        pred=p.argmax(dim=-1)
        return pred



class m2_classifier(nn.Module):
    def __init__(self, x_dim=784, h_dim=500, y_dim=10, if_bn=False):
        super().__init__()
        self.fc1= nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 10)
        
        if if_bn:
            self.bn1=nn.BatchNorm1d(h_dim)
            self.bn2=nn.BatchNorm1d(h_dim)

        else:
            self.bn1=lambda x:x
            self.bn2=lambda x:x

        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x, apply_softmax=True):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        logits = self.fc3(x)
        if not apply_softmax:
            return logits
        probs = F.softmax(logits, dim=-1)
        return probs
    

class m2_classifier_color(nn.Module):
    def __init__(self, if_bn=False,drop_out_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout1=nn.Dropout(drop_out_rate)
        self.dropout2=nn.Dropout(drop_out_rate)
        self.dropout3=nn.Dropout(drop_out_rate)
        self.dropout4=nn.Dropout(drop_out_rate)
        if if_bn:
            self.bn1=nn.BatchNorm2d(6)
            self.bn2=nn.BatchNorm2d(16)
            self.bn3=nn.BatchNorm1d(128)
            self.bn4=nn.BatchNorm1d(64)
        else:
            self.bn1=lambda x:x
            self.bn2=lambda x:x
            self.bn3=lambda x:x
            self.bn4=lambda x:x

    def forward(self, x, apply_softmax=True):
        x = self.dropout1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = torch.flatten(x, 1) 
        x = self.dropout3(F.relu(self.bn3(self.fc1(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc2(x))))
        logits = self.fc3(x)
        if not apply_softmax:
            return logits
        probs = F.softmax(logits, dim=-1)
        return probs
