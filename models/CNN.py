from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3):
        super(CNN, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        layer5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())
            
        self.fs = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            layer5)

        self.fc = nn.Sequential(
            nn.Linear(512 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)) 

    def forward(self, tar, x=None, y=None):
        h = self.fs(tar)
        pred = self.fc(h)
        if self.training == True:
            x = self.fs(x)
            src_pred = self.fc(x)
            loss = F.cross_entropy(src_pred, y)
            return pred, loss
        else:
            return pred
