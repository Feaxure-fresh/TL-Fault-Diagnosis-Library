from torch import nn
import torch.nn.functional as F


class WDCNN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3):
        super(WDCNN, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=32, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten())
            
        self.fs = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            layer5)

        self.fc = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes))

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
