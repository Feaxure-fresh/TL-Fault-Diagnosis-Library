import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 last='tanh'):
        super(ClassifierMLP, self).__init__()
        
        self.last = last
        self.net = nn.Sequential(
                   nn.Dropout(p=dropout),
                   
                   nn.Linear(input_size, int(input_size/4)),
                   nn.ReLU(),
                   
                   nn.Linear(int(input_size/4), int(input_size/16)),
                   nn.ReLU(),
                   
                   nn.Linear(int(input_size/16), output_size))
        
        if last == 'logsm':
            self.last_layer = nn.LogSoftmax(dim=-1)
        elif last == 'sm':
            self.last_layer = nn.Softmax(dim=-1)
        elif last == 'tanh':
            self.last_layer = nn.Tanh()
        elif last == 'sigmoid':
            self.last_layer = nn.Sigmoid()
        elif last == 'relu':
            self.last_layer = nn.ReLU()

    def forward(self, input):
        y = self.net(input)
        if self.last != None:
            y = self.last_layer(y)
        
        return y


class CNNlayer(nn.Module):

    def __init__(self,
                 in_channel=1, kernel_size=8, stride=1, padding=1,
                 mp_kernel_size=2, mp_stride=2, dropout=0.):
        super(CNNlayer, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer5 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())
        
        dp = nn.Dropout(dropout)
        
        self.fs = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            dp,
            layer5)

    def forward(self, tar, x=None, y=None):
        h = self.fs(tar)
        
        return h


class FeatureExtractor(nn.Module):
    
    def __init__(self, in_channel, window_sizes=[4, 8, 16, 24, 32], block=CNNlayer, dropout=0.):
        super(FeatureExtractor, self).__init__()
       
        self.convs = nn.ModuleList([
                       block(in_channel=in_channel, kernel_size=h, dropout=dropout)
                       for h in window_sizes])
                              
        self.fl = nn.Flatten()

    def forward(self, input):
        out = [conv(input) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = self.fl(out)
        
        return out


class BaseModel(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 dropout):
        super(BaseModel, self).__init__()
        
        self.G = FeatureExtractor(in_channel=input_size, dropout=dropout)
        
        self.C = ClassifierMLP(2560, num_classes, dropout, last=None)
        
    def forward(self, input):
        f = self.G(input)
        predictions = self.C(f)
        if self.training:
            return predictions, f
        else:
            return predictions
