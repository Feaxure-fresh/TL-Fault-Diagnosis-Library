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

    def __init__(self, in_channel=1, kernel_size=7):
        super(CNNlayer, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        layer5 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())
        
        self.fs = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            layer5)

    def forward(self, tar, x=None, y=None):
        h = self.fs(tar)
        
        return h
            

class FeatureExtractor(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 dropout):
        super(FeatureExtractor, self).__init__()
        
        window_sizes = [4, 8, 16, 24, 32]
        self.convs = nn.ModuleList([
                       CNNlayer(in_channel=1, kernel_size=h)
                       for h in window_sizes])
                              
        self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=dropout),
                    
                    nn.Linear(2560, output_size),
                    nn.ReLU())

    def forward(self, input):
        out = [conv(input) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = self.fc(out)
        
        return out


class BaseModel(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 num_classes,
                 dropout):
        super(BaseModel, self).__init__()
        
        self.G = FeatureExtractor(input_size, output_size, dropout)
        
        self.C = ClassifierMLP(output_size, num_classes, dropout, last=None)
        
    def forward(self, input):
        f = self.G(input)
        predictions = self.C(f)
        if self.training:
            return predictions, f
        else:
            return predictions
    
    def save_model(self, path):
        torch.save(self.G.state_dict(),
                '{}/netG.pth'.format(path))
        torch.save(self.C.state_dict(),
                '{}/netC.pth'.format(path))

