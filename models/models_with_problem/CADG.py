import torch
import logging
from torch import nn
from tqdm import tqdm
from .resnet import resnet18


def evaluate_acc(dataloaders, F_s, C, device):
    F_s.eval()
    C.eval()
    iters = iter(dataloaders['val'])
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(iters, ascii=True):
            inputs = inputs.to(device)
            targets = targets.to(device)

            features = F_s(inputs)
            outputs = C(features)

            _, pred = torch.max(outputs, -1)
            correct += (pred == targets).sum().item()
            total += pred.shape[0]
        acc = correct / total
        logging.info('Accuracy on {} samples: {:.3f}'.format(total, 100.0*acc))
    return acc


def get_D_labels(labels, num_classes, num_src, device, phase=3, src_idx=None):
    
    if phase == 2:
        D_labels = torch.zeros((labels.shape[0],), dtype=torch.long)
        for i in range(labels.shape[0]):
            D_labels[i] = int(labels[i] + src_idx * num_classes)
    else:
        D_labels = torch.zeros(labels.shape[0], num_classes * num_src)
        for i in range(labels.shape[0]):
            col = [labels[i] + j * num_classes for j in range(num_src)]
            D_labels[i, col] = 1/num_src
    D_labels = D_labels.to(device)
    return D_labels
    

class FeatureExtractor(nn.Module):

    def __init__(self, out_channel=256):
        super(FeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            resnet18(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(2048, out_channel),
            nn.ReLU(inplace=True))

    def forward(self, data):
        f = self.feature_extractor(data)
        return f
        

class Classifier(nn.Module):

    def __init__(self, num_classes=3):
        super(Classifier, self).__init__()

        self.clf = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=-1))

    def forward(self, data):
        out = self.clf(data)
        return out

    
class Discriminator(nn.Module):

    def __init__(self, num_source=1, num_classes=9, dropout=0):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, num_classes * num_source),
            nn.Softmax(dim=-1))

    def forward(self, data):            
        out = self.discriminator(data)
        return out
