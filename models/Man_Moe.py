import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


def save_model(path, F_s, F_p, C, D):
    torch.save(F_s.state_dict(),
            '{}/netF_s.pth'.format(path))
    torch.save(F_p.state_dict(),
            '{}/net_F_p.pth'.format(path))
    torch.save(C.state_dict(),
            '{}/netC.pth'.format(path))
    torch.save(D.state_dict(),
            '{}/netD.pth'.format(path))


def evaluate_acc(dataloaders, F_s, F_p, C, device):
    F_s.eval()
    F_p.eval()
    C.eval()
    iters = iter(dataloaders['val'])
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(iters, ascii=True):
            inputs = inputs.to(device)
            targets = targets.to(device)

            shared_features = F_s(inputs)
            private_feat, _ = F_p(inputs)
            outputs, _ = C((shared_features, private_feat))

            _, pred = torch.max(outputs, -1)
            correct += (pred == targets).sum().item()
            total += pred.shape[0]
        acc = correct / total
        logging.info('Accuracy on {} samples: {:.3f}'.format(total, 100.0*acc))
    return acc


class FeatureExtractor(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 dropout):
        super(FeatureExtractor, self).__init__()
        
        window_sizes = [4, 8, 16, 24, 32]
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=input_size, 
                                        out_channels=64, 
                                        kernel_size=h,
                                        stride=8,
                                        padding=8),
                              nn.ReLU(),
                              nn.AdaptiveMaxPool1d(8))
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
            

class MLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 last='tanh'):
        super(MLP, self).__init__()
        
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

    def forward(self, input):
        y = self.net(input)
        if self.last != None:
            y = self.last_layer(y)
        
        return y
    
    
class MixtureOfExperts(nn.Module):
    
   def __init__(self,
                input_size,
                output_size,
                num_source,
                dropout):
       super(MixtureOfExperts, self).__init__()
       
       self.gate = MLP(input_size=input_size, output_size=num_source,
                       dropout=dropout, last='sm')
       
       self.experts = nn.ModuleList([nn.Sequential(                    
                                    nn.Linear(input_size, output_size),
                                    nn.ReLU()) for _ in range(num_source)])
       
   def forward(self, input):
        gate_input = input.detach()
        gate_outs = self.gate(gate_input)
        
        expert_outs = torch.stack([exp(input) for exp in self.experts], dim=-2)
        output = torch.sum(gate_outs.unsqueeze(-1) * expert_outs, dim=-2)
        
        return output, gate_outs


class ClassifierMoE(nn.Module):
    
   def __init__(self,
                input_size,
                output_size,
                num_source,
                dropout,
                mode=0):
       super(ClassifierMoE, self).__init__()
       
       self.mode = mode
       self.gate = MLP(input_size=input_size if mode != 1 else int(input_size/2),
                       output_size=num_source, dropout=dropout, last='sm')
       
       self.experts = nn.ModuleList([MLP(input_size=input_size if mode != 1 else int(input_size/2),
                output_size=16, dropout=dropout, last='logsm') for _ in range(num_source)])
       
       if mode > 0:
           self.sp =  nn.Linear(in_features=input_size if mode == 2 else int(input_size/2),
                                                              out_features=int(mode))         
       
   def forward(self, input):
       fs, fp = input
       if self.mode == 1:
           a1 = self.sp(fs)
           a2 = self.sp(fp)
           
           alphas = F.softmax(torch.cat([a1, a2], dim=-1), dim=-1)
           features = torch.stack([fs, fp], dim=-2)
           features = torch.sum(alphas.unsqueeze(-1) * features, dim=-2)
       elif self.mode == 2:
           features = torch.cat([fs, fp], dim=-1)
           alphas = F.softmax(self.sp(features), dim=-1)
           
           fs = fs * alphas[:, 0].view(-1, 1)
           fp = fp * alphas[:, 1].view(-1, 1)
           features = torch.cat([fs, fp], dim=-1)
       else:
           features = torch.cat([fs, fp], dim=-1)          
       gate_input = features.detach()
       gate_outs = self.gate(gate_input)
        
       expert_outs = torch.stack([exp(features) for exp in self.experts], dim=-2)
       output = torch.sum(gate_outs.unsqueeze(-1) * expert_outs, dim=-2)
        
       return output, gate_outs
