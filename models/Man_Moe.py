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
        
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=output_size,
                num_layers=2, dropout=dropout, batch_first=True)
        
    def forward(self, input):
            input = input.permute(0, 2, 1)
            output, _ = self.rnn(input, None)
            
            return output


class MLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 is_tagger=False):
        super(MLP, self).__init__()
        
        self.net = nn.Sequential(
                   nn.Dropout(p=dropout),
                   
                   nn.Linear(input_size, 64),
                   nn.ReLU(),
                   
                   nn.Linear(64, 64),
                   nn.ReLU(),
                   
                   nn.Linear(64, output_size))
        
        if is_tagger:
            self.last = nn.LogSoftmax(dim=-1)
        else:
            self.last = nn.Tanh()

    def forward(self, input):
        h = self.net(input)
        y = self.last(h)
        
        return y
    
    
class MixtureOfExperts(nn.Module):
    
   def __init__(self,
                input_size,
                output_size,
                num_source,
                dropout):
       super(MixtureOfExperts, self).__init__()
       
       self.gate = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=128, 
                                           kernel_size=32, stride=8),
                                 nn.ReLU(),
                                 nn.AdaptiveMaxPool1d(1),
                                 nn.Flatten(),
                                 nn.Linear(in_features=128, out_features=num_source),
                                 nn.Softmax(dim=-1))
       
       self.experts = nn.ModuleList([MLP(input_size=input_size, output_size=output_size,
                                         dropout=dropout) for _ in range(num_source)])
       
   def forward(self, input):
        gate_input = input.detach()
        gate_input = gate_input.permute(0, 2, 1)
        gate_outs = self.gate(gate_input)
        
        expert_outs = torch.stack([exp(input) for exp in self.experts], dim=-2)
        
        gate_mult = gate_outs.unsqueeze(1).expand(gate_outs.shape[0], input.shape[1],
                                                  gate_outs.shape[1])
        output = torch.sum(gate_mult.unsqueeze(-1) * expert_outs, dim=-2)
        
        return output, gate_outs


class MoE_CNN(nn.Module):
    
   def __init__(self,
                input_size,
                output_size,
                num_source,
                mode=0):
       super(MoE_CNN, self).__init__()
       
       self.mode = mode
       self.gate = nn.Sequential(nn.Conv1d(in_channels=input_size
                                           if mode != 1 else int(input_size/2),
                                           out_channels=64, kernel_size=32, stride=8),
                                 nn.ReLU(),
                                 nn.AdaptiveMaxPool1d(1),
                                 nn.Flatten(),
                                 nn.Linear(in_features=64, out_features=num_source),
                                 nn.Softmax(dim=-1))
       
       self.experts = nn.ModuleList([
               nn.Sequential(nn.Conv1d(in_channels=input_size
                                       if mode != 1 else int(input_size/2),
                                       out_channels=64, kernel_size=32, stride=8),
                             nn.ReLU(),
                             nn.AdaptiveMaxPool1d(1),
                             nn.Flatten(),
                             nn.Linear(in_features=64, out_features=32),
                             nn.ReLU(),
                             nn.Linear(in_features=32, out_features=16),
                             nn.ReLU())
                             for _ in range(num_source)])
       
       self.fc = nn.Sequential(nn.Linear(in_features=16, out_features=output_size),
                               nn.LogSoftmax(dim=-1))
       
       if mode > 0:
           self.sp = nn.Sequential(nn.Conv1d(in_channels=input_size
                                             if mode == 2 else int(input_size/2),
                                             out_channels=64, kernel_size=32, stride=8),
                                   nn.ReLU(),
                                   nn.AdaptiveMaxPool1d(1),
                                   nn.Flatten(),
                                   nn.Linear(in_features=64, out_features=int(mode)))           
       
   def forward(self, input):
       fs, fp = input
       if self.mode == 1:
           a1 = self.sp(fs.permute(0, 2, 1))
           a2 = self.sp(fp.permute(0, 2, 1))
           
           alphas = F.softmax(torch.cat([a1, a2], dim=-1), dim=-1)
           alphas = alphas.unsqueeze(1).expand(alphas.shape[0], fs.shape[1], alphas.shape[1])
           features = torch.stack([fs, fp], dim=2)
           features = torch.sum(alphas.unsqueeze(-1) * features, dim=2)
       elif self.mode == 2:
           features = torch.cat([fs, fp], dim=-1)
           features = features.permute(0, 2, 1)
           
           alphas = F.softmax(self.sp(features), dim=-1)
           fs = fs * alphas[:, 0].view(-1, 1, 1)
           fp = fp * alphas[:, 1].view(-1, 1, 1)
           features = torch.cat([fs, fp], dim=-1)
       else:
           features = torch.cat([fs, fp], dim=-1)
       features = features.permute(0, 2, 1)
           
       gate_input = features.detach()
       gate_outs = self.gate(gate_input)
        
       expert_outs = torch.stack([exp(features) for exp in self.experts], dim=-2)
       output = torch.sum(gate_outs.unsqueeze(-1) * expert_outs, dim=-2)    
       output = self.fc(output)
        
       return output, gate_outs


class Discriminator(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_source):
        super(Discriminator, self).__init__()
        
        window_sizes = [4, 8, 16, 24, 32]
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=input_size, 
                                        out_channels=128, 
                                        kernel_size=h,
                                        stride=4),
                              nn.ReLU(),
                              nn.AdaptiveMaxPool1d(1))
                              for h in window_sizes])
        
        self.fc = nn.Linear(in_features=128*len(window_sizes),
                            out_features=num_source)
        
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        out = [conv(input) for conv in self.convs]
        out = torch.cat(out, dim=1).squeeze(-1)
        
        out = self.fc(out)
        output = self.softmax(out)
        
        return output