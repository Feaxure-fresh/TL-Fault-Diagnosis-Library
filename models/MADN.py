import utils
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, int((output_size+input_size)/2)),
            nn.BatchNorm1d(int((output_size+input_size)/2)),
            nn.ReLU(inplace=True),
            
            nn.Linear(int((output_size+input_size)/2), output_size))

        self.decoder = nn.Sequential(
            nn.Linear(output_size, int((output_size+input_size)/2)),
            nn.BatchNorm1d(int((output_size+input_size)/2)),
            nn.ReLU(inplace=True),
            
            nn.Linear(int((output_size+input_size)/2), input_size))
            
        self.criterion = nn.MSELoss()
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):

        return self.encoder(x)

    def reconstruct(self, x):
        self.optimizer.zero_grad()
        y = self.encoder(x)
        x_rec = self.decoder(y)
        loss = self.criterion(x_rec, x)
        loss.backward()
        self.optimizer.step() 
        
        return y.detach()


class StackedAutoEncoder(nn.Module):

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = AutoEncoder(1024, 512)
        self.ae2 = AutoEncoder(512, 256)
        self.ae3 = AutoEncoder(256, 128)

    def forward(self, x, rec=False):
        if rec:
            a1_reconstruct = self.ae1.reconstruct(x)
            a2_reconstruct = self.ae2.reconstruct(a1_reconstruct)
            a3_reconstruct = self.ae3.reconstruct(a2_reconstruct)
        else:
            a1 = self.ae1(x)
            a2 = self.ae2(a1)
            a3 = self.ae3(a2)
            return a3
            

class MADN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(MADN, self).__init__()

        self.num_source = num_source
        self.auto_encoder = StackedAutoEncoder()

        self.clf = nn.ModuleList([nn.Sequential(
                   nn.Linear(128, 64),
                   nn.ReLU(),
                                    
                   nn.Linear(64, num_classes)) \
                   for _ in range(num_source)])

        self.discriminator = nn.ModuleList([nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(inplace=True),
            
            nn.Linear(10, 2)) for _ in range(num_source)])
 
        self.grl = utils.GradientReverseLayer()  
    
    def forward(self, target_data, device, source_data=[], source_label=[]):
        target_data = target_data.squeeze(1)
        source_data = [item.squeeze(1) for item in source_data]
        
        if self.training:
            assert len(source_data) == len(source_label) == self.num_source
            self.auto_encoder.train()
            
            self.auto_encoder(target_data, rec=True)
            for data in source_data:
                self.auto_encoder(data, rec=True)
        else:
            assert len(source_data) == self.num_source
            self.auto_encoder.eval()
        
        batch_size = target_data.shape[0]
        feat_src = [self.auto_encoder(data) for data in source_data]
        feat_tgt = self.auto_encoder(target_data)

        logits_tgt = [clf(feat_tgt) for clf in self.clf]
        
        if self.training:
            logits_src = [self.clf[i](feat_src[i]) for i in range(self.num_source)]
            loss_cls = 0.0
            for i in range(self.num_source):
                loss_cls += F.cross_entropy(logits_src[i], source_label[i])
            loss_cls /= self.num_source
        
        loss_dm = []
        labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.long),
                    torch.zeros(batch_size, dtype=torch.long)), dim=0).to(device)
        for i in range(self.num_source):
            feat = self.grl(torch.concat((feat_src[i], feat_tgt), dim=0))
            logits_dm = self.discriminator[i](feat)
            loss_dm.append(F.cross_entropy(logits_dm, labels_dm))
        loss_dm_sum = sum(loss_dm)

        pred = torch.zeros_like(logits_tgt[0])
        for i in range(self.num_source):
            pred += loss_dm[i] / loss_dm_sum * F.softmax(logits_tgt[i], dim=1)
                
        if self.training:
            loss_dm_sum /= self.num_source
            return pred, loss_cls, loss_dm_sum
        else:
            return pred
