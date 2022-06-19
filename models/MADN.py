import utils
import torch
from torch import nn
import torch.nn.functional as F


class encoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, output_size),
            nn.ReLU())

    def forward(self, x):
        return self.encoder(x)
        

class decoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, output_size))

    def forward(self, x):
        return self.decoder(x)


class MADN(nn.Module):

    def __init__(self, lr, in_channel=1, num_classes=3, num_source=1):
        super(MADN, self).__init__()

        self.num_source = num_source
        self.encoder = encoder(1024, 64)
        self.decoder = decoder(64, 1024)

        self.clf = nn.ModuleList([nn.Sequential(
                   nn.Linear(64, 32),
                   nn.ReLU(),
                                    
                   nn.Linear(32, num_classes)) \
                   for _ in range(num_source)])

        self.discriminator = nn.ModuleList([nn.Sequential(
            nn.Linear(64, 10),
            nn.ReLU(inplace=True),
            
            nn.Linear(10, 2)) for _ in range(num_source)])
 
        self.grl = utils.GradientReverseLayer()
        
        self.criterion = nn.MSELoss()
        
        self.optimizer = torch.optim.SGD([{'params': self.encoder.parameters()},
                                          {'params': self.decoder.parameters()}], lr=lr)
    
    def forward(self, target_data, device, source_data=[], source_label=[], rec=False):
        target_data = target_data.squeeze(1)
        source_data = [item.squeeze(1) for item in source_data]
        
        if self.training:
            assert len(source_data) == len(source_label) == self.num_source
            self.auto_encoder.train()
            
            if rec:
                recx = self.auto_encoder(target_data, rec=True)
                loss_ae = self.criterion(recx, target_data)
                self.optimizer.zero_grad()
                loss_ae.backward()
                self.optimizer.step()
                
                for data in source_data:
                    recx = self.auto_encoder(data, rec=True)
                    loss_ae = self.criterion(recx, target_data)
                    self.optimizer.zero_grad()
                    loss_ae.backward()
                    self.optimizer.step()
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

model = MADN(0.1)
print(list(model.parameters()))