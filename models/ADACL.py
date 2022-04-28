import utils
import torch
from torch import nn
import torch.nn.functional as F


class ADACL(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(ADACL, self).__init__()

        self.num_source = num_source
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=9, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(4, 8, kernel_size=9, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(8, 16, kernel_size=9, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=9, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=9, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=9, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),

            nn.Flatten(),
            nn.Linear(4*128, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True))

        self.clf = nn.ModuleList([nn.Linear(128, num_classes) \
                                              for _ in range(num_source)])

        self.discriminator = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, int(1+num_source)))
 
        self.grl = utils.GradientReverseLayer()

    def forward(self, target_data, device, source_data=None, source_label=None, source_idx=None):
        if self.training == True:
            batch_size = source_data.shape[0]
            feat_src = self.feature_extractor(source_data)
            feat_tgt = self.feature_extractor(target_data)
    
            logits = self.clf[source_idx](feat_src)
            logits_tgt = [cl(feat_tgt) for cl in self.clf]
            loss_cls = F.cross_entropy(logits, source_label)
           
            labels_dm = torch.concat((torch.full((batch_size,), source_idx, dtype=torch.long),
                                      torch.zeros(batch_size, dtype=torch.long)), dim=0).to(device)
            feat = self.grl(torch.concat((feat_src, feat_tgt), dim=0))
            logits_dm = self.discriminator(feat)
            loss_d = F.cross_entropy(logits_dm, labels_dm)
            
            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
            loss_l1 = 0.0
            for i in range(self.num_source - 1):
                for j in range(i+1, self.num_source):
                    loss_l1 += torch.abs(logits_tgt[i] - logits_tgt[j]).sum()
            loss_l1 /= self.num_source
                       
            return logits_tgt[source_idx], loss_cls, loss_d, loss_l1
        else:
            feat_tgt = self.feature_extractor(target_data)
            logits_tgt = [cl(feat_tgt) for cl in self.clf]
            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
            
            pred = torch.zeros((logits_tgt[0].shape)).to(device)
            for i in range(self.num_source):
                pred += logits_tgt[i]
                
            return pred
