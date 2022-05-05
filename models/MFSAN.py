import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ADDneck, resnet18


class MFSAN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(MFSAN, self).__init__()
        
        self.num_source = num_source
        self.sharedNet = resnet18(False, in_channel=in_channel)
        self.sonnet = nn.ModuleList([ADDneck(512, 256) for _ in range(num_source)])
        self.cls_fc_son = nn.ModuleList(([nn.Linear(256, num_classes)
                                                       for _ in range(num_source)]))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                     nn.Flatten())

    def forward(self, data_tgt, data_src=None, label_src=None, source_idx=None, device=None):
        if self.training == True:
            feat_src = self.sharedNet(data_src)
            feat_tgt = self.sharedNet(data_tgt)

            feat_tgt = [son(feat_tgt) for son in self.sonnet]
            feat_tgt = [self.avgpool(data) for data in feat_tgt]

            feat_src = self.sonnet[source_idx](feat_src)
            feat_src = self.avgpool(feat_src)
            
            loss_mmd = utils.MFSAN_mmd(feat_src, feat_tgt[source_idx])
            
            logits_src = self.cls_fc_son[source_idx](feat_src)
            logits_tgt = [self.cls_fc_son[i](feat_tgt[i]) for i in range(self.num_source)]
            loss_cls = F.cross_entropy(logits_src, label_src)

            loss_l1 = 0.0
            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
            for i in range(self.num_source - 1):
                for j in range(i+1, self.num_source):
                    loss_l1 += torch.abs(logits_tgt[i] - logits_tgt[j]).sum()
            loss_l1 /= self.num_source

            return logits_tgt[source_idx], loss_cls, loss_mmd, loss_l1
        else:
            feat = self.sharedNet(data_tgt)

            feat = [son(feat) for son in self.sonnet]
            feat = [self.avgpool(data) for data in feat]
            logits_tgt = [self.cls_fc_son[i](feat[i]) for i in range(self.num_source)]
            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
            
            pred = torch.zeros((logits_tgt[0].shape)).to(device)
            for i in range(self.num_source):
                pred += logits_tgt[i]

            return pred
