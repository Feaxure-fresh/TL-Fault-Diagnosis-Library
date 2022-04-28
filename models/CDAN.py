import utils
import torch
from torch import nn
import torch.nn.functional as F


class CDAN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, dropout=0):
        super(CDAN, self).__init__()

        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(4*128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))

        self.clf = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes))

        self.discriminator = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2))

        self.grl = utils.GradientReverseLayer()

    def forward(self, target_data, source_data=None, source_label=None):
        if self.training == True:
            batch_size = source_data.shape[0]
            feat_src = self.feature_extractor(source_data)
            feat_tgt = self.feature_extractor(target_data)
    
            logits = self.clf(feat_src)
            logits_tgt = self.clf(feat_tgt)
    
            loss = F.cross_entropy(logits, source_label)
    
            softmax_output_src = F.softmax(logits, dim=-1)
            softmax_output_tgt = F.softmax(logits_tgt, dim=-1)
           
            labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.long),
                torch.zeros(batch_size, dtype=torch.long)), dim=0).to(target_data.device)
    
            feat_src_ = torch.bmm(softmax_output_src.unsqueeze(2),
                            feat_src.unsqueeze(1)).view(batch_size, self.num_classes*128)
            feat_tgt_ = torch.bmm(softmax_output_tgt.unsqueeze(2),
                            feat_tgt.unsqueeze(1)).view(batch_size, self.num_classes*128)
            feat = self.grl(torch.concat((feat_src_, feat_tgt_), dim=0))
            logits_dm = self.discriminator(feat)
            loss_dm = F.cross_entropy(logits_dm, labels_dm)
    
            return logits_tgt, loss, loss_dm
        else:
            feat_tgt = self.feature_extractor(target_data)
            logits_tgt = self.clf(feat_tgt)
            
            return logits_tgt
