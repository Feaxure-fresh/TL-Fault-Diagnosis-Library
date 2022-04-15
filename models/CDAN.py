import torch
from torch import nn
import torch.nn.functional as F
from utils import *


class CDAN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, beta = 1.):
        super(CDAN, self).__init__()

        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(4*128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout())

        self.clf = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes))

        self.discriminator = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2))

        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                 max_iters=1000, auto_step=True)

    def forward(self, source_data, target_data, source_label):
        batch_size = source_data.shape[0]
        feat_src = self.feature_extractor(source_data)
        feat_tgt = self.feature_extractor(target_data)

        logits = self.clf(feat_src)
        logits_tgt = self.clf(feat_tgt)

        loss = F.nll_loss(F.log_softmax(logits, dim=1), source_label)

        softmax_output_src = F.softmax(logits, dim=-1)
        softmax_output_tgt = F.softmax(logits_tgt, dim=-1)
       
        labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.int32),
                                    torch.zeros(batch_size, dtype=torch.int32)), dim=0)

        feat_src_ = torch.bmm(softmax_output_src.unsqueeze(2),
                                     feat_src.unsqueeze(1)).view(-1, self.num_classes*256)
        feat_tgt_ = torch.bmm(softmax_output_tgt.unsqueeze(2),
                                     feat_tgt.unsqueeze(1)).view(-1, self.num_classes*256)
        print("feat_tgt_:", feat_tgt_.shape)
        feat = self.grl(torch.concat((feat_src_, feat_tgt_), dim=0))
        logits_dm = self.discriminator(feat)
        loss_dm = F.nll_loss(F.log_softmax(logits_dm, dim=1), labels_dm)
        loss_total = loss  + loss_dm

        return loss_total


if __name__ == '__main__':
    x = torch.randn((32, 1, 1024))
    tar = torch.randn((32, 1, 1024))
    label = torch.randint(3, (32,))
    model = CDAN()
    loss = model(x, tar, label) 
    print(loss)