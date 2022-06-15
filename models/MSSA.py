import utils
import torch
from torch import nn
import torch.nn.functional as F


class SpecificClassifier(nn.Module):

    def __init__(self, in_channel, num_classes):
        super(SpecificClassifier, self).__init__()

        self.clf = nn.Sequential(
             nn.Linear(in_channel, 64),
             nn.ReLU(inplace=True),

             nn.Linear(64, num_classes))

    def forward(self, input):
        y = self.clf(input)

        return y


class SharedFeatureExtractor(nn.Module):

    def __init__(self, in_channel=1, out_channel=128):
        super(SharedFeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, input):
        feat = self.feature_extractor(input)

        return feat


class SpecificFeatureExtractor(nn.Module):

    def __init__(self, in_channel=128, out_channel=128):
        super(SpecificFeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),

            nn.Flatten(),
            nn.Linear(4*in_channel, out_channel),
            nn.ReLU(inplace=True))

    def forward(self, input):
        feat = self.feature_extractor(input)

        return feat


class MSSA(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(MSSA, self).__init__()

        self.num_classes = num_classes
        self.num_source = num_source
        self.shared_fs = SharedFeatureExtractor(1, 128)

        self.specific_fs = nn.ModuleList(SpecificFeatureExtractor(128, 128) \
                                              for _ in range(num_source))

        self.clf = nn.ModuleList(SpecificClassifier(128, num_classes) \
                                              for _ in range(num_source))

        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
          
    def forward(self, target_data, device, source_data=[], source_label=[]):
        assert len(source_data) == len(source_label) == self.num_source
            
        shared_feat = [self.shared_fs(data) for data in source_data]
        specific_feat = [self.specific_fs[i](shared_feat[i]) for i in range(self.num_source)]

        shared_feat_tgt = self.shared_fs(target_data)
        specific_feat_tgt = [fs(shared_feat_tgt) for fs in self.specific_fs]
        
        logits_tgt = [self.clf[i](specific_feat_tgt[i]) for i in range(self.num_source)]
        logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
        
        if self.training:
            logits = [self.clf[i](specific_feat[i]) for i in range(self.num_source)]
            loss_cls = 0.0
            for i in range(self.num_source):
                loss_cls += F.cross_entropy(logits[i], source_label[i])
        
        loss_mmd = []
        for i in range(self.num_source):
            mmd_single_src = 0.0
            oh_label = utils.one_hot(source_label[i], self.num_classes)
            for j in range(self.num_classes):
                w_src = oh_label[:, j].view(-1, 1).to(device)
                w_tgt = logits_tgt[i][:, j].view(-1, 1).to(device)
                mmd_single_src += self.mkmmd(w_src*specific_feat[i], w_tgt*specific_feat_tgt[i])
            loss_mmd.append(mmd_single_src/self.num_classes)
        sum_mmd = sum(loss_mmd)
        
        pred = torch.zeros_like(logits_tgt[0]).to(device)
        for i in range(self.num_source):
            pred += loss_mmd[i] / sum_mmd * logits_tgt[i]
        
        if self.training:
            return pred, loss_cls, sum_mmd
        else:         
            return pred
