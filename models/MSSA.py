import torch
from torch import nn
import torch.nn.functional as F
import utils


class SpecificClassifier(nn.Module):

    def __init__(self, in_channel, num_classes):
        super(SpecificClassifier, self).__init__()

        self.clf = nn.Sequential(
             nn.Linear(in_channel, 64),
             nn.ReLU(inplace=True),

             nn.Linear(in_channel, num_classes))

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
            nn.Maxpool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
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
          
    def forward(self, target_data, source_data,
                               source_label=None, source_idx=None):
        if self.training:
            shared_feat = self.shared_fs(source_data)
            specific_feat = self.specific_fs[source_idx](shared_feat)

            shared_feat_tgt = self.shared_fs(target_data)
            specific_feat_tgt = self.specific_fs[source_idx](shared_feat_tgt)

            logits = self.clf[source_idx](specific_feat)
            logits_tgt = self.clf[source_idx](specific_feat_tgt)
            loss_cls = F.nll_loss(F.log_softmax(logits, dim=1), source_label)

            oh_label = utils.one_hot(source_label, self.num_classes)
            loss_mmd = torch.Tensor([0.])
            for i in range(self.num_classes):
                w_src = oh_label[:, i].view(-1, 1)
                w_tgt = logits_tgt[:, i].view(-1, 1)
                loss_mmd += self.mkmmd(w_src*specific_feat, w_tgt*specific_feat_tgt)

            loss_total = loss_cls + loss_mmd / self.num_classes

            return loss_total
        else:
            shared_feat_tgt = self.shared_fs(target_data)
            specific_feat_tgt = [spe_fs(shared_feat_tgt) for spe_fs in self.specific_fs]
            logits = [cl(feat_src) for cl in self.clf]

            loss_mmd = torch.Tensor([])
            for i in range(self.num_source):
                shared_feat = self.shared_fs(source_data[i])
                specific_feat = self.specific_fs[i](shared_feat)
                loss_mmd.append(self.mkmmd(specific_feat, specific_feat_tgt[i]))
            sum_loss = loss_mmd.sum()

            pred = torch.zeros_like(logits[0])
            for i in range(self.num_source):
                pred += loss_mmd[i] / sum_loss * F.softmax(logits[i], dim=1)

            return pred


if __name__ == '__main__':
    #x = torch.randn((32, 1, 1024))
    #tar = torch.randn((32, 1, 1024))
    label = torch.randint(3, (32,))
    #model = CDAN()
    #loss = model(x, tar, label)
    print(label)
    print(torch.Tensor([0.]))