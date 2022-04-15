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
        self.shared_fs = SharedFeatureExtractor(1, 128)

        self.specific_fs = nn.ModuleList(SpecificFeatureExtractor(128, 128) \
                                              for _ in range(num_source))

        self.clf = nn.ModuleList(SpecificClassifier(128, num_classes) \
                                              for _ in range(num_source))

        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=True)

    def forward(self, target_data, source_data,
                               source_label=None, source_idx=None):
        if self.training:
            shared_feat = self.shared_fs(source_data)
            specific_feat = self.specific_fs[source_idx](shared_feat)

            shared_feat_tgt = self.shared_fs(target_data)
            specific_feat_tgt = self.specific_fs[source_idx](shared_feat_tgt)

            loss_mmd = self.mkmmd(specific_feat, specific_feat_tgt)

            logits = self.clf[source_idx](feat_src)
            loss_cls = F.nll_loss(F.log_softmax(logits, dim=1), source_label)

            loss_total = loss_cls + loss_mmd

            return loss_total
        else:
            shared_feat_tgt = self.shared_fs(target_data)
            specific_feat_tgt = [spe_fs(shared_feat_tgt) for spe_fs in self.specific_fs]
            logits = [cl(feat_src) for cl in self.clf]

            loss_mmd = torch.Tensor([])
            for i in range(self.num_classes):
                shared_feat = self.shared_fs(source_data[i])
                specific_feat = self.specific_fs[i](shared_feat)
                loss_mmd.append(self.mkmmd(specific_feat, specific_feat_tgt[i]))
            sum_loss = loss_mmd.sum()

            pred = torch.zeros_like(logits[0])
            for i in range(self.num_classes):
                pred += loss_mmd[i] / sum_loss * F.softmax(logits[i], dim=1)

            return pred


def train(self):
    args = self.args
    device = self.device

    F_s = FeatureExtractor(1, 128)
    D = Discriminator(128, int(args.num_source+1))
    C = Classifier(128, args.num_classes)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(*map(list,
                                            [F_s.parameters(), C.parameters()]))),
                                            lr=args.lr, weight_decay=args.weight_decay)
    optimizerD = optim.Adam(D.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    num_iter = int(utils.gmean([len(self.dataloaders[x]) for x in args.source_name]))

    for epoch in range(0, (args.max_epoch+1)):
        F_s.train()
        D.train()
        C.train()

        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        d_correct = 0

        for i in tqdm(range(num_iter)):

            # D iterations
            utils.freeze_net(F_s)
            utils.freeze_net(C)
            utils.unfreeze_net(D)
            D.zero_grad()

            for src in args.all_src:
                idx = args.all_src.index(src)
                d_inputs, _ = utils.get_next_batch(self.dataloaders, self.iters, src, device)
                feat = F_s(d_inputs)
                d_outputs = D(feat)
                d_targets = idx.expand(args.batch_size)

                loss_d = F.nll_loss(F.log_softmax(d_outputs, dim=1), d_targets)
                _, pred = torch.max(d_outputs, -1)
                d_correct += (pred==d_targets).sum().item()
                        
                loss_d.backward()
            optimizerD.step()

           # F&C iteration
           utils.unfreeze_net(F_s)
           utils.unfreeze_net(C)
           utils.freeze_net(D)
           F_s.zero_grad()
           C.zero_grad()

            for src in args.source_name:
                idx = args.all_src.index(src)
                inputs, targets = utils.get_next_batch(self.dataloaders, self.iters, src, device)

                feat = F_s(d_inputs)
                logits = C(feat, idx)

                loss = F.nll_loss(F.log_softmax(logits, dim=1), source_label)
                _, pred = torch.max(logits, -1)

                total[src] += args.batch_size
                correct[src] += (pred == targets).sum().item()

                loss.backward()

            for src in args.all_src:
                idx = args.all_src.index(src)
                d_inputs, _ = utils.get_next_batch(self.dataloaders, self.iters, src, device)
                feat = F_s(d_inputs)
                d_outputs = D(feat)
                d_targets = idx.expand(args.batch_size)

                loss_d = -F.nll_loss(F.log_softmax(d_outputs, dim=1), d_targets)
                _, pred = torch.max(d_outputs, -1)
                d_correct += (pred==d_targets).sum().item()
                        
                loss_d.backward()
            optimizer.step()

if __name__ == '__main__':
    x = torch.randn((32, 1, 1024))
    tar = torch.randn((32, 1, 1024))
    label = torch.randint(3, (32,))
    model = CDAN()
    loss = model(x, tar, label) 
    print(loss)