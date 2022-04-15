import torch
from torch import nn
import torch.nn.functional as F
import utils


class AutoEncoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True))

        self.backward_pass = nn.Sequential(
            nn.Linear(output_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU(inplace=True))

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        y = self.forward_pass(x)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = AutoEncoder(1024, 256)
        self.ae2 = AutoEncoder(256, 128)
        self.ae3 = AutoEncoder(128, 64)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)

        if self.training:
            return a3
        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
            a2_reconstruct = self.ae3.reconstruct(x)
            a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
            x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
            return x_reconstruct

class MADN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(MADN, self).__init__()

        self.num_classes = num_classes
        self.auto_encoder = StackedAutoEncoder()

        self.clf = nn.ModuleList([nn.Linear(128, num_classes) \
                                              for _ in range(num_source)])

        self.discriminator = nn.ModuleList([nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)) for _ in range(num_source)])
 
        self.grl = utils.GradientReverseLayer()

    def forward(self, target_data, source_data=None,
                               source_label=None, source_idx=None):
        if self.training:
            self.auto_encoder.train()
            batch_size = source_data.shape[0]
            feat_src = self.auto_encoder(source_data)
            feat_tgt = self.auto_encoder(target_data)

            logits = self.clf[source_idx](feat_src)
            loss_cls = F.nll_loss(F.log_softmax(logits, dim=1), source_label)
       
            feat = self.grl(torch.concat((feat_src, feat_tgt), dim=0))
            logits_dm = self.discriminator[source_idx](feat)
            labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.int32),
                                  torch.zeros(batch_size, dtype=torch.int32)), dim=0)
            loss_d = F.nll_loss(F.log_softmax(logits_dm, dim=1), labels_dm)

            loss_total = loss_cls + loss_d

            return loss_total
        else:
            self.auto_encoder.eval()
            feat_src = self.auto_encoder(target_data)

            logits = [cl(feat_src) for cl in self.clf]
            logits_dm = [dis(feat_src) for dis in self.discriminator]

            labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.int32),
                                  torch.zeros(batch_size, dtype=torch.int32)), dim=0)
            loss_dm = torch.Tensor([F.nll_loss(F.log_softmax(logits_dm, dim=1), labels_dm)])
            sum_loss = loss_dm.sum()

            pred = torch.zeros_like(logits[0])
            for i in range(self.num_classes):
                pred += loss_dm[i] / sum_loss * F.softmax(logits[i], dim=1)

            return pred

if __name__ == '__main__':
    x = torch.randn((32, 1024))
    tar = torch.randn((32, 1024))
    label = torch.randint(3, (32,))
    model = StackedAutoEncoder()
    model.train()
    print(model)
    feat = model(x) 