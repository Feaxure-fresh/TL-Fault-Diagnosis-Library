import utils
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from resnet import BasicBlock
import torch.autograd as autograd


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.uniform((real_samples.size(0), 1, 1, 1)))

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).requires_grad(True)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Discriminator(nn.Module):

    def __init__(self, in_channel=256):
        super(Discriminator, self).__init__()

        self.dm = nn.Sequential(
             nn.Linear(in_channel, 256),
             nn.LeakyReLU(inplace=True),

             nn.Linear(265, 256),
             nn.LeakyReLU(inplace=True),

             nn.Linear(256, 1))

    def forward(self, input):
        y = self.dm(input)

        return y


class PDA_Subnet(nn.Module):

    def __init__(self, in_channel=1, num_classes=3):
        super(PDA_Subnet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, 20, kernel_size=3, padding=1),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.Maxpool1d(kernel_size=2, stride=2),

            BasicBlock(20, 20),
            BasicBlock(20, 20),
            BasicBlock(20, 20),
            BasicBlock(20, 20),
            nn.Maxpool1d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_channel*5, 256),
            nn.ReLU(inplace=True))

        self.fc = nn.Linear(256, int(num_classes+1))

        self.sm = nn.Softmax(dim=1)

    def forward(self, input):
        feat = self.feature_extractor(input)
        y = self.fc(feat)
        y = self.sm(y)

        return feat, y


class MSTLN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(MSTLN, self).__init__()

        self.num_classes = num_classes
        self.num_source = num_source
        
        self.pda = nn.ModuleList([PDA_Subnet(in_channel, num_classes) \
                                              for _ in range(num_source)])
          
    def forward(self, inputs, source_idx=None):
        if self.training:
            feat, y = self.pda[source_idx](inputs)

            return feat, y


def train(self):
    args = self.args
    device = self.device

    model = MSTLN(1, args.num_classes, args.num_source)
    D = nn.ModuleList([Discriminator(256) for _ in range(args.num_source)])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(0, (args.max_epoch+1)):
        model.train()
        D.train()
        
        tgt_inputs, tgt_labels = utils.get_next_batch(self.dataloaders, self.iters, 'train', device)
        tgt_feat_list = []
        loss_k, loss_shared, loss_da, loss_da_ms = 0.0, 0.0, 0.0, 0.0
        for idx, src in enumerate(args.source_name):
            optimizer.zero_grad()
            optimizerD.zero_grad()

            src_inputs, src_labels = utils.get_next_batch(self.dataloaders, self.iters, src, device)

            src_feat, src_pred = model(src_inputs, idx)
            tgt_feat, tgt_pred = model(tgt_inputs, idx)
            tgt_feat_list.append(tgt_feat)

            loss_k += F.nll_loss(torch.log(src_pred), src_labels)

            unknown_mean = tgt_pred[:, -1].sum() / args.batch_size
            loss_unknown = torch.Tensor([0.0])
            for i in range(args.batch_size):
                if tgt_pred[i, -1] >= unknown_mean:
                    loss_unknown -= torch.log(tgt_pred[i, -1])
            loss_unknown /= args.batch_size
            loss_unknown.backward()
            optimizer.step()

            sums = (1 - tgt_pred[:, -1]).sum()
            for i in range(args.batch_size):
                loss_shared -= (1 - tgt_pred[i, -1]) / sums * \
                    torch.mm(tgt_pred[i, :-1], torch.log(tgt_pred[i, :-1].view(-1, 1)))

            src_validity = D[idx](src_feat)
            tgt_validity = D[idx](tgt_feat)
            
            gradient_penalty = compute_gradient_penalty(D[idx], src_feat, tgt_feat)
            loss_adv = -torch.sum(src_validity) + torch.sum(tgt_validity) + 10 * gradient_penalty
            loss_adv.backward()
            optimizerD.step()

            src_validity = F.softmax(src_validity, dim=1)
            tgt_validity = F.softmax(tgt_validity, dim=1)
            means = torch.mean((1 - tgt_pred[:, -1]).view(-1, 1) * tgt_validity)
            cb_factors = (1 - tgt_pred[:, -1]).view(-1, 1) * tgt_validity / means
            loss_da += utils.mmd(src_feat, tgt_feat, cb=cb_factors)
        
        for i in range(args.num_source - 1):
            for j in range(i+1, args.num_source):
                loss_da_ms += utils.mmd(tgt_feat_list[i], tgt_feat_list[j])
        
        loss_tl = loss_k + loss_da + loss_shared + loss_da_ms
        loss_tl.backward()
        optimizer.step()


if __name__ == '__main__':
    #x = torch.randn((32, 1, 1024))
    #tar = torch.randn((32, 1, 1024))
    label = torch.randint(3, (32,))
    #model = CDAN()
    #loss = model(x, tar, label)
    print(label)
    print(torch.Tensor([0.]))