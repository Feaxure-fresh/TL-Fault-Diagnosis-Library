import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from .resnet import BasicBlock


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.uniform(0.0, 1.0, real_samples.size(0))).view(-1, 1)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).requires_grad_(False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def evaluate_acc(dataloaders, model, device):
        model.eval()
        iters = iter(dataloaders['val'])
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(iters):
                inputs = inputs.to(device)
                targets = targets.to(device)

                tgt_pred = model(inputs)
                unknown = [(1 - data[:, -1]).view(-1, 1) for data in tgt_pred]
                unknown = torch.cat(unknown, dim=1)
                sum_unknown = unknown.sum(dim=1).view(-1, 1)
                
                pred = 0
                for i in range(unknown.size(-1)):
                    pred += unknown[:, i].view(-1, 1) / sum_unknown * tgt_pred[i][:, :-1]
                
                _, pred = torch.max(pred, -1)
                correct += (pred == targets).sum().item()
                total += targets.shape[0]
            acc = correct / total
            logging.info('Accuracy on {} samples: {:.3f}%'.format(total, 100.0*acc))
        return acc
    

class Discriminator(nn.Module):

    def __init__(self, in_channel=256):
        super(Discriminator, self).__init__()

        self.dm = nn.Sequential(
             nn.Linear(in_channel, 256),
             nn.LeakyReLU(inplace=True),

             nn.Linear(256, 256),
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
            nn.MaxPool1d(kernel_size=2, stride=2),

            BasicBlock(20, 20),
            BasicBlock(20, 20),
            BasicBlock(20, 20),
            BasicBlock(20, 20),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(5120, 256),
            nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Linear(256, int(num_classes+1)),
            nn.Tanh())

        self.sm = nn.Softmax(dim=-1)

    def forward(self, input):
        feat = self.feature_extractor(input)
        y = self.fc(feat)
        y = self.sm(y)
        
        if self.training:
            return feat, y
        else:
            return y


class MSTLN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(MSTLN, self).__init__()

        self.num_source = num_source
        self.pda = nn.ModuleList([PDA_Subnet(in_channel, num_classes) \
                                              for _ in range(num_source)])
          
    def forward(self, inputs, source_idx=None):
        if self.training:
            feat, y = self.pda[source_idx](inputs)
            return feat, y
        else:
            y = [module(inputs) for module in self.pda]
            return y
