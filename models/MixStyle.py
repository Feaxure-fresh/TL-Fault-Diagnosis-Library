'''
Paper: Zhou, K., Yang, Y., Qiao, Y. and Xiang, T., 2021. Domain generalization with mixstyle.
       arXiv preprint arXiv:2104.02008.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import random
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import modules
from train_utils import TrainerBase


class MixStyle(nn.Module):
    
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        batch_size = x.size(0)

        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        sigma = (var + self.eps).sqrt()
        mu, sigma = mu.detach(), sigma.detach()
        x_normed = (x - mu) / sigma

        interpolation = self.beta.sample((batch_size, 1, 1))
        interpolation = interpolation.to(x.device)

        # split into two halves and swap the order
        perm = torch.arange(batch_size - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(batch_size // 2)]
        perm_a = perm_a[torch.randperm(batch_size // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu_perm, sigma_perm = mu[perm], sigma[perm]
        mu_mix = mu * interpolation + mu_perm * (1 - interpolation)
        sigma_mix = sigma * interpolation + sigma_perm * (1 - interpolation)

        return x_normed * sigma_mix + mu_mix


class MixStyleLayer(nn.Module):
    def __init__(self,
                 in_channel=1, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 mp_kernel_size=2, 
                 mp_stride=2,
                 num_layer=5):
        super(MixStyleLayer, self).__init__()
        
        layers = []
        current_in_channels = in_channel
        predefined_output_channels = [4, 16, 32, 64, 128]
        num_predefined_layers = len(predefined_output_channels)
        mixstyle = MixStyle()

        for i in range(num_layer):
            if i < num_predefined_layers:
                current_out_channels = predefined_output_channels[i]
            else:
                current_out_channels = 128  # Use 128 for any additional layers
            
            layers.append(nn.Conv1d(current_in_channels, current_out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding))
            layers.append(nn.BatchNorm1d(current_out_channels))
            layers.append(nn.ReLU(inplace=True))
            if i < num_layer - 1:  # For all but the last layer
                layers.append(nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))
            if i <= 1:
                layers.append(mixstyle)
            current_in_channels = current_out_channels

        layers.append(nn.AdaptiveMaxPool1d(1))  # Final adaptive max pool
        layers.append(nn.Flatten())  # Flatten for the output

        self.net = nn.Sequential(*layers)
        self.out_dim = current_out_channels

    def forward(self, input):
        output = self.net(input)
        return output


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.G = modules.MSCNN(in_channel=1, block=MixStyleLayer)
        self.C = modules.MLP(self.G.out_dim, args.num_classes[0], args.dropout, last=None)
        self.model = nn.Sequential(self.G, self.C).to(self.device)
        self._init_data()

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.num_iter = len(self.dataloaders[self.src])
        self.optimizer = self._get_optimizer([self.G, self.C])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
    
    def save_model(self):
        torch.save({
            'model': self.model.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])
    
    def _set_to_train(self):
        self.model.train()
    
    def _set_to_eval(self):
        self.model.eval()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            source_data, source_labels = self._get_next_batch(self.src)

            # forward
            self.optimizer.zero_grad()
            pred = self.model(source_data)

            # compute loss
            loss = F.cross_entropy(pred, source_labels)

            # log information
            epoch_acc['Source Data'] += self._get_accuracy(pred, source_labels)
            epoch_loss['Source Classifier'] += loss

            # backward
            loss.backward()
            self.optimizer.step()
        return epoch_acc, epoch_loss
            
    def _eval(self, data, actual_labels, correct, total):
        pred = self.model(data)
        actual_pred = self._get_actual_label(pred, idx=0)
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]; total['acc'] += output[1]
        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(actual_pred, actual_labels, return_acc=False, idx=0, mode='closed-set')
            correct['Closed-set-acc'] += output[0]; total['Closed-set-acc'] += output[1]
        return correct, total
