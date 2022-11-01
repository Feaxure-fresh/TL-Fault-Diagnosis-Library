import random
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import torch.nn as nn

import utils
from train_utils import InitTrain
import model_base


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

        mu = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], keepdim=True)
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


class CNNlayer(nn.Module):

    def __init__(self, in_channel=1, kernel_size=7):
        super(CNNlayer, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())
        
        self.mixstyle = MixStyle()

    def forward(self, tar, x=None, y=None):
        h = self.layer1(tar)
        h = self.mixstyle(h)
        h = self.layer2(h)
        h = self.mixstyle(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        
        return h
            

class FeatureExtractor(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 dropout):
        super(FeatureExtractor, self).__init__()
        
        window_sizes = [4, 8, 16, 24, 32]
        self.convs = nn.ModuleList([
                       CNNlayer(in_channel=1, kernel_size=h)
                       for h in window_sizes])
                              
        self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=dropout),
                    
                    nn.Linear(2560, output_size),
                    nn.ReLU())

    def forward(self, input):
        out = [conv(input) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = self.fc(out)
        
        return out


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
                    kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        self.G = FeatureExtractor(input_size=1, output_size=1024, dropout=args.dropout).to(self.device)
        self.C = model_base.ClassifierMLP(input_size=1024, output_size=args.num_classes,
                                          dropout=args.dropout, last=None).to(self.device)
    
    def train(self):
        args = self.args
        self._init_data()

        if args.train_mode == 'supervised':
            src = None
        elif args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained with multi-source data.")
        
        self.optimizer = self._get_optimizer([self.G, self.C])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        
        best_acc = 0.0
        best_epoch = 0
   
        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
   
            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                epoch_acc = defaultdict(float)
   
                # Set model to train mode or evaluate mode
                if phase == 'train':
                    self.G.train()
                    self.C.train()
                    epoch_loss = defaultdict(float)
                    tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
                else:
                    self.G.eval()
                    self.C.eval()
                
                num_iter = len(self.iters[phase])
                for i in tqdm(range(num_iter), ascii=True):
                    target_data, target_labels = utils.get_next_batch(self.dataloaders,
                    						 self.iters, phase, self.device)
                    if phase == 'train':
                        if src != None:
                            source_data, source_labels = utils.get_next_batch(self.dataloaders,
                        						     self.iters, src, self.device)
                        else:
                            source_data, source_labels = target_data, target_labels
                        with torch.set_grad_enabled(True):
                            # forward
                            self.optimizer.zero_grad()
                            data = torch.cat((source_data, target_data), dim=0)
                            
                            f = self.G(data)
                            y = self.C(f)
                            y_s, _ = y.chunk(2, dim=0)
                            f_s, f_t = f.chunk(2, dim=0)
                            
                            loss_mmd = self.mkmmd(f_s, f_t)
                            loss_c = F.cross_entropy(y_s, source_labels)
                            loss = loss_c + tradeoff[0] * loss_mmd
                            epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                            
                            epoch_loss['Source Classifier'] += loss_c
                            epoch_loss['Mk MMD'] += loss_mmd

                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred = self.C(self.G(target_data))
                            epoch_acc['Target Data']  += utils.get_accuracy(pred, target_labels)
                
                # Print the train and val information via each epoch
                if phase == 'train':
                    for key in epoch_loss.keys():
                        logging.info('{}-Loss {}: {:.4f}'.format(phase, key, epoch_loss[key]/num_iter))
                for key in epoch_acc.keys():
                    logging.info('{}-Acc {}: {:.4f}'.format(phase, key, epoch_acc[key]/num_iter))
                               
                # log the best model according to the val accuracy
                if phase == 'val':
                    new_acc = epoch_acc['Target Data']/num_iter
                    if new_acc >= best_acc:
                        best_acc = new_acc
                        best_epoch = epoch
                    logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
