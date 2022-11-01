import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import utils
from train_utils import InitTrain
import model_base


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:

    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


class RandomizedMultiLinearMap(nn.Module):

    def __init__(self, features_dim: int, num_classes: int, output_dim: int = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)
    

class ConditionalDomainAdversarialLoss(nn.Module):
   
    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: bool = False,
                 randomized: bool = False, num_classes: int = -1,
                 features_dim: int = -1, randomized_dim: int = 1024,
                 reduction: str = 'mean', sigmoid=True, grl = None):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = utils.WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) \
                                                                                        if grl is None else grl
        self.entropy_conditioning = entropy_conditioning
        self.sigmoid = sigmoid
        self.reduction = reduction

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size

        if self.sigmoid:
            d_label = torch.cat((
                torch.ones((g_s.size(0), 1)).to(g_s.device),
                torch.zeros((g_t.size(0), 1)).to(g_t.device),
            ))
            self.domain_discriminator_accuracy = utils.binary_accuracy(d, d_label)
            if self.entropy_conditioning:
                return F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction=self.reduction)
            else:
                return F.binary_cross_entropy(d, d_label, reduction=self.reduction)
        else:
            d_label = torch.cat((
                torch.ones((g_s.size(0), )).to(g_s.device),
                torch.zeros((g_t.size(0), )).to(g_t.device),
            )).long()
            self.domain_discriminator_accuracy = utils.get_accuracy(d, d_label)
            if self.entropy_conditioning:
                raise NotImplementedError("entropy_conditioning")
            return F.cross_entropy(d, d_label, reduction=self.reduction)


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        
        self.domain_discri = model_base.ClassifierMLP(input_size=1024 * args.num_classes, output_size=1,
                        dropout=args.dropout, last='sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = ConditionalDomainAdversarialLoss(self.domain_discri, grl=grl)
        self.model = model_base.BaseModel(input_size=1, output_size=1024,
                                     num_classes=args.num_classes, dropout=args.dropout).to(self.device)
        
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

        self.optimizer = self._get_optimizer([self.model, self.domain_discri])
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
                    self.model.train()
                    self.domain_discri.train()
                    epoch_loss = defaultdict(float)
                    tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
                else:
                    self.model.eval()
                
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
                            
                            y, f = self.model(data)
                            f_s, f_t = f.chunk(2, dim=0)
                            y_s, y_t = y.chunk(2, dim=0)
                    
                            loss_c = F.cross_entropy(y_s, source_labels)
                            loss_d = self.domain_adv(y_s, f_s, y_t, f_t)
                            loss = loss_c + tradeoff[0] * loss_d
                            epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                            epoch_acc['Discriminator']  += self.domain_adv.domain_discriminator_accuracy
                            
                            epoch_loss['Source Classifier'] += loss_c
                            epoch_loss['Discriminator'] += loss_d

                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred = self.model(target_data)
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
