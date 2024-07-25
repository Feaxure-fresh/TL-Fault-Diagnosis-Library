'''
Paper: Long, M., Cao, Z., Wang, J. and Jordan, M.I., 2018. Conditional adversarial
       domain adaptation. Advances in neural information processing systems, 31.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


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

        weight = 1.0 + torch.exp(-utils.entropy(g))
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


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.model = modules.ClassifierBase(input_size=1, num_classes=args.num_classes[0],
                                            backbone=args.backbone, dropout=args.dropout).to(self.device)
        self.domain_discri = modules.MLP(input_size=self.model.feature_dim*args.num_classes[0], output_size=1,
                                         dropout=args.dropout, last='sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = ConditionalDomainAdversarialLoss(self.domain_discri, grl=grl)
        self._init_data()

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")

        self.optimizer = self._get_optimizer([self.model, self.domain_discri])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = len(self.dataloaders[self.src])

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
        self.domain_discri.train()
    
    def _set_to_eval(self):
        self.model.eval()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            target_data, _ = self._get_next_batch('train')                  
            source_data, source_labels = self._get_next_batch(self.src)

            # forward
            self.optimizer.zero_grad()
            data = torch.cat((source_data, target_data), dim=0)
            y, f = self.model(data)
            f_s, f_t = f.chunk(2, dim=0)
            y_s, y_t = y.chunk(2, dim=0)

            # compute loss
            loss_c = F.cross_entropy(y_s, source_labels)
            loss_d = self.domain_adv(y_s, f_s, y_t, f_t)
            loss = loss_c + self.tradeoff[0] * loss_d

            # log information
            epoch_acc['Source Data']  += self._get_accuracy(y_s, source_labels)
            epoch_acc['Discriminator']  += self.domain_adv.domain_discriminator_accuracy
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['Discriminator'] += loss_d

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
    
