'''
Paper: You, K., Long, M., Cao, Z., Wang, J. and Jordan, M.I., 2019. Universal domain adaptation.
       In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 2720-2729).
Reference code: https://github.com/thuml/Universal-Domain-Adaptation
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


def get_source_share_weight(domain_logit, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = F.softmax(before_softmax, dim=-1)
    domain_logit = domain_logit / domain_temperature
    domain_out = torch.sigmoid(domain_logit)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator, reduction = 'mean', grl = None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

    def forward(self, f, w_s = None, w_t = None):
        f = self.grl(f)
        d = self.domain_discriminator(f)
        logit_s, logit_t = d.chunk(2, dim=0)
        d_s, d_t = torch.sigmoid(d).chunk(2, dim=0)
        
        d_label_s = torch.ones((d_s.size(0), 1)).to(d_s.device)
        d_label_t = torch.zeros((d_t.size(0), 1)).to(d_t.device)
        
        d_accuracy = 0.5 * (utils.binary_accuracy(d_s, d_label_s) \
                            + utils.binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        loss = 0.5 * (self.bce(d_s, d_label_s, w_s) + \
                                    self.bce(d_t, d_label_t, w_t))
        return logit_s, logit_t, loss, d_accuracy
    

class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.model = modules.ClassifierBase(input_size=1, num_classes=args.num_classes[0], backbone=args.backbone,
                                            dropout=args.dropout, use_batchnorm=True, use_cls_feat=0).to(self.device)
        self.domain_discri = modules.MLP(input_size=self.model.feature_dim, output_size=1,
                                         dropout=args.dropout, last=None).to(self.device)
        self.sep_discri = modules.MLP(input_size=self.model.feature_dim, output_size=1,
                                      dropout=args.dropout, last=None).to(self.device)
        grl = utils.GradientReverseLayer()
        self.domain_adv = DomainAdversarialLoss(self.domain_discri, grl=grl)
        self.sep_adv = DomainAdversarialLoss(self.sep_discri, grl=grl)
        self._init_data()
        self.w_0 = -0.5

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer = self._get_optimizer([self.model, self.domain_discri, self.sep_discri])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = len(self.dataloaders[self.src]) 

    def save_model(self):
        torch.save({
            'model': self.model.state_dict(),
            'sep_discri': self.sep_discri.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])
        self.sep_discri.load_state_dict(ckpt['sep_discri'])
    
    def _set_to_train(self):
        self.model.train()
        self.domain_discri.train()
        self.sep_discri.train()
    
    def _set_to_eval(self):
        self.model.eval()
        self.sep_discri.eval()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src)

            # forward
            self.optimizer.zero_grad()
            data = torch.cat((source_data, target_data), dim=0)
            y, f = self.model(data)
            prob_sep_s, prob_sep_t, loss_adv_sep, acc_sep = self.sep_adv(f.detach())
            y_s, y_t = y.chunk(2, dim=0)

            source_share_weight = get_source_share_weight(prob_sep_s, y_s)
            source_share_weight = normalize_weight(source_share_weight)
            target_share_weight = get_target_share_weight(prob_sep_t, y_t)
            target_share_weight = normalize_weight(target_share_weight)

            # compute loss
            _, _, loss_adv_dom, acc_dom = self.domain_adv(f, w_s=source_share_weight, w_t=target_share_weight)
            loss_c = F.cross_entropy(y_s, source_labels)
            loss = loss_c + self.tradeoff[0] * loss_adv_dom + self.tradeoff[1] * loss_adv_sep

            # log information
            epoch_acc['Source Data']  += self._get_accuracy(y_s, source_labels)
            epoch_acc['Domain Discriminator']  += acc_dom
            epoch_acc['Sep Discriminator']  += acc_sep
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['Domain Discriminator'] += loss_adv_dom
            epoch_loss['Sep Discriminator']  += loss_adv_sep

            # backward
            loss.backward()
            self.optimizer.step()
        return epoch_acc, epoch_loss
                            
    def _eval(self, data, actual_labels, correct, total):
        f, pred = self.model.C(self.model.G(data))
        domain_logit = self.sep_discri(f)
        target_share_weight = get_target_share_weight(domain_logit, pred, domain_temperature=1.0,
                                                      class_temperature=1.0).view(-1)
        actual_pred = self._get_actual_label(pred, idx=0)
        actual_pred = torch.where(target_share_weight >= self.w_0, actual_pred, -1)
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False, idx=0, mode='detect_unknown')
        correct['acc'] += output[0]; total['acc'] += output[1]
        return correct, total
        
