'''
Paper: Xu, R., Li, G., Yang, J. and Lin, L., 2019. Larger norm more transferable: An adaptive feature norm approach for unsupervised domain adaptation.
       In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1426-1435).
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase       
            

class AdaptiveFeatureNorm(nn.Module):

    def __init__(self, delta):
        super(AdaptiveFeatureNorm, self).__init__()
        self.delta = delta

    def forward(self, f):
        radius = f.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + self.delta
        loss = ((f.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss


class ModelAFN(modules.ClassifierBase):

    def __init__(self,
                 input_size,
                 num_classes,
                 backbone,
                 dropout,
                 num_layer=3):
        super(ModelAFN, self).__init__(input_size,
                                       num_classes,
                                       backbone,
                                       dropout,
                                       num_layer,
                                       use_batchnorm=True,
                                       use_cls_feat=1)
        self.dropout = dropout
                
    def forward(self, input):
        f, predictions = self.C(self.G(input))
        
        if self.training:
            f = f * math.sqrt(1 - self.dropout)
        
        if self.training:
            return predictions, f
        else:
            return predictions
                    

class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.model = ModelAFN(input_size=1, num_classes=args.num_classes[0], backbone=args.backbone,
                              dropout=args.dropout).to(self.device)
        self.adaptive_feature_norm = AdaptiveFeatureNorm(delta=1).to(self.device)
        self._init_data()

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer = self._get_optimizer(self.model)
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
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            # compute loss
            loss_norm = self.adaptive_feature_norm(f_s) + self.adaptive_feature_norm(f_t)
            loss_c = F.cross_entropy(y_s, source_labels)
            loss_ent = utils.entropy(F.softmax(y_t, dim=1), reduction='mean')
            loss = loss_c + self.tradeoff[0] * loss_norm + self.tradeoff[1] * loss_ent

            # log information
            epoch_acc['Source Data'] += self._get_accuracy(y_s, source_labels)
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['Norm loss'] += loss_norm
            epoch_loss['Entropy']  += loss_ent

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

