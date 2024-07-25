'''
Paper: Chen, X., Wang, S., Long, M. and Wang, J., 2019, May. Transferability vs. 
       discriminability: Batch spectral penalization for adversarial domain adaptation. 
       In International conference on machine learning (pp. 1081-1090). PMLR.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


class BatchSpectralPenalizationLoss(nn.Module):

    def __init__(self,  bsp_tradeoff):
        super(BatchSpectralPenalizationLoss, self).__init__()
        
        self.bsp_tradeoff= bsp_tradeoff

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.svd(f_s)
        _, s_t, _ = torch.svd(f_t)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return self.bsp_tradeoff * loss


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.model = modules.ClassifierBase(input_size=1, num_classes=args.num_classes[0],
                                            backbone=args.backbone, dropout=args.dropout).to(self.device)
        self.domain_discri = modules.MLP(input_size=self.model.feature_dim, output_size=1,
                                         dropout=args.dropout, last='sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
        self.bsp = BatchSpectralPenalizationLoss(bsp_tradeoff=2e-4)
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
            y_s, _ = y.chunk(2, dim=0)

            # compute loss
            loss_c = F.cross_entropy(y_s, source_labels)
            loss_d, acc_d = self.domain_adv(f_s, f_t)
            loss_bsp = self.bsp(f_s, f_t)
            loss = loss_c + self.tradeoff[0] * loss_d + self.tradeoff[1] * loss_bsp

            # log information
            epoch_acc['Source Data']  += self._get_accuracy(y_s, source_labels)
            epoch_acc['Discriminator']  += acc_d
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['Discriminator'] += loss_d
            epoch_loss['BSP'] += loss_bsp

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
