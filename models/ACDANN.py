'''
Paper: Wang, Q., Taal, C. and Fink, O., 2021. Integrating expert knowledge with domain adaptation 
       for unsupervised fault diagnosis. IEEE Transactions on Instrumentation and Measurement, 71, pp.1-12.
Reference code: https://github.com/qinenergy/syn2real
Note: Augmented Conditional Domain Alignment Neural Network (ACDANN) is not an official name in the paper.
'''
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import modules
from train_utils import TrainerBase


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.model = modules.ClassifierBase(input_size=1, num_classes=args.num_classes[0],
                                            backbone=args.backbone, dropout=args.dropout).to(self.device)
        self.discriminator = modules.MLP(input_size=args.num_classes[0]*self.model.feature_dim, output_size=2,
                                         dropout=args.dropout, last=None).to(self.device)
        self.grl = utils.GradientReverseLayer()
        self.dist_beta = torch.distributions.beta.Beta(1., 1.)
        self._init_data()

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer = self._get_optimizer([self.model, self.discriminator])
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
        self.discriminator.train()
    
    def _set_to_eval(self):
        self.model.eval()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src)
            
            # forward
            batch_size = source_data.shape[0]
            self.optimizer.zero_grad()
            data = torch.cat((source_data, target_data), dim=0)
            y, f = self.model(data)
            f_s, f_t = f.chunk(2, dim=0)
            y_s, y_t = y.chunk(2, dim=0)
            
            softmax_output_src = F.softmax(y_s, dim=-1)
            softmax_output_tgt = F.softmax(y_t, dim=-1)
            
            lmb = self.dist_beta.sample((batch_size, 1)).to(self.device)
            labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.long),
                    torch.zeros(batch_size, dtype=torch.long)), dim=0).to(self.device)
    
            idxx = np.arange(batch_size)
            np.random.shuffle(idxx)
            f_s = lmb * f_s + (1.-lmb) * f_s[idxx]
            f_t = lmb * f_t + (1.-lmb) * f_t[idxx]

            softmax_output_src = lmb * softmax_output_src + (1.-lmb) * softmax_output_src[idxx]
            softmax_output_tgt = lmb * softmax_output_tgt + (1.-lmb) * softmax_output_tgt[idxx]
                                            
            feat_src_ = torch.bmm(softmax_output_src.unsqueeze(2),
                                    f_s.unsqueeze(1)).view(batch_size, -1)
            feat_tgt_ = torch.bmm(softmax_output_tgt.unsqueeze(2),
                                    f_t.unsqueeze(1)).view(batch_size, -1)

            feat = self.grl(torch.concat((feat_src_, feat_tgt_), dim=0))
            logits_dm = self.discriminator(feat)
            
            # compute loss
            loss_dm = F.cross_entropy(logits_dm, labels_dm)
            loss_c = F.cross_entropy(y_s, source_labels)
            loss = loss_c + self.tradeoff[0] * loss_dm
            
            # log information
            epoch_acc['Source Data']  += self._get_accuracy(y_s, source_labels)
            epoch_acc['Discriminator']  += self._get_accuracy(logits_dm, labels_dm)
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['Discriminator'] += loss_dm

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
