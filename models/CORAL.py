'''
Paper: Sun, B. and Saenko, K., 2016. Deep coral: Correlation alignment for deep domain 
    adaptation. In Computer Vision–ECCV 2016 Workshops: Amsterdam,
    The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14 
    (pp. 443-450). Springer International Publishing.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain


class CorrelationAlignmentLoss(nn.Module):

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        
        self.coral = CorrelationAlignmentLoss()
        self.model = model_base.BaseModel(input_size=1, num_classes=args.num_classes,
                                      dropout=args.dropout).to(self.device)
        self._init_data()
    
    def save_model(self):
        torch.save({
            'model': self.model.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])
        
    def train(self):
        args = self.args
        
        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        
        best_acc = 0.0
        best_epoch = 0
   
        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
   
            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)
   
            # Set model to train mode or evaluate mode
            self.model.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            num_iter = len(self.dataloaders['train'])                 
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
            						     self.iters, src, self.device)
                # forward
                self.optimizer.zero_grad()
                data = torch.cat((source_data, target_data), dim=0)
                y, f = self.model(data)
                src_feat, tgt_feat = f.chunk(2, dim=0)
                pred, _ = y.chunk(2, dim=0)
                
                loss_penalty = self.coral(src_feat, tgt_feat)
                loss_c = F.cross_entropy(pred, source_labels)
                loss = loss_c + tradeoff[0] * loss_penalty
                epoch_acc['Source Data']  += utils.get_accuracy(pred, source_labels)
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['CORAL'] += loss_penalty

                # backward
                loss.backward()
                self.optimizer.step()
                
            # Print the train and val information via each epoch
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
                            
            # log the best model according to the val accuracy
            new_acc = self.test()
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
    def test(self):
        self.model.eval()
        acc = 0.0
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred = self.model(target_data)
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc