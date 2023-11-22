'''
Paper: Li, D., Yang, Y., Song, Y.Z. and Hospedales, T., 2018, April. Learning to generalize: Meta-learning for domain generalization.
       In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import higher
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
                    kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
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
        src = args.source_name
        
        if args.train_mode != 'multi_source':
            raise Exception("For this model, invalid train mode: {}".format(args.train_mode))

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
                source_data, source_labels = [], []
                for idx in range(self.num_source):
                    source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
                                                self.iters, src[idx], self.device)
                    source_data.append(source_data_item)
                    source_labels.append(source_labels_item)
                idx_range = torch.arange(0, self.num_source)
                train_idx = np.random.choice(idx_range, size=(self.num_source - 1,), replace=False)
                test_idx = np.setdiff1d(idx_range, train_idx)
                
                self.optimizer.zero_grad()
                with higher.innerloop_ctx(self.model, self.optimizer, copy_initial_weights=False) as (inner_model, inner_optimizer):
                    for _ in range(1):  # Single gradient update (for simplicity)
                        loss_inner = 0
                        for idx in train_idx:
                            y, _ = inner_model(source_data[idx])
                            loss_inner += F.cross_entropy(y, source_labels[idx]) / len(train_idx)
                        inner_optimizer.step(loss_inner)

                    loss_outer = 0
                    cls_acc = 0

                    for idx in train_idx:
                        y, _ = self.model(source_data[idx])
                        loss_outer += F.cross_entropy(y, source_labels[idx]) / len(train_idx)

                    for idx in test_idx:
                        y, _ = inner_model(source_data[idx])
                        loss_outer += F.cross_entropy(y, source_labels[idx]) * tradeoff[0] / len(test_idx)
                        cls_acc += utils.get_accuracy(y, source_labels[idx]) / len(test_idx)
                epoch_acc['Source Data']  += cls_acc
                
                epoch_loss['Meta-train'] += loss_inner
                epoch_loss['Meta_test'] += loss_outer

                # backward
                loss_outer.backward()
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
