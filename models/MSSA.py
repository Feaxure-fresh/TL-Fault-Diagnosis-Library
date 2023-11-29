'''
Paper: Tian, J., Han, D., Li, M. and Shi, P., 2022. A multi-source information transfer 
    learning method with subdomain adaptation for cross-domain fault diagnosis.
    Knowledge-Based Systems, 243, p.108466.
Note: The code has been developed in accordance with the methodologies described in the paper.
    Should there be any discrepancies in performance relative to the published results, we welcome
    feedback on potential errors or provision of source code. During our testing, we observed that
    certain feature extractors and the LMMD technique outlined in the paper may lead to a decrease
    in performance. Those elements can refer to the sections of the code that are commented out.
Author: Feaxure
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


def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
        self.G_shared = model_base.FeatureExtractor(in_channel=1).to(self.device)
        '''
        # Specific feature extractors defined in the paper will not be used.
        self.Gs_specific = nn.ModuleList([nn.Sequential(
                                                    nn.Dropout(args.dropout),
                                                    nn.Linear(output_size, output_size),
                                                    nn.ReLU()) \
                                                    for _ in range(self.num_source)]).to(self.device)
        '''
        self.Cs = nn.ModuleList([model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                                                          dropout=args.dropout, last=None) \
                                                          for _ in range(self.num_source)]).to(self.device)
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
                    kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
    
    def save_model(self):
        torch.save({
            'G_shared': self.G_shared.state_dict(),
            # 'Gs_specific': self.Gs_specific.state_dict(),
            'Cs': self.Cs.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G_shared.load_state_dict(ckpt['G_shared'])
        # self.Gs_specific.load_state_dict(ckpt['Gs_specific'])
        self.Cs.load_state_dict(ckpt['Cs'])
        
    def train(self):
        args = self.args
        self._init_data()
        src = args.source_name
        
        '''
        # Specific feature extractors defined in the paper will not be used.
        self.optimizer = self._get_optimizer([self.G_shared, self.Gs_specific, self.Cs])
        '''
        self.optimizer = self._get_optimizer([self.G_shared, self.Cs])
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
            self.G_shared.train()
            '''
            # Specific feature extractors defined in the paper will not be used.
            self.Gs_specific.train()
            '''
            self.Cs.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            num_iter = len(self.dataloaders['train'])                
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)                 
                source_data, source_labels = [], []
                if args.train_mode == 'source_combine':
                    source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
                						     self.iters, src, self.device)
                    source_data.append(source_data_item)
                    source_labels.append(source_labels_item)
                else:
                    for idx in range(self.num_source):
                        source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
                    						     self.iters, src[idx], self.device)
                        source_data.append(source_data_item)
                        source_labels.append(source_labels_item)
                # forward
                self.optimizer.zero_grad()
                data = torch.cat(source_data+[target_data], dim=0)
                f_shared = self.G_shared(data)
                f = f_shared.chunk(self.num_source+1, dim=0)
                
                '''
                # Specific feature extractors defined in the paper will not be used.
                f = [self.Gs_specific[j](f[j]) for j in range(self.num_source)]
                '''
                
                '''
                # LMMD defined in the paper will not be used.
                f_specific_t = [fs(f_shared_t) for fs in self.Gs_specific]
                y_s = [self.Cs[j](f_specific[j]) for j in range(self.num_source)]
                '''
                
                y_s = [self.Cs[j](f[j]) for j in range(self.num_source)]
                loss_cls = 0.0
                for j in range(self.num_source):
                    loss_cls += F.cross_entropy(y_s[j], source_labels[j])
                loss_sum_mmd = 0.0
                for k in range(self.num_source):
                    loss_sum_mmd += self.mkmmd(f[k], f[-1])
                
                '''
                # LMMD defined in the paper will not be used.
                y_t = [self.Cs[j](f_specific_t[j]) for j in range(self.num_source)]
                y_t = [F.softmax(data, dim=1) for data in y_t] 
                for k in range(self.num_source):
                    oh_label = one_hot(source_labels[k], args.num_classes)
                    for j in range(args.num_classes):
                        w_src = oh_label[:, j].view(-1, 1).to(self.device)
                        w_tgt = y_t[k][:, j].view(-1, 1).to(self.device)
                        loss_sum_mmd += self.mkmmd(w_src*f_specific[k], w_tgt*f_specific_t[k])
                '''
                
                loss = loss_cls + tradeoff[0] * loss_sum_mmd
                
                for j in range(self.num_source):
                    epoch_acc['Source Data %d'%j] += utils.get_accuracy(y_s[j], source_labels[j])
                
                epoch_loss['Source Classifier'] += loss_cls
                epoch_loss['MMD'] += loss_sum_mmd
                
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
        self.G_shared.eval()
        '''
        # Specific feature extractors defined in the paper will not be used.
        self.Gs_specific.eval()
        '''
        self.Cs.eval()
        acc = 0.0
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                feat_tgt = self.G_shared(target_data)
                '''
                # Specific feature extractors defined in the paper will not be used.
                feat_tgt = [self.Gs_specific[j](feat_tgt) for j in range(self.num_source)]
                logits_tgt = [self.Cs[j](feat_tgt[j]) for j in range(self.num_source)]
                '''
                logits_tgt = [self.Cs[j](feat_tgt) for j in range(self.num_source)]
                logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
                
                pred = torch.zeros((logits_tgt[0].shape)).to(self.device)
                for j in range(self.num_source):
                    pred += logits_tgt[j]
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc
