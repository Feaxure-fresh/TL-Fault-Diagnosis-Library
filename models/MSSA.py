import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import utils
from train_utils import InitTrain
import model_base


def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        
        self.G_shared = model_base.FeatureExtractor(input_size=1, output_size=1024, dropout=args.dropout).to(self.device)
        # self.Gs_specific = nn.ModuleList([model_base.ClassifierMLP(input_size=1024, output_size=1024,
        #                                                   dropout=args.dropout, last='relu') \
        #                                                   for _ in range(self.num_source)]).to(self.device)
        self.Cs = nn.ModuleList([model_base.ClassifierMLP(input_size=1024, output_size=args.num_classes,
                                                          dropout=args.dropout, last=None) \
                                                          for _ in range(self.num_source)]).to(self.device)
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
                    kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
    
    def train(self):
        args = self.args
        self._init_data()
        
        if args.train_mode == 'supervised':
            raise Exception("This model cannot be trained in suprevised mode.")
        else:
            src = args.source_name
        
        # self.optimizer = self._get_optimizer([self.G_shared, self.Gs_specific, self.Cs])
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
            for phase in ['train', 'val']:
                epoch_acc = defaultdict(float)
   
                # Set model to train mode or evaluate mode
                if phase == 'train':
                    self.G_shared.train()
                    # self.Gs_specific.train()
                    self.Cs.train()
                    epoch_loss = defaultdict(float)
                    tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
                else:
                    self.G_shared.eval()
                    # self.Gs_specific.eval()
                    self.Cs.eval()
                
                num_iter = len(self.iters[phase])                
                for i in tqdm(range(num_iter), ascii=True):
                    target_data, target_labels = utils.get_next_batch(self.dataloaders,
                    						 self.iters, phase, self.device)                 
                                            
                    if phase == 'train':
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
                        with torch.set_grad_enabled(True):
                            # forward
                            self.optimizer.zero_grad()
                            data = torch.cat(source_data+[target_data], dim=0)
                            f_shared = self.G_shared(data)
                            f = f_shared.chunk(self.num_source+1, dim=0)
                            
                            # f_specific = [self.Gs_specific[j](f_shared[j]) for j in range(self.num_source)]
                            # f_specific_t = [fs(f_shared_t) for fs in self.Gs_specific]
                            # y_s = [self.Cs[j](f_specific[j]) for j in range(self.num_source)]
                            
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
                    else:
                        with torch.no_grad():
                            feat_tgt = self.G_shared(target_data)
                            # feat_tgt = [self.Gs_specific[j](feat_tgt) for j in range(self.num_source)]
                            # logits_tgt = [self.Cs[j](feat_tgt[j]) for j in range(self.num_source)]
                            logits_tgt = [self.Cs[j](feat_tgt) for j in range(self.num_source)]
                            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
                            
                            pred = torch.zeros((logits_tgt[0].shape)).to(self.device)
                            for j in range(self.num_source):
                                pred += logits_tgt[j]
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
            
