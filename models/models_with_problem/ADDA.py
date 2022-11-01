import torch
import logging
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

if __name__ == '__main__': 
    import sys
    sys.path.extend(['../..', '../data_loader'])

import utils
from train_utils import InitTrain
import model_base


class ADDA(model_base.BaseModel):
    def __init__(self,
                 input_size,
                 output_size,
                 num_classes,
                 dropout):
        super(ADDA, self).__init__(input_size, output_size, num_classes, dropout)
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

    def get_models(self, optimize_c=True):
        if optimize_c:
            models = [self.G, self.C]
        else:
            models = self.G
        return models


class Trainset(InitTrain):
    
    def __init__(self, args, save_dir):
        super(Trainset, self).__init__(args, save_dir)
        
        self.domain_discri = model_base.ClassifierMLP(input_size=1024, output_size=1,
                        dropout=args.dropout, last='sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
    
    def train(self):
        args = self.args
        self._init_data()
        
        self.pretrain_model = ADDA(input_size=1, output_size=1024,
                                num_classes=args.num_classes, dropout=args.dropout).to(self.device)
        
        if args.train_mode == 'supervised':
            raise Exception("This model cannot be trained in suprevised mode.")
        elif args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained with multi-source data.")

        self.pretrain_op = self._get_optimizer(self.pretrain_model)
        self.pretrain_lr_scheduler = self._get_lr_scheduler(self.pretrain_op)
        for epoch in range(args.max_epoch+1):
            logging.info('-'*5 + 'Source-iter Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.pretrain_lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.pretrain_lr_scheduler.get_last_lr()))
            epoch_acc = defaultdict(float)
   
            # Set model to train mode or evaluate mode
            self.pretrain_model.train()
            epoch_loss = defaultdict(float)
            
            num_source_data = [len(self.iters[s]) for s in args.source_name]
            num_iter = sum(num_source_data)
            
            for i in tqdm(range(num_iter), ascii=True):
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
             						     self.iters, src, self.device)

                with torch.set_grad_enabled(True):
                    # forward
                    self.pretrain_op.zero_grad()
                    
                    y, _ = self.pretrain_model(source_data)
            
                    loss_c = F.cross_entropy(y, source_labels)
                    epoch_acc['Source Data']  += utils.get_accuracy(y, source_labels)                   
                    epoch_loss['Source Classifier'] += loss_c

                    # backward
                    loss_c.backward()
                    self.pretrain_op.step() 
    
            # Print the train and val information via each epoch
            for key in epoch_loss.keys():
                logging.info('{}-Loss {}: {:.4f}'.format('Source-iter train', key, epoch_loss[key]/num_iter))
            for key in epoch_acc.keys():
                logging.info('{}-Acc {}: {:.4f}'.format('Source-iter train', key, epoch_acc[key]/num_iter))
            
            if self.pretrain_lr_scheduler is not None:
                self.pretrain_lr_scheduler.step()
        
        self.model = copy.deepcopy(self.pretrain_model)
        self.optimizer = self._get_optimizer([self.model.get_models(False), self.domain_discri])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        utils.freeze_net(self.pretrain_model)
        self.pretrain_model.freeze_bn()
        
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(args.max_epoch+1):
            logging.info('-'*5 + 'Target-iter {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
   
            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                epoch_acc = defaultdict(float)
   
                # Set model to train mode or evaluate mode
                if phase == 'train':
                    self.model.train()
                    epoch_loss = defaultdict(float)
                else:
                    self.model.eval()
                
                num_iter = len(self.iters[phase])
                if args.train_mode == 'source_combine':
                    num_iter *= len(src)
                
                for i in tqdm(range(num_iter), ascii=True):
                    target_data, target_labels = utils.get_next_batch(self.dataloaders,
                    						 self.iters, phase, self.device)
                    source_data, source_labels = utils.get_next_batch(self.dataloaders,
                						     self.iters, src, self.device)
                    
                    if phase == 'train':
                        with torch.set_grad_enabled(True):
                            # forward
                            self.optimizer.zero_grad()
                            
                            _, f_s = self.pretrain_model(source_data)
                            _, f_t = self.model(target_data)
                                                        
                            loss_d, acc_d = self.domain_adv(f_s, f_t)
                            epoch_acc['Discriminator']  += acc_d
                            epoch_loss['Discriminator'] += loss_d

                            # backward                            
                            loss_d.backward()
                            self.optimizer.step()    
                    else:
                        with torch.no_grad():
                            pred = self.model(target_data)
                            epoch_acc['Target Data']  += utils.get_accuracy(pred, target_labels)
                                
                # Print the train and val information via each epoch
                if phase == 'train':
                    for key in epoch_loss.keys():
                        logging.info('Target-iter {}-Loss {}: {:.4f}'.format(phase, key, epoch_loss[key]/num_iter))
                for key in epoch_acc.keys():
                    logging.info('Target-iter {}-Acc {}: {:.4f}'.format(phase, key, epoch_acc[key]/num_iter))
                
                
                # log the best model according to the val accuracy
                if phase == 'val':
                    new_acc = epoch_acc['Target Data']/num_iter
                    if new_acc >= best_acc:
                        best_acc = new_acc
                        best_epoch = epoch
                    logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
                    
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
