import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import utils
from train_utils import InitTrain
import model_base


class BatchSpectralPenalizationLoss(nn.Module):

    def __init__(self,  bsp_tradeoff):
        super(BatchSpectralPenalizationLoss, self).__init__()
        
        self.bsp_tradeoff= bsp_tradeoff

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.svd(f_s)
        _, s_t, _ = torch.svd(f_t)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return self.bsp_tradeoff * loss


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        
        self.domain_discri = model_base.ClassifierMLP(input_size=1024, output_size=1,
                        dropout=args.dropout, last='sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
        self.bsp = BatchSpectralPenalizationLoss(bsp_tradeoff=2e-4)
        self.model = model_base.BaseModel(input_size=1, output_size=1024,
                                     num_classes=args.num_classes, dropout=args.dropout).to(self.device)
        
    def train(self):
        args = self.args
        self._init_data()
        
        if args.train_mode == 'supervised':
            src = None
        elif args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained with multi-source data.")

        self.optimizer = self._get_optimizer([self.model, self.domain_discri])
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
                    self.model.train()
                    self.domain_discri.train()
                    epoch_loss = defaultdict(float)
                    tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
                else:
                    self.model.eval()
                
                num_iter = len(self.iters[phase])               
                for i in tqdm(range(num_iter), ascii=True):
                    target_data, target_labels = utils.get_next_batch(self.dataloaders,
                    						 self.iters, phase, self.device)                    
                    if phase == 'train':
                        if src != None:
                            source_data, source_labels = utils.get_next_batch(self.dataloaders,
                        						     self.iters, src, self.device)
                        else:
                            source_data, source_labels = target_data, target_labels
                        with torch.set_grad_enabled(True):
                            # forward
                            self.optimizer.zero_grad()
                            data = torch.cat((source_data, target_data), dim=0)
                            
                            y, f = self.model(data)
                            f_s, f_t = f.chunk(2, dim=0)
                            y_s, _ = y.chunk(2, dim=0)
                    
                            loss_c = F.cross_entropy(y_s, source_labels)
                            loss_d, acc_d = self.domain_adv(f_s, f_t)
                            loss_bsp = self.bsp(f_s, f_t)
                            loss = loss_c + tradeoff[0] * loss_d + tradeoff[1] * loss_bsp
                            epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                            epoch_acc['Discriminator']  += acc_d
                            
                            epoch_loss['Source Classifier'] += loss_c
                            epoch_loss['Discriminator'] += loss_d
                            epoch_loss['BSP'] += loss_bsp

                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred = self.model(target_data)
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
            
