'''
Paper: Wang, Q., Taal, C. and Fink, O., 2021. Integrating expert knowledge with domain adaptation 
    for unsupervised fault diagnosis. IEEE Transactions on Instrumentation and Measurement, 71, pp.1-12.
Reference code: https://github.com/qinenergy/syn2real
Note: Augmented Conditional Domain Alignment Neural Network (ACDANN) was not an official name in the paper.
'''
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
        self.discriminator = model_base.ClassifierMLP(input_size=args.num_classes*output_size, output_size=2,
                        dropout=args.dropout, last=None).to(self.device)
        self.grl = utils.GradientReverseLayer()
        self.dist_beta = torch.distributions.beta.Beta(1., 1.)
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
        
        self.optimizer = self._get_optimizer([self.model, self.discriminator])
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
            self.discriminator.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            num_iter = len(self.dataloaders['train'])
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
            						     self.iters, src, self.device)
                # forward
                batch_size = source_data.shape[0]
                self.optimizer.zero_grad()
                data = torch.cat((source_data, target_data), dim=0)
                
                y, f = self.model(data)
                f_s, f_t = f.chunk(2, dim=0)
                y_s, y_t = y.chunk(2, dim=0)
                
                loss_c = F.cross_entropy(y_s, source_labels)
                
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
                loss_dm = F.cross_entropy(logits_dm, labels_dm)
                loss = loss_c + tradeoff[0] * loss_dm
                
                epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                epoch_acc['Discriminator']  += utils.get_accuracy(logits_dm, labels_dm)
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['Discriminator'] += loss_dm

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
