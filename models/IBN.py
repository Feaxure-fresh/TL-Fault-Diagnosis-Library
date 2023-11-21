'''
Paper: Pan, X., Luo, P., Shi, J. and Tang, X., 2018. Two at once: Enhancing learning and generalization capacities via ibn-net.
       In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 464-479).
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain


class IBN(nn.Module):
    def __init__(self, in_channel):
        super(IBN, self).__init__()
        self.half_channel = int(in_channel/2)
        self.IN = nn.InstanceNorm1d(self.half_channel)
        self.BN = nn.BatchNorm1d(self.half_channel)
    
    def forward(self, x):
        split = torch.split(x, self.half_channel, dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), dim=1)
        return out


class IBNlayer(nn.Module):

    def __init__(self,
                 in_channel=1, kernel_size=8, stride=1, padding=1,
                 mp_kernel_size=2, mp_stride=2, dropout=0.):
        super(IBNlayer, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, stride=stride, padding=padding),
            IBN(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            IBN(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer5 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())
        
        dp = nn.Dropout(dropout)
        
        self.fs = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            dp,
            layer5)

    def forward(self, tar, x=None, y=None):
        h = self.fs(tar)
        
        return h


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
        self.model = nn.Sequential(
            model_base.FeatureExtractor(in_channel=1, block=IBNlayer, dropout=args.dropout),
            model_base.ClassifierMLP(output_size, args.num_classes, args.dropout, last=None)).to(self.device)
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
        
            num_iter = len(self.dataloaders['train'])              
            for i in tqdm(range(num_iter), ascii=True):
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
                                            self.iters, src, self.device)
                # forward
                self.optimizer.zero_grad()
                pred = self.model(source_data)
                loss = F.cross_entropy(pred, source_labels)
                epoch_acc['Source Data']  += utils.get_accuracy(pred, source_labels)
                
                epoch_loss['Source Classifier'] += loss

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
