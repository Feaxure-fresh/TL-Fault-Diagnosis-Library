import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain
        

class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
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
                source_data, labels_per_domain = [], []
                for idx in range(self.num_source):
                    source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
                                                self.iters, src[idx], self.device)
                    source_data.append(source_data_item)
                    labels_per_domain.append(source_labels_item)
                source_data = torch.cat(source_data, dim=0)
                source_labels = torch.cat(labels_per_domain, dim=0)
                
                # forward
                self.optimizer.zero_grad()
                pred_all, _ = self.model(source_data)
                pred_per_domain = pred_all.chunk(self.num_source, dim=0)

                loss_ce_per_domain = torch.zeros(self.num_source).to(self.device)
                for idx in range(self.num_source):
                    loss_ce_per_domain[idx] = F.cross_entropy(pred_per_domain[idx], labels_per_domain[idx])

                # cls loss
                loss_ce = loss_ce_per_domain.mean()
                # penalty loss
                loss_penalty = ((loss_ce_per_domain - loss_ce) ** 2).mean()

                loss = loss_ce + tradeoff[0] * loss_penalty
                epoch_acc['Source Data']  += utils.get_accuracy(pred_all, source_labels)
                
                epoch_loss['Source Classifier'] += loss_ce
                epoch_loss['Risk Variance'] += loss_penalty

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
