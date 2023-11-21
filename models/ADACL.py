'''
Paper: Zhang, Y., Ren, Z., Zhou, S. and Yu, T., 2020. Adversarial domain adaptation with
    classifier alignment for cross-domain intelligent fault diagnosis of multiple source domains.
    Measurement Science and Technology, 32(3), p.035102.
Note: The code is reproduced according to the paper. If there is a performance loss compared to the paper,
    please point out the mistakes or provide the source code.
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


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
        self.discriminator = model_base.ClassifierMLP(input_size=output_size, output_size=(self.num_source+1),
                        dropout=args.dropout, last=None).to(self.device)
        self.grl = utils.GradientReverseLayer()
        self.G = model_base.FeatureExtractor(in_channel=1).to(self.device)
        self.Cs = nn.ModuleList([model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                                                          dropout=args.dropout, last=None) \
                                                          for _ in range(self.num_source)]).to(self.device)
        self._init_data()
    
    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Cs': self.Cs.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.Cs.load_state_dict(ckpt['Cs'])
        
    def train(self):
        args = self.args
        src = args.source_name
        
        self.optimizer = self._get_optimizer([self.G, self.Cs, self.discriminator])
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
            self.G.train()
            self.Cs.train()
            self.discriminator.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            num_iter = len(self.dataloaders['train'])               
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)
                source_data, source_labels, src_idx = utils.get_next_batch(self.dataloaders,
            						     self.iters, src[int(i%len(args.source_name))], self.device, return_idx=True)
                if args.train_mode == 'multi_source':
                    src_idx = src_idx[0] 
                else:
                    src_idx = 0
                # forward
                self.optimizer.zero_grad()
                
                batch_size = source_data.shape[0]
                data = torch.cat((source_data, target_data), dim=0)
                
                f = self.G(data)
                f_s, f_t = f.chunk(2, dim=0)
                y_s = self.Cs[src_idx](f_s)
                y_t = [cl(f_t) for cl in self.Cs]
                
                loss_c = F.cross_entropy(y_s, source_labels)
                
                labels_dm = torch.concat((torch.full((batch_size,), src_idx+1, dtype=torch.long),
                                          torch.zeros(batch_size, dtype=torch.long)), dim=0).to(self.device)
                feat = self.grl(torch.concat((f_s, f_t), dim=0))
                logits_dm = self.discriminator(feat)
                loss_d = F.cross_entropy(logits_dm, labels_dm)
                
                logits_tgt = [F.softmax(t, dim=1) for t in y_t]
                loss_l1 = 0.0
                for k in range(self.num_source - 1):
                    for j in range(k+1, self.num_source):
                        # We use mean value of this result, even though the sum value is used in the paper.
                        loss_l1 += torch.abs(logits_tgt[k] - logits_tgt[j]).mean()
                loss_l1 /= self.num_source
           
                loss = loss_c + tradeoff[0] * loss_d + tradeoff[1] * loss_l1
                
                epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                epoch_acc['Discriminator']  += utils.get_accuracy(logits_dm, labels_dm)
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['Discriminator'] += loss_d
                epoch_loss['L1'] += loss_l1

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
        self.G.eval()
        self.Cs.eval()
        acc = 0.0
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                feat_tgt = self.G(target_data)
                logits_tgt = [cl(feat_tgt) for cl in self.Cs]
                logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
                
                pred = torch.zeros((logits_tgt[0].shape)).to(self.device)
                for j in range(self.num_source):
                    pred += logits_tgt[j]
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc
            
