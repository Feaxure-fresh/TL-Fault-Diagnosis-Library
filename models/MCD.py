'''
Paper: Saito, K., Watanabe, K., Ushiku, Y. and Harada, T., 2018. Maximum classifier 
    discrepancy for unsupervised domain adaptation. In Proceedings of the IEEE 
    conference on computer vision and pattern recognition (pp. 3723-3732).
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain


def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predictions1 - predictions2))


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
        self.C1 = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                        dropout=args.dropout, last=None).to(self.device)
        self.C2 = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                        dropout=args.dropout, last=None).to(self.device)
        self.G = model_base.FeatureExtractor(in_channel=1).to(self.device)
        self._init_data()
    
    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'C1': self.C1.state_dict(),
            'C2': self.C2.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.C1.load_state_dict(ckpt['C1'])
        self.C2.load_state_dict(ckpt['C2'])
        
    def train(self):
        args = self.args
        
        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")

        self.optimizer_G = self._get_optimizer(self.G)
        self.optimizer_C = self._get_optimizer([self.C1, self.C2])
        self.lr_scheduler_G = self._get_lr_scheduler(self.optimizer_G)
        self.lr_scheduler_C = self._get_lr_scheduler(self.optimizer_C)
        
        best_acc = 0.0
        best_epoch = 0
   
        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler_G is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler_G.get_last_lr()))
   
            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)
   
            # Set model to train mode or evaluate mode
            self.G.train()
            self.C1.train()
            self.C2.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            num_iter = len(self.dataloaders['train'])               
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
            						     self.iters, src, self.device)
                # forward
                data = torch.cat((source_data, target_data), dim=0)
                self.optimizer_G.zero_grad()
                self.optimizer_C.zero_grad()
                
                f = self.G(data)
                y_1 = self.C1(f)
                y_2 = self.C2(f)
                y_1, _ = y_1.chunk(2, dim=0)
                y_2, _ = y_2.chunk(2, dim=0)
                f_s, f_t = f.chunk(2, dim=0)
                loss = F.cross_entropy(y_1, source_labels) + F.cross_entropy(y_2, source_labels)
                
                epoch_acc['Classifier 1 source train']  += utils.get_accuracy(y_1, source_labels)
                epoch_acc['Classifier 2 source train']  += utils.get_accuracy(y_2, source_labels)
                epoch_loss['Step 1: Source domain'] += loss
                loss.backward()
                self.optimizer_G.step()
                self.optimizer_C.step()
                
                self.optimizer_G.zero_grad()
                self.optimizer_C.zero_grad()
                 
                f = self.G(data)
                y_1 = self.C1(f)
                y_2 = self.C2(f)
                y1_s, y1_t = y_1.chunk(2, dim=0)
                y2_s, y2_t = y_2.chunk(2, dim=0)
                f_s, f_t = f.chunk(2, dim=0)
                y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
                loss = F.cross_entropy(y1_s, source_labels) + F.cross_entropy(y2_s, source_labels)  \
                                 - classifier_discrepancy(y1_t, y2_t) * tradeoff[0]
                epoch_loss['Step 2: Maximize discrepancy'] += loss
                loss.backward()
                self.optimizer_C.step()

                for k in range(4):
                    self.optimizer_G.zero_grad()
                    f = self.G(target_data)
                    y_1 = self.C1(f)
                    y_2 = self.C2(f)
                    y1_t, y2_t = F.softmax(y_1, dim=1), F.softmax(y_2, dim=1)
                    loss_mcd = classifier_discrepancy(y1_t, y2_t)
                    loss = loss_mcd * tradeoff[0]
                    epoch_loss['Step 3: Minimize discrepancy'] += loss_mcd
                    loss.backward()
                    self.optimizer_G.step()
                
            # Print the train and val information via each epoch
            for key in epoch_loss.keys():
                if key == 'Step 3: Minimize discrepancy':
                    logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/(4.*num_iter)))
                else:
                    logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
            
            # log the best model according to the val accuracy
            new_acc = self.test()
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
                    
            if self.lr_scheduler_G is not None:
                self.lr_scheduler_G.step()
            if self.lr_scheduler_C is not None:
                self.lr_scheduler_C.step()
    
    def test(self):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        epoch_acc = defaultdict(float)
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                f = self.G(target_data)
                y_1 = self.C1(f)
                y_2 = self.C2(f)
                pred = y_1 + y_2
                epoch_acc['Classifier 1 Target Data']  += utils.get_accuracy(y_1, target_labels)
                epoch_acc['Classifier 2 Target Data']  += utils.get_accuracy(y_2, target_labels)
                epoch_acc['Target Data']  += utils.get_accuracy(pred, target_labels)
        for key in epoch_acc.keys():
            logging.info('Val-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
        return epoch_acc['Target Data']/num_iter
