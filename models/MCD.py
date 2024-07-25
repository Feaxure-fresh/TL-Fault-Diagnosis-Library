'''
Paper: Saito, K., Watanabe, K., Ushiku, Y. and Harada, T., 2018. Maximum classifier discrepancy for unsupervised domain adaptation.
       In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3723-3732).
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import modules
from train_utils import TrainerBase


def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predictions1 - predictions2))


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=1).to(self.device)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
        else:
            raise Exception(f"unknown backbone type {args.backbone}")
        self.C1 = modules.MLP(input_size=self.G.out_dim, output_size=args.num_classes[0],
                              dropout=args.dropout, last=None).to(self.device)
        self.C2 = modules.MLP(input_size=self.G.out_dim, output_size=args.num_classes[0],
                              dropout=args.dropout, last=None).to(self.device)
        self._init_data()

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer_G = self._get_optimizer(self.G)
        self.optimizer_C = self._get_optimizer([self.C1, self.C2])
        self.lr_scheduler_G = self._get_lr_scheduler(self.optimizer_G)
        self.lr_scheduler_C = self._get_lr_scheduler(self.optimizer_C)
        self.num_iter = len(self.dataloaders[self.src])
        self.num_inner_loop = 4

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

    def _set_to_train(self):
        self.G.train()
        self.C1.train()
        self.C2.train()
    
    def _set_to_eval(self):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src)

            # forward Step 1
            data = torch.cat((source_data, target_data), dim=0)
            self.optimizer_G.zero_grad()
            self.optimizer_C.zero_grad()
            f = self.G(data)
            y_1 = self.C1(f)
            y_2 = self.C2(f)
            y_1, _ = y_1.chunk(2, dim=0)
            y_2, _ = y_2.chunk(2, dim=0)

            # compute loss
            loss = F.cross_entropy(y_1, source_labels) + F.cross_entropy(y_2, source_labels)
            
            # log information
            epoch_acc['Classifier 1 source train']  += self._get_accuracy(y_1, source_labels)
            epoch_acc['Classifier 2 source train']  += self._get_accuracy(y_2, source_labels)
            epoch_loss['Step 1: Source domain'] += loss

            # backward
            loss.backward()
            self.optimizer_G.step()
            self.optimizer_C.step()
            
            # forward Step 2
            self.optimizer_G.zero_grad()
            self.optimizer_C.zero_grad()
            f = self.G(data)
            y_1 = self.C1(f)
            y_2 = self.C2(f)
            y1_s, y1_t = y_1.chunk(2, dim=0)
            y2_s, y2_t = y_2.chunk(2, dim=0)
            y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)

            # compute loss
            loss = F.cross_entropy(y1_s, source_labels) + F.cross_entropy(y2_s, source_labels)  \
                    - self.tradeoff[0] * classifier_discrepancy(y1_t, y2_t)
            
            # log information
            epoch_loss['Step 2: Maximize discrepancy'] += loss

            # backward
            loss.backward()
            self.optimizer_C.step()

            for _ in range(self.num_inner_loop):
                # forward Step 3
                self.optimizer_G.zero_grad()
                f = self.G(target_data)
                y_1 = self.C1(f)
                y_2 = self.C2(f)
                y1_t, y2_t = F.softmax(y_1, dim=1), F.softmax(y_2, dim=1)

                # compute loss
                loss_mcd = classifier_discrepancy(y1_t, y2_t)
                loss = self.tradeoff[1] * loss_mcd

                # log information
                epoch_loss['Step 3: Minimize discrepancy'] += loss_mcd

                # backward
                loss.backward()
                self.optimizer_G.step()
        return epoch_acc, epoch_loss
    
    def _eval(self, data, actual_labels, correct, total):
        f = self.G(data)
        y_1 = self.C1(f)
        y_2 = self.C2(f)
        pred = y_1 + y_2
        actual_pred = self._get_actual_label(pred, idx=0)
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]; total['acc'] += output[1]
        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(actual_pred, actual_labels, return_acc=False, idx=0, mode='closed-set')
            correct['Closed-set-acc'] += output[0]; total['Closed-set-acc'] += output[1]
        return correct, total

    def train(self):
        args = self.args
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
            self._set_to_train()
            epoch_loss = defaultdict(float)
            self.tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            epoch_acc, epoch_loss = self._train_one_epoch(epoch_acc, epoch_loss)
            
            # Print the train and val information via each epoch
            for key in epoch_loss.keys():
                if key == 'Step 3: Minimize discrepancy':
                    logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/(4.*self.num_iter)))
                else:
                    logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/self.num_iter))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/self.num_iter))
            
            # Log the best model according to the val accuracy
            new_acc = self.test()
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
                    
            if self.lr_scheduler_G is not None:
                self.lr_scheduler_G.step()
            if self.lr_scheduler_C is not None:
                self.lr_scheduler_C.step()
