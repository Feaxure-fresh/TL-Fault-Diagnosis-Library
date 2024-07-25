'''
Paper: Arjovsky, M., Bottou, L., Gulrajani, I. and Lopez-Paz, D., 2019. Invariant risk minimization.
       arXiv preprint arXiv:1907.02893.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import modules
from train_utils import TrainerBase


class InvariancePenaltyLoss(nn.Module):

    def __init__(self):
        super(InvariancePenaltyLoss, self).__init__()
        self.scale = torch.tensor(1.).requires_grad_()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_1 = F.cross_entropy(y[::2] * self.scale, labels[::2])
        loss_2 = F.cross_entropy(y[1::2] * self.scale, labels[1::2])
        grad_1 = torch.autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = torch.autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        
        return penalty


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.model = modules.ClassifierBase(input_size=1, num_classes=args.num_classes[0], backbone=args.backbone,
                                            dropout=args.dropout, use_cls_feat=1).to(self.device)
        self.irm = InvariancePenaltyLoss()
        self._init_data()

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = len(self.dataloaders[self.src])
    
    def save_model(self):
        torch.save({
            'model': self.model.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])
    
    def _set_to_train(self):
        self.model.train()
    
    def _set_to_eval(self):
        self.model.eval()
        
    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            source_data, source_labels = self._get_next_batch(self.src)

            # forward
            self.optimizer.zero_grad()
            pred, _ = self.model(source_data)
            
            # compute loss
            loss_c = F.cross_entropy(pred, source_labels)
            loss_irm = self.irm(pred, source_labels)
            loss = loss_c + self.tradeoff[0] * loss_irm

            # log information
            epoch_acc['Source Data'] += self._get_accuracy(pred, source_labels)
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['IRM'] += loss_irm

            # backward
            loss.backward()
            self.optimizer.step()
        return epoch_acc, epoch_loss
                            
    def _eval(self, data, actual_labels, correct, total):
        pred = self.model(data)
        actual_pred = self._get_actual_label(pred, idx=0)
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]; total['acc'] += output[1]
        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(actual_pred, actual_labels, return_acc=False, idx=0, mode='closed-set')
            correct['Closed-set-acc'] += output[0]; total['Closed-set-acc'] += output[1]
        return correct, total
