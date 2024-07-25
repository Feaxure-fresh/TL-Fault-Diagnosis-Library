'''
Paper: Tian, J., Han, D., Li, M. and Shi, P., 2022. A multi-source information transfer learning method
       with subdomain adaptation for cross-domain fault diagnosis. Knowledge-Based Systems, 243, p.108466.
Note: The code is reproduced according to the paper. Only the structure of MSSA is utilized (without other components proposed in the paper).  
Author: Feaxure
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=1).to(self.device)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
        else:
            raise Exception(f"unknown backbone type {args.backbone}")
        self.Fs = nn.ModuleList([modules.MLP(input_size=self.G.out_dim, dropout=args.dropout, num_layer=2, output_layer=False)
                                 for _ in range(self.num_source)]).to(self.device)
        self.Cs = nn.ModuleList([modules.MLP(input_size=self.Fs[i].feature_dim, output_size=args.num_classes[i],
                                             num_layer=1, last=None) for i in range(self.num_source)]).to(self.device)
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
                     kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        self._init_data()

        if args.train_mode == 'source_combine':
            self.src = ['concat_source']
        else: self.src = args.source_name

        self.optimizer = self._get_optimizer([self.G, self.Fs, self.Cs])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = sum([len(self.dataloaders[s]) for s in self.src])
    
    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.Fs.state_dict(),
            'Cs': self.Cs.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.Fs.load_state_dict(ckpt['Fs'])
        self.Cs.load_state_dict(ckpt['Cs'])
    
    def _set_to_train(self):
        self.G.train()
        self.Fs.train()
        self.Cs.train()
    
    def _set_to_eval(self):
        self.G.eval()
        self.Fs.eval()
        self.Cs.eval()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for i in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            cur_src_idx = int(i % self.num_source)
            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src[cur_src_idx])

            # forward
            self.optimizer.zero_grad()
            data = torch.cat((source_data, target_data), dim=0)
            f = self.Fs[cur_src_idx](self.G(data))
            f_s, f_t = f.chunk(2, dim=0)
            y = self.Cs[cur_src_idx](f)
            y_s, _ = y.chunk(2, dim=0)
            
             # compute loss
            loss_cls = F.cross_entropy(y_s, source_labels)
            loss_mmd = self.mkmmd(f_s, f_t)
            loss = loss_cls + self.tradeoff[0] * loss_mmd
            
            # log information
            epoch_acc['Source Data'] += self._get_accuracy(y_s, source_labels)
            epoch_loss['Source Classifier'] += loss_cls
            epoch_loss['MMD'] += loss_mmd
            
            # backward
            loss.backward()
            self.optimizer.step()     
        return epoch_acc, epoch_loss

    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)
        logits_tgt = [F.softmax(self.Cs[i](self.Fs[i](feat_tgt)), dim=1) for i in range(self.num_source)]
        actual_pred = self._combine_prediction(logits_tgt, idx=list(range(self.num_source)))
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]; total['acc'] += output[1]
        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(actual_pred, actual_labels, return_acc=False, idx=list(range(self.num_source)), mode='closed-set')
            correct['Closed-set-acc'] += output[0]; total['Closed-set-acc'] += output[1]
        return correct, total
