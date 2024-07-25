'''
Paper: Zhang, Y., Ren, Z., Zhou, S. and Yu, T., 2020. Adversarial domain adaptation with
       classifier alignment for cross-domain intelligent fault diagnosis of multiple source domains.
       Measurement Science and Technology, 32(3), p.035102.
Note: The code is reproduced according to the paper. Please point out any possible errors.
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


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)
        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=1).to(self.device)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2]).to(self.device)
        else:
            raise Exception(f"unknown backbone type {args.backbone}")
        self.Cs = nn.ModuleList([modules.MLP(input_size=self.G.out_dim, output_size=num_classes,
                                             dropout=args.dropout, last=None) \
                                             for _ in range(self.num_source)]).to(self.device)
        self.discriminator = modules.MLP(input_size=self.G.out_dim, output_size=self.num_source+1,
                                         dropout=args.dropout, last=None).to(self.device)
        self.grl = utils.GradientReverseLayer()
        self._init_data()

        if args.train_mode == 'source_combine':
            self.src = ['concat_source']
        else: self.src = args.source_name
        
        self.optimizer = self._get_optimizer([self.G, self.Cs, self.discriminator])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = sum([len(self.dataloaders[s]) for s in self.src])
    
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

    def _set_to_train(self):
        self.G.train()
        self.Cs.train()
        self.discriminator.train()
    
    def _set_to_eval(self):
        self.G.eval()
        self.Cs.eval()

    def _train_one_epoch(self, epoch_acc, epoch_loss):
        for i in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            cur_src_idx = int(i % self.num_source)
            target_data, _ = self._get_next_batch('train')
            source_data, source_labels = self._get_next_batch(self.src[cur_src_idx], return_actual=True)
            source_labels = self._get_train_label(source_labels, label_set=self.src_labels_flat)

            # forward
            self.optimizer.zero_grad()
            batch_size = source_data.shape[0]
            data = torch.cat((source_data, target_data), dim=0)
            f = self.G(data)
            f_s, f_t = f.chunk(2, dim=0)
            y_s = self.Cs[cur_src_idx](f_s)
            y_t = [cl(f_t) for cl in self.Cs]
            labels_dm = torch.concat((torch.full((batch_size,), cur_src_idx+1, dtype=torch.long),
                                      torch.zeros(batch_size, dtype=torch.long)), dim=0).to(self.device)
            feat = self.grl(torch.concat((f_s, f_t), dim=0))
            
            # compute loss
            loss_c = F.cross_entropy(y_s, source_labels)
            logits_dm = self.discriminator(feat)
            loss_d = F.cross_entropy(logits_dm, labels_dm)
            logits_tgt = [F.softmax(t, dim=1) for t in y_t]
            loss_l1 = 0.0
            for k in range(self.num_source - 1):
                for j in range(k+1, self.num_source):
                    # We use the mean of the differences, even though the sum is used in the paper.
                    loss_l1 += torch.abs(logits_tgt[k] - logits_tgt[j]).mean()
            loss_l1 /= self.num_source
            loss = loss_c + self.tradeoff[0] * loss_d + self.tradeoff[1] * loss_l1
            
            # log information
            epoch_acc['Source Data']  += self._get_accuracy(y_s, source_labels)
            epoch_acc['Discriminator']  += self._get_accuracy(logits_dm, labels_dm)
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['Discriminator'] += loss_d
            epoch_loss['L1'] += loss_l1

            # backward
            loss.backward()
            self.optimizer.step()
        return epoch_acc, epoch_loss
    
    def _eval(self, data, actual_labels, correct, total):
        feat_tgt = self.G(data)
        logits_tgt = [F.softmax(cl(feat_tgt), dim=1) for cl in self.Cs]
        pred = torch.sum(torch.stack(logits_tgt), dim=0).argmax(dim=1)
        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]; total['acc'] += output[1]
        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(actual_pred, actual_labels, return_acc=False, idx=0, mode='closed-set')
            correct['Closed-set-acc'] += output[0]; total['Closed-set-acc'] += output[1]
        return correct, total
              
