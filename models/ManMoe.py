import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
from train_utils import InitTrain
import model_base


def get_gate_label(gate_out, idx, device):
    labels = torch.full(gate_out.size()[:-1], idx, dtype=torch.long)
    labels = labels.to(device)
    return labels


def evaluate_acc(dataloaders, F_s, F_p, C, device):
    F_s.eval()
    F_p.eval()
    C.eval()
    iters = iter(dataloaders['val'])
    # labels, lab_tot = defaultdict(int), defaultdict(int)
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, targets, idx in tqdm(iters, ascii=True):
            inputs = inputs.to(device)
            targets = targets.to(device)

            shared_features = F_s(inputs)
            private_feat, _ = F_p(inputs)
            outputs, _ = C((shared_features, private_feat))

            _, pred = torch.max(outputs, -1)
            equ = (pred == targets)
            # for i, lab in enumerate(targets):
            #     lab_tot['%d'%lab] += 1
            #     if equ[i]:
            #         labels['%d'%lab] += 1       
            correct += equ.sum().item()
            total += pred.shape[0]
        acc = correct / total
        logging.info('Accuracy on {} samples: {:.3f}'.format(total, 100.0*acc))
        # logging.info('Correct labels:')
        # logging.info('\t'.join(labels.keys()))
        # logging.info('\t'.join(['%.03f' % (100.0*labels[d]/lab_tot[d])
        #                                                 for d in labels.keys()]))
    return acc


class MixtureOfExperts(nn.Module):
    
   def __init__(self,
                input_size,
                output_size,
                num_source,
                dropout):
       super(MixtureOfExperts, self).__init__()
       
       self.gate = model_base.ClassifierMLP(input_size=input_size, output_size=num_source,
                                            dropout=dropout, last='sm')
       
       self.experts = nn.ModuleList([nn.Sequential(                    
                                     nn.Linear(input_size, output_size),
                                     nn.ReLU()) for _ in range(num_source)])
              
   def forward(self, input):
        gate_input = input.detach()
        gate_outs = self.gate(gate_input)
        
        expert_outs = torch.stack([exp(input) for exp in self.experts], dim=-2)
        output = torch.sum(gate_outs.unsqueeze(-1) * expert_outs, dim=-2)
        
        return output, gate_outs


class ClassifierMoE(nn.Module):
    
   def __init__(self,
                input_size,
                output_size,
                num_source,
                dropout):
       super(ClassifierMoE, self).__init__()
       
       self.gate = model_base.ClassifierMLP(input_size=input_size,
                       output_size=num_source, dropout=dropout, last='sm')
       
       self.experts = nn.ModuleList([model_base.ClassifierMLP(
                input_size=input_size,
                output_size=output_size, dropout=dropout, last='logsm') for _ in range(num_source)])
              
       
   def forward(self, input):
       fs, fp = input
       features = torch.cat([fs, fp], dim=-1)
       gate_input = features.detach()
       gate_outs = self.gate(gate_input)

       expert_outs = torch.stack([exp(features) for exp in self.experts], dim=-2)
       output = torch.sum(gate_outs.unsqueeze(-1) * expert_outs, dim=-2)
        
       return output, gate_outs


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
    
    def train(self):
        args = self.args
        self._init_data(concat_src=True)
        if args.train_mode == 'source_combine':
            args.source_name = [args.source_name]
        args.all_src = args.source_name + ['train']
        
        F_s = model_base.FeatureExtractor(input_size=1, output_size=1024, dropout=args.dropout)
        
        F_p = nn.Sequential(
            model_base.FeatureExtractor(input_size=1, output_size=1024, dropout=args.dropout),
            MixtureOfExperts(input_size=1024, num_source=self.num_source,
                                     output_size=1024, dropout=args.dropout))
        
        C = ClassifierMoE(input_size=2048, num_source=self.num_source,
                    output_size=args.num_classes, dropout=args.dropout)
        
        D = model_base.ClassifierMLP(input_size=1024, output_size=int(self.num_source+1),
                        dropout=args.dropout, last='logsm')
        
        F_s, F_p, C, D = F_s.to(self.device), F_p.to(self.device), \
                         C.to(self.device), D.to(self.device)
                         
        # optimizers
        optimizer = self._get_optimizer([F_s, F_p, C])
        optimizerD = self._get_optimizer(D)
        self.lr_scheduler = self._get_lr_scheduler(optimizer)
        self.lr_scheduler_D = self._get_lr_scheduler(optimizerD)
        
        # training
        best_acc = 0.0
        best_epoch = 0
        
        num_iter = len(self.iters['train'])            
        for epoch in range(1, args.max_epoch+1):
            F_s.train()
            F_p.train()
            C.train()
            D.train()
            
            # training accuracy
            acc_train, loss_train = defaultdict(float), defaultdict(float)
            loss_D, D_num = defaultdict(float), defaultdict(int)
            d_correct = defaultdict(int)
            
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            if self.lr_scheduler_D is not None:
                logging.info('current Discriminator lr: {}'.format(self.lr_scheduler_D.get_last_lr()))
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)
            for i in tqdm(range(num_iter), ascii=True):
            
                # D iterations
                utils.freeze_net(F_s)
                utils.freeze_net(F_p)
                utils.freeze_net(C)
                utils.unfreeze_net(D)
                
                D.zero_grad()
                # train on both labeled and unlabeled source
                for src in args.all_src:
                    # targets not used
                    d_inputs, _ = utils.get_next_batch(self.dataloaders,
                                                       self.iters, src, self.device)
                    idx = args.all_src.index(src)
                    
                    shared_feat = F_s(d_inputs)
                    d_outputs = D(shared_feat)
                    
                    # if token-level D, we can reuse the gate label generator
                    d_targets = get_gate_label(d_outputs, idx, self.device)
                    
                    # D accuracy
                    _, pred = torch.max(d_outputs, -1)
                    d_correct['%s'%src] += (pred == d_targets).sum().item()/pred.shape[0]
                    loss_d = F.nll_loss(d_outputs, d_targets)
                        
                    loss_D['%s'%src] += loss_d
                    D_num['%s'%src] += 1
                    loss_d.backward()
                optimizerD.step()
    
                # F&C iteration
                utils.unfreeze_net(F_s)
                utils.unfreeze_net(F_p)
                utils.unfreeze_net(C)
                utils.freeze_net(D)
                
                F_s.zero_grad()
                F_p.zero_grad()
                C.zero_grad()
                inputs, targets, src_idx = utils.get_concat_dataset_next_batch(self.dataloaders,
                                                       self.iters, 'concat_source', self.device, return_idx=True)
                shared_feat = F_s(inputs)
                private_feat, gate_outputs = F_p(inputs)
                c_outputs, c_gate_outputs = C((shared_feat, private_feat))
                
                # token-level gate loss
                loss_gate = F.nll_loss(torch.log(gate_outputs), src_idx)
                loss_train['Specific Fs Moe'] += loss_gate
                
                _, gate_pred = torch.max(gate_outputs, -1)
                acc_train['Specific Fs Moe'] += (gate_pred == src_idx).sum().item()/gate_pred.shape[0]
                
                loss_c_gate = F.nll_loss(torch.log(c_gate_outputs), src_idx)
                loss_train['Classifier Moe'] += loss_c_gate

                _, c_gate_pred = torch.max(c_gate_outputs, -1)
                acc_train['Classifier Moe'] += (c_gate_pred == src_idx).sum().item()/c_gate_pred.shape[0]

                loss_clf = F.nll_loss(c_outputs, targets)
                _, pred = torch.max(c_outputs, -1)
                acc_train['Source Domain'] += (pred == targets).sum().item()/pred.shape[0]
                loss_train['Source Domain'] += loss_clf
                loss = loss_clf + tradeoff[0] * loss_gate + tradeoff[1] * loss_c_gate
                loss.backward()
    
                # update F with D gradients on all source
                for src in args.all_src:
                    inputs, _ = utils.get_next_batch(self.dataloaders,
                                                     self.iters, src, self.device)
                    idx = args.all_src.index(src)
                    shared_feat = F_s(inputs)
                    d_outputs = D(shared_feat)
                    
                    # if token-level D, we can reuse the gate label generator
                    d_targets = get_gate_label(d_outputs, idx, self.device)
                    loss_d = F.nll_loss(d_outputs, d_targets)
                    loss_d *= -tradeoff[2]
                    loss_d.backward()
                optimizer.step()
                
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.lr_scheduler_D is not None:
                self.lr_scheduler_D.step()
                
            # end of epoch
            logging.info('Ending epoch {}'.format(epoch))
            logging.info('D Training Accuracy:')
            logging.info('\t'.join(d_correct.keys()))
            logging.info('\t'.join(['%.03f' % (100.0*d_correct[d]/D_num[d])
                                                            for d in d_correct.keys()]))
            logging.info('D Loss:')
            logging.info('\t'.join(loss_D.keys()))
            logging.info('\t'.join(['%.04f' % (loss_D[d]/D_num[d])
                                                            for d in loss_D.keys()]))
            logging.info('Training loss:')
            for key in loss_train.keys():
                logging.info('{}-Loss: {:.4f}'.format(key, loss_train[key]/num_iter))
            logging.info('Training acc:')
            for key in acc_train.keys():
                logging.info('{}-Acc: {:.4f}'.format(key, acc_train[key]/num_iter))
            logging.info('Evaluating validation sets:')
            acc = evaluate_acc(self.dataloaders, F_s, F_p, C, self.device)
    
            if acc >= best_acc:
                best_acc = acc
                best_epoch = epoch
            logging.info('Best epoch {} accuracy: {:.3f}'.format(best_epoch, 100.0*best_acc))

