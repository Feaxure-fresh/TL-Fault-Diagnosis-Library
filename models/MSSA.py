'''
Paper: Tian, J., Han, D., Li, M. and Shi, P., 2022. A multi-source information transfer learning method
       with subdomain adaptation for cross-domain fault diagnosis. Knowledge-Based Systems, 243, p.108466.
Note: The code is reproduced according to the paper (only the structure of MSSA and LMMD are utilized).  
Author: Feaxure
'''
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


class LMMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.shape[0]) + int(target.shape[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
        total1 = total.unsqueeze(1).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label, class_num):
        batch_size = source.shape[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=class_num)
        weight_ss = torch.tensor(weight_ss, dtype=torch.float32, device=source.device)
        weight_tt = torch.tensor(weight_tt, dtype=torch.float32, device=source.device)
        weight_st = torch.tensor(weight_st, dtype=torch.float32, device=source.device)

        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        loss = torch.tensor(0.0, dtype=torch.float32, device=source.device)

        if torch.sum(torch.isnan(kernels)):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        
        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.shape[0]
        s_sca_label = s_label.cpu().numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100 # any number except 0
        s_vec_label = s_vec_label / s_sum

        t_sca_label = torch.argmax(t_label, dim=1).cpu().numpy()
        t_vec_label = t_label.cpu().numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


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
        self.lmmd = LMMD_loss()
        self.num_class = args.num_classes
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
            y_s, y_t = y.chunk(2, dim=0)
            
             # compute loss
            loss_cls = F.cross_entropy(y_s, source_labels)
            loss_mmd = self.lmmd.get_loss(f_s, f_t, source_labels, F.softmax(y_t.detach(), dim=1),
                                          self.num_class[cur_src_idx])
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
