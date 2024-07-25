'''
Paper: Li, D., Yang, Y., Song, Y.Z. and Hospedales, T., 2018, April. Learning to generalize: Meta-learning for domain generalization.
       In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import higher
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)
        self.model = modules.ClassifierBase(input_size=1, num_classes=num_classes, backbone=args.backbone,
                                            dropout=args.dropout, use_cls_feat=1).to(self.device)
        self.mkmmd = utils.MultipleKernelMaximumMeanDiscrepancy(
                            kernels=[utils.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        self._init_data()

        assert args.train_mode == 'multi_source', "this model can only be trained in multi_source mode"
	
        self.src = args.source_name
        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = int(sum([len(self.dataloaders[s]) for s in self.src]) / self.num_source)
    
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
        idx_range = torch.arange(0, self.num_source)
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            source_data, source_labels = [], []
            for idx in range(self.num_source):
                source_data_item, source_labels_item = self._get_next_batch(self.src[idx], return_actual=True)
                source_labels_item = self._get_train_label(source_labels_item, label_set=self.src_labels_flat)
                source_data.append(source_data_item)
                source_labels.append(source_labels_item)
            train_idx = np.random.choice(idx_range, size=(self.num_source - 1,), replace=False)
            test_idx = np.setdiff1d(idx_range, train_idx)
            
            # forward
            self.optimizer.zero_grad()
            with higher.innerloop_ctx(self.model, self.optimizer, copy_initial_weights=False) as (inner_model, inner_optimizer):
                for _ in range(1):  # Single gradient update (for simplicity)
                    loss_inner = 0
                    for idx in train_idx:
                        y, _ = inner_model(source_data[idx])
                        loss_inner += F.cross_entropy(y, source_labels[idx]) / len(train_idx)
                    inner_optimizer.step(loss_inner)

                loss_outer = 0
                cls_acc = 0

                for idx in train_idx:
                    y, _ = self.model(source_data[idx])
                    loss_outer += F.cross_entropy(y, source_labels[idx]) / len(train_idx)

                for idx in test_idx:
                    y, _ = inner_model(source_data[idx])
                    loss_outer += F.cross_entropy(y, source_labels[idx]) * self.tradeoff[0] / len(test_idx)
                    cls_acc += self._get_accuracy(y, source_labels[idx]) / len(test_idx)
            
            # log information
            epoch_acc['Source Data']  += cls_acc
            epoch_loss['Meta-train'] += loss_inner
            epoch_loss['Meta_test'] += loss_outer

            # backward
            loss_outer.backward()
            self.optimizer.step()
        return epoch_acc, epoch_loss
            
    def _eval(self, data, actual_labels, correct, total):
        pred = self.model(data)
        actual_pred = self._get_actual_label(pred, label_set=self.src_labels_flat)
        output = self._get_accuracy(actual_pred, actual_labels, return_acc=False)
        correct['acc'] += output[0]; total['acc'] += output[1]
        if self.args.da_scenario in ['open-set', 'universal']:
            output = self._get_accuracy(actual_pred, actual_labels, return_acc=False, idx=0, mode='closed-set')
            correct['Closed-set-acc'] += output[0]; total['Closed-set-acc'] += output[1]
        return correct, total