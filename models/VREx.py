'''
Paper: Krueger, D., Caballero, E., Jacobsen, J.H., Zhang, A., Binas, J., Zhang, D., Le Priol, R. and Courville, A., 2021, July.
       Out-of-distribution generalization via risk extrapolation (rex). In International conference on machine learning (pp. 5815-5826). PMLR.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F

import modules
from train_utils import TrainerBase
        

class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)
        self.model = modules.ClassifierBase(input_size=1, num_classes=num_classes,
                                            backbone=args.backbone, dropout=args.dropout).to(self.device)
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
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            source_data, labels_per_domain = [], []
            for idx in range(self.num_source):
                source_data_item, source_labels_item = self._get_next_batch(self.src[idx], return_actual=True)
                source_labels_item = self._get_train_label(source_labels_item, label_set=self.src_labels_flat)
                source_data.append(source_data_item)
                labels_per_domain.append(source_labels_item)
            source_data = torch.cat(source_data, dim=0)
            source_labels = torch.cat(labels_per_domain, dim=0)
            
            # forward
            self.optimizer.zero_grad()
            pred_all, _ = self.model(source_data)
            pred_per_domain = pred_all.chunk(self.num_source, dim=0)

            # compute loss
            loss_ce_per_domain = torch.zeros(self.num_source).to(self.device)
            for idx in range(self.num_source):
                loss_ce_per_domain[idx] = F.cross_entropy(pred_per_domain[idx], labels_per_domain[idx])
            loss_ce = loss_ce_per_domain.mean()
            loss_penalty = ((loss_ce_per_domain - loss_ce) ** 2).mean()
            loss = loss_ce + self.tradeoff[0] * loss_penalty

            # log information
            epoch_acc['Source Data']  += self._get_accuracy(pred_all, source_labels)
            epoch_loss['Source Classifier'] += loss_ce
            epoch_loss['Risk Variance'] += loss_penalty

            # backward
            loss.backward()
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
