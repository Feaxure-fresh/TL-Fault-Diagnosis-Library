'''
Paper: Sagawa, S., Koh, P.W., Hashimoto, T.B. and Liang, P., 2019. Distributionally robust neural networks for group shifts:
       On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F

import modules
from train_utils import TrainerBase


class AutomaticUpdateDomainWeightModule(object):
    def __init__(self, num_domains: int, eta: float, device):
        self.domain_weight = torch.ones(num_domains).to(device) / num_domains
        self.eta = eta

    def get_domain_weight(self, sampled_domain_idxes):
        """Get domain weight to calculate final objective.

        Inputs:
            - sampled_domain_idxes (list): sampled domain indexes in current mini-batch

        Shape:
            - sampled_domain_idxes: :math:`(D, )` where D means the number of sampled domains in current mini-batch
            - Outputs: :math:`(D, )`
        """
        domain_weight = self.domain_weight[sampled_domain_idxes]
        domain_weight = domain_weight / domain_weight.sum()
        return domain_weight

    def update(self, sampled_domain_losses: torch.Tensor, sampled_domain_idxes):
        """Update domain weight using loss of current mini-batch.

        Inputs:
            - sampled_domain_losses (tensor): loss of among sampled domains in current mini-batch
            - sampled_domain_idxes (list): sampled domain indexes in current mini-batch

        Shape:
            - sampled_domain_losses: :math:`(D, )` where D means the number of sampled domains in current mini-batch
            - sampled_domain_idxes: :math:`(D, )`
        """
        sampled_domain_losses = sampled_domain_losses.detach()

        for loss, idx in zip(sampled_domain_losses, sampled_domain_idxes):
            self.domain_weight[idx] *= (self.eta * loss).exp()


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.src_labels_flat = sorted(list(set([label for sublist in args.label_sets[:-1] for label in sublist])))
        num_classes = len(self.src_labels_flat)
        self.model = modules.ClassifierBase(input_size=1, num_classes=num_classes, backbone=args.backbone,
                                            dropout=args.dropout, use_cls_feat=1).to(self.device)
        self.domain_weight_module = AutomaticUpdateDomainWeightModule(num_domains=self.num_source,
                                                                      eta=1e-2, device=self.device)
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
        domain_idxes = list(range(self.num_source))
        for _ in tqdm(range(self.num_iter), ascii=True):
            # obtain data
            source_data, source_labels = [], []
            for idx in range(self.num_source):
                source_data_item, source_labels_item = self._get_next_batch(self.src[idx], return_actual=True)
                source_labels_item = self._get_train_label(source_labels_item, label_set=self.src_labels_flat)
                source_data.append(source_data_item)
                source_labels.append(source_labels_item)
            
            # forward
            cls_acc = 0
            self.optimizer.zero_grad()
            loss_per_domain = torch.zeros(self.num_source).to(self.device)
            for idx in range(self.num_source):
                y_per_domain, _ = self.model(source_data[idx])
                loss_per_domain[idx] = F.cross_entropy(y_per_domain, source_labels[idx])
                cls_acc += self._get_accuracy(y_per_domain, source_labels[idx]) / self.num_source

            # update domain weight
            self.domain_weight_module.update(loss_per_domain, domain_idxes)
            domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)

            # weighted cls loss
            loss = (loss_per_domain * domain_weight).sum()

            # log information
            epoch_acc['Source Data']  += cls_acc
            epoch_loss['Source Classifier'] += loss

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
