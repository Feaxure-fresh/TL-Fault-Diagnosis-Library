import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain


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


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        self.model = model_base.BaseModel(input_size=1, num_classes=args.num_classes,
                                       dropout=args.dropout).to(self.device)
        self.domain_weight_module = AutomaticUpdateDomainWeightModule(num_domains=self.num_source,
                                       eta=1e-2, device=self.device)
        self._init_data()
    
    def save_model(self):
        torch.save({
            'model': self.model.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])
        
    def train(self):
        args = self.args
        src = args.source_name
       
        if args.train_mode != 'multi_source':
            raise Exception("For this model, invalid train mode: {}".format(args.train_mode))
        
        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        
        best_acc = 0.0
        best_epoch = 0

        domain_idxes = list(range(self.num_source))
        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
   
            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)
   
            # Set model to train mode or evaluate mode
            self.model.train()
            epoch_loss = defaultdict(float)
        
            num_iter = len(self.dataloaders['train'])          
            for i in tqdm(range(num_iter), ascii=True):
                source_data, source_labels = [], []
                for idx in range(self.num_source):
                    source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
                                                self.iters, src[idx], self.device)
                    source_data.append(source_data_item)
                    source_labels.append(source_labels_item)
                
                # forward
                cls_acc = 0
                self.optimizer.zero_grad()
                loss_per_domain = torch.zeros(self.num_source).to(self.device)
                for idx in range(self.num_source):
                    y_per_domain, _ = self.model(source_data[idx])
                    loss_per_domain[idx] = F.cross_entropy(y_per_domain, source_labels[idx])
                    cls_acc += utils.get_accuracy(y_per_domain, source_labels[idx]) / self.num_source

                # update domain weight
                self.domain_weight_module.update(loss_per_domain, domain_idxes)
                domain_weight = self.domain_weight_module.get_domain_weight(domain_idxes)

                # weighted cls loss
                loss = (loss_per_domain * domain_weight).sum()
                epoch_acc['Source Data']  += cls_acc
                
                epoch_loss['Source Classifier'] += loss

                # backward
                loss.backward()
                self.optimizer.step()
                
            # Print the train and val information via each epoch
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
            
            # log the best model according to the val accuracy
            new_acc = self.test()
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
    def test(self):
        self.model.eval()
        acc = 0.0
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred = self.model(target_data)
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc
