'''
Paper: Zhang, Y., Liu, T., Long, M. and Jordan, M., 2019, May. Bridging theory and 
    algorithm for domain adaptation. In International conference on machine learning (pp. 7404-7413). PMLR.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import utils
import model_base
from train_utils import InitTrain


def shift_log(x: torch.Tensor, offset: float = 1e-6) -> torch.Tensor:

    return torch.log(torch.clamp(x + offset, max=1.))


class MarginDisparityDiscrepancy(nn.Module):

    def __init__(self, source_disparity, target_disparity,
                 margin: float = 4, reduction: str = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: torch.Tensor = None, w_t: torch.Tensor = None) -> torch.Tensor:

        source_loss = -self.margin * self.source_disparity(y_s, y_s_adv)
        target_loss = self.target_disparity(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        loss = source_loss + target_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
   
    def __init__(self, margin: float = 4, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return F.cross_entropy(y_adv, prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')

        super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin,
                                                                       **kwargs)


class GeneralModule(nn.Module):
    
    def __init__(self, args, grl):
        super(GeneralModule, self).__init__()
        output_size = 2560
        self.G = model_base.FeatureExtractor(in_channel=1)
        self.C1 = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                        dropout=args.dropout, last=None)
        self.C2 = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                        dropout=args.dropout, last=None)
        self.grl_layer = utils.WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=False) if grl is None else grl

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """"""
        features = self.G(x)
        outputs = self.C1(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.C2(features_adv)
        if self.training:
            return outputs, outputs_adv
        else:
            return outputs

    def step(self):
        """
        Gradually increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        self.mdd = ClassificationMarginDisparityDiscrepancy().to(self.device)
        grl = utils.GradientReverseLayer()
        self.model = GeneralModule(args, grl=grl).to(self.device)
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
        
        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")

        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        
        best_acc = 0.0
        best_epoch = 0
   
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
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            num_iter = len(self.dataloaders['train'])                
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)   
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
            						     self.iters, src, self.device)
                # forward
                self.optimizer.zero_grad()
                data = torch.cat((source_data, target_data), dim=0)
                
                outputs, outputs_adv = self.model(data)
                y_s, y_t = outputs.chunk(2, dim=0)
                y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)
        
                # compute cross entropy loss on source domain
                cls_loss = F.cross_entropy(y_s, source_labels)
                # compute margin disparity discrepancy between domains
                # for adversarial classifier, minimize negative mdd is equal to maximize mdd
                transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
                loss = cls_loss + tradeoff[0] * transfer_loss
                self.model.step()
                
                epoch_acc['Source train']  += utils.get_accuracy(y_s, source_labels)
                epoch_loss['Source domain'] += cls_loss
                epoch_loss['MDD'] += transfer_loss
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
