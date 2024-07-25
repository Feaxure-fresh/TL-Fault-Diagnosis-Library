'''
Paper: Zhang, Y., Liu, T., Long, M. and Jordan, M., 2019, May. Bridging theory and algorithm for
       domain adaptation. In International conference on machine learning (pp. 7404-7413). PMLR.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase


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

        super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin, **kwargs)


class GeneralModule(nn.Module):
    
    def __init__(self, args, grl):
        super(GeneralModule, self).__init__()
        if args.backbone == 'CNN':
            self.G = modules.MSCNN(in_channel=1)
        elif args.backbone == 'ResNet':
            self.G = modules.ResNet(in_channel=1, layers=[2, 2, 2, 2])
        else:
            raise Exception(f"unknown backbone type {args.backbone}")
        self.C1 = modules.MLP(input_size=self.G.out_dim, output_size=args.num_classes[0],
                              dropout=args.dropout, last=None)
        self.C2 = modules.MLP(input_size=self.G.out_dim, output_size=args.num_classes[0],
                              dropout=args.dropout, last=None)
        self.grl_layer = utils.WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                             auto_step=False) if grl is None else grl

    def forward(self, x: torch.Tensor):
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


class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.grl = None
        self.model = GeneralModule(args, grl=self.grl).to(self.device)
        self.mdd = ClassificationMarginDisparityDiscrepancy().to(self.device)
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
            target_data, _ = self._get_next_batch('train')  
            source_data, source_labels = self._get_next_batch(self.src)

            # forward
            self.optimizer.zero_grad()
            data = torch.cat((source_data, target_data), dim=0)
            outputs, outputs_adv = self.model(data)
            y_s, y_t = outputs.chunk(2, dim=0)
            y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

            # compute loss
            cls_loss = F.cross_entropy(y_s, source_labels)
            # compute margin disparity discrepancy between domains for adversarial
            # classifier, minimize negative mdd is equal to maximize mdd
            transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
            loss = cls_loss + self.tradeoff[0] * transfer_loss

            # log information
            epoch_acc['Source train']  += self._get_accuracy(y_s, source_labels)
            epoch_loss['Source domain'] += cls_loss
            epoch_loss['MDD'] += transfer_loss

            # backward
            if self.grl is None:
                self.model.step()
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
