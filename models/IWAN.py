'''
Paper: Zhang, J., Ding, Z., Li, W. and Ogunbona, P., 2018. Importance weighted adversarial nets for partial domain adaptation.
       In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 8156-8164).
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F

import utils
import modules
from train_utils import TrainerBase       


class ImportanceWeightModule(object):

    def __init__(self, discriminator):
        self.discriminator = discriminator

    def get_importance_weight(self, feature):
        weight = 1. - self.discriminator(feature)
        weight = weight / (weight.mean() + 1e-5)
        weight = weight.detach()
        return weight
    

class Trainer(TrainerBase):
    
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.model = modules.ClassifierBase(input_size=1, num_classes=args.num_classes[0], backbone=args.backbone,
                                            dropout=args.dropout, use_batchnorm=True, use_cls_feat=1).to(self.device)
        self.domain_discri = modules.MLP(input_size=self.model.feature_dim, output_size=1,
                                         dropout=args.dropout, last='sigmoid').to(self.device)
        self.discri_0 = modules.MLP(input_size=self.model.feature_dim, output_size=1,
                                    dropout=args.dropout, last='sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
        self.adv_0 = utils.DomainAdversarialLoss(self.discri_0, grl=grl)
        self.importance_weight_module = ImportanceWeightModule(self.domain_discri)
        self._init_data()

        if args.train_mode == 'single_source':
            self.src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            self.src = 'concat_source'
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer = self._get_optimizer([self.model, self.domain_discri, self.discri_0])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.num_iter = len(self.dataloaders[self.src])
    
    def save_model(self):
        torch.save({
            'model': self.model.state_dict(),
            'domain_discri': self.domain_discri.state_dict(),
            'discri_0': self.discri_0.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])
        self.discri_0.load_state_dict(ckpt['discri_0'])
        self.domain_discri.load_state_dict(ckpt['domain_discri'])
    
    def _set_to_train(self):
        self.model.train()
        self.domain_discri.train()
        self.discri_0.train()
    
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
            y, f = self.model(data)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)
            w_s = self.importance_weight_module.get_importance_weight(f_s)

            # compute loss
            loss_adv, acc_adv = self.domain_adv(f_s.detach(), f_t.detach())
            loss_D0, acc_D0 = self.adv_0(f_s, f_t, w_s=w_s)
            loss_ent = utils.entropy(F.softmax(y_t, dim=1), reduction='mean')
            loss_c = F.cross_entropy(y_s, source_labels)
            loss = loss_c + self.tradeoff[0] * loss_adv + self.tradeoff[1] * loss_D0 + self.tradeoff[2] * loss_ent

            # log information
            epoch_acc['Source Data']  += self._get_accuracy(y_s, source_labels)
            epoch_acc['Domain Discriminator']  += acc_adv
            epoch_acc['Discriminator 0']  += acc_D0
            epoch_loss['Source Classifier'] += loss_c
            epoch_loss['Domain Discriminator'] += loss_adv
            epoch_loss['Discriminator 0']  += loss_D0
            epoch_loss['Entropy']  += loss_ent

            # backward
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
