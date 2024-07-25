import math
import torch
import logging
import importlib
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data.dataset import ConcatDataset

import utils


class TrainerBase(object):
    
    def __init__(self, args):
        self.args = args
        if args.cuda_device:
            self.device = torch.device("cuda:" + args.cuda_device)
            logging.info('using {} / {} gpus'.format(len(args.cuda_device.split(',')), torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            logging.info('using cpu')
        if args.train_mode == 'source_combine':
            self.num_source = 1
        else:
            self.num_source = len(args.source_name)

    
    def _get_lr_scheduler(self, optimizer):
        '''
        Get learning rate scheduler for optimizer.
        '''
        args = self.args
        assert args.lr_scheduler in ['step', 'exp', 'stepLR', 'fix'], f"lr scheduler should be 'step', 'exp', 'stepLR' or 'fix', but got {args.lr_scheduler}"
        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            lr_scheduler = None
        return lr_scheduler
    
    
    def _get_optimizer(self, model):
        '''
        Get optimizer for model.
        '''
        args = self.args
        if type(model) == list:
            par =  [{'params': md.parameters()} for md in model]
        else:
            par = model.parameters()
        
        # Define the optimizer
        assert args.opt in ['sgd', 'adam'], f"optimizer should be 'sgd' or 'adam', but got {args.opt}"
        if args.opt == 'sgd':
            optimizer = optim.SGD(par, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            optimizer = optim.Adam(par, lr=args.lr, betas=args.betas,
                                   weight_decay=args.weight_decay)
        return optimizer
    
    
    def _get_tradeoff(self, tradeoff_list, epoch=None):
        '''
        Get trade-off parameters for loss.
        '''
        tradeoff = []
        for item in tradeoff_list:
            if item == 'exp':
                tradeoff.append(2 / (1 + math.exp(-self.args.zeta * (epoch-1) / (self.args.max_epoch-1))) - 1)
            elif type(item) == float or type(item) == int:
                tradeoff.append(item)
            else:
                raise Exception(f"unknown trade-off type {item}")
        return tradeoff
    
    
    def _get_actual_label(self, labels, idx=None, label_set=None):
        if idx is not None:
            label_set = self.args.label_sets[idx]
        else: assert label_set is not None
        actual_labels = []
        if len(labels.size()) > 1:
            labels = labels.argmax(dim=1)
        for label in labels.cpu():
            actual_labels.append(label_set[label])
        return torch.tensor(actual_labels).to(labels.device)
    
        
    def _get_train_label(self, labels, idx=None, label_set=None):
        if idx is not None:
            label_set = self.args.label_sets[idx]
        else: assert label_set is not None
        train_labels = []
        for label in labels.cpu():
            train_labels.append(label_set.index(label))
        return torch.tensor(train_labels).to(labels.device)
    

    def _combine_prediction(self, preds, idx, weights=None):
        # Ensure weights is not None and has the same length as labels and idx
        if weights is None:
            weights = [1.0] * len(preds)
        assert len(preds) == len(idx) == len(weights), "labels, idx, and weights must have the same length"
        
        batch_size = preds[0].size(0)
        # Dictionary to accumulate weighted predictions for each class
        class_predictions = defaultdict(lambda: torch.zeros(batch_size, dtype=torch.float, device=preds[0].device))
        class_weights = defaultdict(lambda: torch.zeros(1, dtype=torch.float, device=preds[0].device))

        # Process each source domain's predictions
        for label, domain_idx, weight in zip(preds, idx, weights):
            label_set = self.args.label_sets[domain_idx]
            
            # Calculate the weighted sum of predictions for each class
            for class_index in range(label.size(1)):
                actual_class_label = label_set[class_index]
                class_predictions[actual_class_label] += label[:, class_index] * weight
                class_weights[actual_class_label] += weight

        # Convert class_predictions to a tensor of shape (batch_size, num_classes)
        unique_classes = sorted(class_predictions.keys())
        predictions_tensor = torch.stack([class_predictions[cls]/class_weights[cls] if class_weights[cls] > 0 else class_predictions[cls] \
                                          for cls in unique_classes], dim=1)

        # Take the argmax to get the most confident predictions
        predicted_indices = predictions_tensor.argmax(dim=1)
        
        # Map indices back to actual labels
        predicted_labels = [unique_classes[idx.item()] for idx in predicted_indices]
        predicted_labels_tensor = torch.tensor(predicted_labels, device=preds[0].device)
        return predicted_labels_tensor
    
    
    def _get_accuracy(self, preds, targets, return_acc=True, idx=None, mode='normal'):
        assert preds.shape[0] == targets.shape[0]
        if len(preds.size()) > 1:
            preds = preds.argmax(dim=1)
        total = preds.shape[0]
        if mode != "normal":
            if isinstance(idx, list):
                # Combine all self.args.label_sets with the indices in idx
                label_set = torch.cat([torch.tensor(self.args.label_sets[i]) for i in idx])
            else:
                label_set = torch.tensor(self.args.label_sets[idx])
            targets = torch.where(torch.isin(targets, label_set), targets, -1)
        if mode == "closed-set":
            unknown_num = torch.sum(targets == -1).item()
            total -= unknown_num
        correct = torch.eq(preds.cpu(), targets.cpu()).float().sum().item()
        if return_acc:
            accuracy = correct/total if total > 0 else 0
            return accuracy
        else:
            return correct, total
    
    
    def _get_next_batch(self, src, return_actual=False):
        try:
            inputs, actual_labels = next(self.iters[src])
        except StopIteration:
            self.iters[src] = iter(self.dataloaders[src])
            inputs, actual_labels = next(self.iters[src])
        
        if return_actual:
            output = [inputs, actual_labels]
        else:
            src_idx = self.dataset_keys.index(src)
            if src in ['train', 'val']:
                src_idx = -1
            output = [inputs, self._get_train_label(actual_labels, src_idx)]
        output = [item.to(self.device) for item in output]
        return output
    
    
    def _init_data(self):
        '''
        Initialize the datasets.
        '''
        args = self.args
        
        self.datasets = {}
        for i, source in enumerate(args.source_name):
            dataset, condition, _ = utils.get_info_from_name(source)
            if condition is not None:
                Dataset = importlib.import_module("data_loader.conditional_load").dataset
                self.datasets[source] = Dataset(args, dataset, i+1 if args.train_mode == 'source_combine' else i,
                                                condition=condition).data_preprare(is_src=True)
            else:
                Dataset = importlib.import_module("data_loader.load").dataset
                self.datasets[source] = Dataset(args, dataset, i+1 if args.train_mode == 'source_combine' else i).data_preprare(is_src=True) 
        for key in self.datasets.keys():
            logging.info('Source set {} number of samples: {}.'.format(key, len(self.datasets[key])))
            self.datasets[key].summary()
        
        dataset, condition, _ = utils.get_info_from_name(args.target)
        if condition is not None:
            Dataset = importlib.import_module("data_loader.conditional_load").dataset
            self.datasets['train'], self.datasets['val'] = Dataset(args, dataset, -1, condition=condition).data_preprare(is_src=False)
        else:
            Dataset = importlib.import_module("data_loader.load").dataset
            self.datasets['train'], self.datasets['val'] = Dataset(args, dataset, -1).data_preprare(is_src=False)           
        logging.info('Training set number of samples: {}.'.format(len(self.datasets['train'])))
        self.datasets['train'].summary()
        logging.info('Validation set number of samples: {}.'.format(len(self.datasets['val'])))
        self.datasets['val'].summary()
        
        if args.train_mode == 'source_combine':
            self.datasets['concat_source'] = ConcatDataset([self.datasets[s] for s in args.source_name])
            self.dataset_keys = ['concat_source', 'train', 'val']
        else:
            self.dataset_keys = args.source_name + ['train', 'val']

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                              batch_size=args.batch_size,
                                              shuffle=(False if x == 'val' else True),
                                              num_workers=args.num_workers,
                                              drop_last=(False if x == 'val' else True),
                                              pin_memory=(True if self.device == 'cuda' else False))
                                              for x in self.dataset_keys}
        self.iters = {x: iter(self.dataloaders[x]) for x in self.dataset_keys}

    def _train_one_epoch(self):
        raise NotImplementedError("Subclasses should implement '_train_one_epoch' method")
    
    def _set_to_train(self):
        raise NotImplementedError("Subclasses should implement '_set_to_train' method")
        
    def _eval(self, data, actual_labels, correct, total):
        raise NotImplementedError("Subclasses should implement '_eval' method")
        
    def _set_to_eval(self):
        raise NotImplementedError("Subclasses should implement '_set_to_eval' method")
    
    def _log_epoch_info(self, epoch_loss, epoch_acc, num_iter):
        # Print the train and val information via each epoch
        for key in epoch_loss.keys():
            logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
        for key in epoch_acc.keys():
            logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))

    def train(self):
        args = self.args
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
            self._set_to_train()
            epoch_loss = defaultdict(float)
            self.tradeoff = self._get_tradeoff(args.tradeoff, epoch)
            
            epoch_acc, epoch_loss = self._train_one_epoch(epoch_acc, epoch_loss)
            
            # Log epoch information
            self._log_epoch_info(epoch_loss, epoch_acc, self.num_iter)
            
            # Log the best model according to the val accuracy
            new_acc = self.test()
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
    def test(self):
        self._set_to_eval()
        correct = defaultdict(int)
        total = defaultdict(int)
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for _ in tqdm(range(num_iter), ascii=True):
                target_data, actual_labels = next(iters)
                target_data = target_data.to(self.device)
                correct, total = self._eval(target_data, actual_labels, correct, total)
        for key in correct.keys():
            logging.info('Val-{}: {:.4f}'.format(key, correct[key]/total[key]))
        return correct['acc']/total['acc']
    
    
