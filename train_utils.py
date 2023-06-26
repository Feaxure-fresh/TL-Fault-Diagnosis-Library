import os
import math
import torch
import logging
import importlib
from torch import optim
from torch.utils.data.dataset import ConcatDataset


class InitTrain(object):
    
    def __init__(self, args):
        self.args = args
        if args.cuda_device:
            self.device = torch.device("cuda:"+args.cuda_device)
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
        else:
            raise Exception("lr schedule not implemented")
            
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
        if args.opt == 'sgd':
            optimizer = optim.SGD(par, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            optimizer = optim.Adam(par, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implemented")
            
        return optimizer
    
    
    def _get_tradeoff(self, tradeoff_list, epoch):
        '''
        Get trade-off parameters for loss.
        '''
        tradeoff = []
        for item in tradeoff_list:
            if item == 'exp':
                tradeoff.append(2 / (1 + math.exp(-10 * (epoch-1) / (self.args.max_epoch-1))) - 1)
            else:
                tradeoff.append(item)
                
        return tradeoff
    
    
    def _init_data(self, concat_src=False, concat_all=False):
        '''
        Initialize the datasets.
        '''
        args = self.args
        
        self.datasets = {}
        idx = 0          
        for i, source in enumerate(args.source_name):
            if args.train_mode == 'multi_source':
                idx = i
            if '_' in source:
                src, op = source.split('_')[0], source.split('_')[1]
                data_root = os.path.join(args.data_dir, src)
                try:
                    Dataset = importlib.import_module("data_loader.{}".format('%s_op' % src)).dataset
                except:
                    raise Exception("data name type not implemented")
                self.datasets[source] = Dataset(data_root, args.normlizetype, random_state=args.random_state,
                                                op=op).data_preprare(source_label=idx, is_src=True)
            else:
                data_root = os.path.join(args.data_dir, source)
                try:
                    Dataset = importlib.import_module("data_loader.{source}").dataset
                except:
                    raise Exception("data name type not implemented")
                self.datasets[source] = Dataset(data_root, args.normlizetype, random_state=args.random_state,
                                                ).data_preprare(source_label=idx, is_src=True) 
        for key in self.datasets.keys():
            logging.info('source set {} length {}.'.format(key, len(self.datasets[key])))
            self.datasets[key].summary()
        
        if '_' in args.target_name:
            tgt, op = args.target_name.split('_')[0], args.target_name.split('_')[1]
            data_root = os.path.join(args.data_dir, tgt)
            try:
                Dataset = importlib.import_module("data_loader.{}".format('%s_op' % tgt)).dataset
            except:
                raise Exception("data name type not implemented")
            self.datasets['train'], self.datasets['val'] = Dataset(data_root, args.normlizetype, random_state=args.random_state,
                                                           op=op).data_preprare(source_label=idx+1)
        else:
            data_root = os.path.join(args.data_dir, args.target_name).dataset
            try:
                Dataset = importlib.import_module("data_loader.{args.target_name}")
            except:
                raise Exception("data name type not implemented")
            self.datasets['train'], self.datasets['val'] = Dataset(data_root, args.normlizetype, random_state=args.random_state,
                                                                    ).data_preprare(source_label=idx+1)           
        logging.info('training set length {}, validation set length {}.'.format(
                                         len(self.datasets['train']), len(self.datasets['val'])))
        self.datasets['train'].summary(); self.datasets['val'].summary()
        
        sources = args.source_name + ['train', 'val']
        if concat_src:
            self.datasets['concat_source'] = ConcatDataset([self.datasets[s] for s in args.source_name])
            sources.append('concat_source')
        if concat_all:
            self.datasets['concat_all'] = ConcatDataset([self.datasets[s] for s in args.source_name]+[self.datasets['train']])
            sources.append('concat_all')

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                              batch_size=args.batch_size,
                                              shuffle=(False if x == 'val' else True),
                                              num_workers=args.num_workers, drop_last=True,
                                              pin_memory=(True if self.device == 'cuda' else False))
                                              for x in sources}
        self.iters = {x: iter(self.dataloaders[x]) for x in sources}
