import os
import sys
sys.path.extend(['./models', './data_loader'])
import logging
from datetime import datetime

import models
from opt import parse_args


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
        
    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    return logger
    

def creat_file(args):
    # prepare the saving path for the model
    file_name = str(args.source_name) + 'To' + \
            args.target_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S') + '.log'
    save_dir = os.path.join(args.checkpoint_dir, args.model_name, args.train_mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    logger = setlogger(os.path.join(save_dir, file_name))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    return logger


def check_multi(args, mode):
    if mode == 1:
        args.source_name = ['CWRU_1']
        args.target_name = 'CWRU_3'
        args.train_mode = 'single_source'
        args.num_classes = 9
    elif mode == 2:
        args.source_name = ['CWRU_2']
        args.target_name = 'CWRU_3'
        args.train_mode = 'single_source'
        args.num_classes = 9
    elif mode == 3:
        args.source_name = ['CWRU_1','CWRU_2']
        args.target_name = 'CWRU_3'
        args.train_mode = 'source_combine'
        args.num_classes = 9
    elif mode == 4:
        args.source_name = ['MFPT_0']
        args.target_name = 'MFPT_1'
        args.train_mode = 'single_source'
        args.num_classes = 3
    elif mode == 5:
        args.source_name = ['MFPT_2']
        args.target_name = 'MFPT_1'
        args.train_mode = 'single_source'
        args.num_classes = 3
    elif mode == 6:
        args.source_name = ['MFPT_0', 'MFPT_2']
        args.target_name = 'MFPT_1'
        args.train_mode = 'source_combine'
        args.num_classes = 3
    elif mode == 7:
        args.source_name = ['CWRU_1','CWRU_2']
        args.target_name = 'CWRU_3'
        args.train_mode = 'multi_source'
        args.num_classes = 9
    elif mode == 8:
        args.source_name = ['MFPT_0', 'MFPT_2']
        args.target_name = 'MFPT_1'
        args.train_mode = 'multi_source'
        args.num_classes = 3
    else:
        pass
    return args
        

if __name__ == '__main__': 
    args = parse_args()
    
    if not args.cuda_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    if args.train_mode == 'single_source':
        assert len(args.source_name) == 1
    elif args.train_mode == 'supervised':
        assert len(args.source_name) == 0
    else:
        assert len(args.source_name) > 1
    
    # training
    if len(args.train_all) > 0:
        for i in range(args.train_all[0], args.train_all[1]+1):
            args = check_multi(args, i)
            logger = creat_file(args)
            trainer = getattr(models, args.model_name).Trainset(args)
            trainer.train()
            logger.handlers.clear()
    else:
        logger = creat_file(args)
        trainer = getattr(models, args.model_name).Trainset(args)
        trainer.train()
        logger.handlers.clear()