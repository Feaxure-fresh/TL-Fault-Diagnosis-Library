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
    source = ''
    for src in args.source_name:
        source += src
    file_name = '[' + source + ']' + 'To' + '[' +\
            args.target_name + ']' + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.save_dir, args.model_name, args.train_mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_path = os.path.join(save_dir, file_name)
    
    # set the logger
    logger = setlogger(args.save_path  + '.log')

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    return logger, args


if __name__ == '__main__': 
    args = parse_args()
    
    if not args.cuda_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    args.source_name = list(args.source_name.split(','))
    if args.train_mode == 'single_source':
        assert len(args.source_name) == 1
    elif args.train_mode == 'supervised':
        assert len(args.source_name) == 0
    else:
        assert len(args.source_name) > 1
    
    # training
    logger, args = creat_file(args)
    trainer = getattr(models, args.model_name).Trainset(args)
    if args.load_path:
        trainer.load_model()
        trainer.test()
    else:
        trainer.train()
        if args.save:
            trainer.save_model()
    logger.handlers.clear()