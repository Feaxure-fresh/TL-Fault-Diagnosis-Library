import os
import logging
from opt import parse_args
from utils import setlogger
from datetime import datetime
from train_utils import train_utils

if __name__ == '__main__': 
    args = parse_args()
    if not args.cuda_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    if args.train_mode == 'single_source':
        assert len(args.source_name) <= 1
    else:
        assert len(args.source_name) > 1
    
    # prepare the saving path for the model
    sub_dir = args.model_name + '_' + str(args.source_name) + 'To' + \
            args.target_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    
    # training
    trainer = train_utils(args, save_dir)
    if args.model_name == 'Man_Moe':
    	trainer.train_man_moe()
    elif args.model_name == 'MSTLN':
        trainer.train_mstln()
    else:
        if args.train_mode != 'multi_source':
            trainer.train_1src_and_srccomb()
        else:
            trainer.train_multi_src()
        
