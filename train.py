import argparse
import os
import logging
from datetime import datetime
from train_utils import train_utils

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, choices=['DA_Resnet1d', 'MSFAN', 'Man_Moe'], default='Man_Moe', help='the name of the model')
    parser.add_argument('--source_name', type=list, default=['CWRU', 'PU'], help='the name of the source data')
    parser.add_argument('--target_name', type=str, default='MFPT', help='the name of the target data')
    parser.add_argument('--data_dir', type=str, default="/home/zjy/Course/Diagnosis/Dataset", help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std'], default='-1-1', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_NA',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/zjy/Course/Diagnosis/Ckpt', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of training process')
    parser.add_argument('--classes', type=int, default=3, help='the classes of data')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--alpha', type=float, default=0.05, help='coefficient of mmd loss')
    parser.add_argument('--lambd', type=float, default=0.002)
    parser.add_argument('--steps', type=str, default='20', help='the learning rate decay for step and stepLR')
    parser.add_argument('--gate_lw', type=float, default=0.1, help='gate loss weight')
    parser.add_argument('--n_critic', type=int, default=1)

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=30, help='max number of epoch')
    parser.add_argument('--save_step', type=int, default=0, help='the interval of save training model. 0: no saving')
    args = parser.parse_args()
    return args

def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    
    # Prepare the saving path for the model
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
    
    trainer = train_utils(args, save_dir)
    if args.model_name == 'Man_Moe':
    	trainer.train_man_moe()
    elif args.model_name == 'DA_Resnet1d':
    	trainer.setup()
    	trainer.train_1src()
    elif args.model_name == 'MSFAN':
    	trainer.setup()
    	trainer.train_2src()
        
