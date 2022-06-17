import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='MSTLN', help='the name of the model')
    parser.add_argument('--source_name', type=list, default=['CWRU_1'], help='the name of the source data')
    parser.add_argument('--target_name', type=str, default='CWRU_1', help='the name of the target data')
    parser.add_argument('--data_dir', type=str, default="./dataset", help='the directory of the data')
    parser.add_argument('--train_mode', type=str, choices=['single_source', 'source_combine', 'multi_source'],
                        default="single_source", help='the mode for training')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std'], default='-1-1', help='data normalization methods')
    parser.add_argument('--cuda_device', type=str, default='', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./Ckpt/src_com', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of training process')
    parser.add_argument('--num_classes', type=int, default=9, help='the classes of data')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--tradeoff', type=list, default=['exp', 0.1, 0.1], help='coefficients of loss')
    parser.add_argument('--dropout', type=float, default=0., help='coefficient of dropout layers')
    parser.add_argument('--steps', type=str, default='10', help='the learning rate decay for step and stepLR')
    parser.add_argument('--C_mode', type=int, default=2, help='the training mode for Man Moe classifier')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=5, help='max number of epoch')
    parser.add_argument('--save_step', type=int, default=0, help='the interval of save training model. 0: no saving')
    args = parser.parse_args()
    return args
    
