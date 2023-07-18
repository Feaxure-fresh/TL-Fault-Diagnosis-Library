import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Domain Adaptation for Fault Diagnosis')
 
    # basic parameters
    parser.add_argument('--model_name', type=str, default='DAN',
                        help='the name of the model (must in ./models directory)')
    parser.add_argument('--source_name', type=str, default='CWRU_1',
                        help='the name of the source data (select different conditions of a dataset with dataset_num, such as CWRU_0)')
    parser.add_argument('--target_name', type=str, default='CWRU_2',
                        help='the name of the target data (select different conditions of a dataset with dataset_num, such as CWRU_0)')
    parser.add_argument('--data_dir', type=str, default="./dataset",
                        help='the directory of the datasets')
    parser.add_argument('--train_mode', type=str, default='single_source',
                        choices=['single_source', 'source_combine', 'multi_source', 'supervised'],
                        help='the mode for training (choose correctly before training)')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='the number of classes for data (must be same for source and target data)')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std'], default='-1-1',
                        help='data normalization methods')
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='assign device, only one GPU will be used ('' means using cpu)')
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='the directory to save the log and the model')
    parser.add_argument('--max_epoch', type=int, default=30,
                        help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='the number of workers for dataloader')
    parser.add_argument('--random_state', type=int, default=1,
                        help='the random state for train test split')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='the type of learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='parameter for learning rate scheduler (not for fix)')
    parser.add_argument('--steps', type=str, default='10',
                        help='the step of learning rate decay for step and stepLR')
    parser.add_argument('--tradeoff', type=list, default=['exp', 'exp', 'exp'],
                        help='trade-off coefficients for sum of losses, integer or exp (exp means increasing from 0 to 1)')
    parser.add_argument('--dropout', type=float, default=0., help='coefficient of dropout layers')
    
    # save, load and display information
    parser.add_argument('--save', type=bool, default=True, help='whether to save trained model')
    parser.add_argument('--load_path', type=str, default='',
                        help='load a trained model from this path if needed')
    args = parser.parse_args()
    return args
    
