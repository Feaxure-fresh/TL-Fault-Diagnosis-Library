import os
import sys
sys.path.extend(['./models', './data_loader'])
import torch
import logging
import importlib
import numpy as np
from datetime import datetime

import utils
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
    
    file_name = '[' + '_'.join(args.source_name) + ']' + 'To' + '[' +\
            args.target + ']' + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.save_dir, args.model_name, args.train_mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_path = os.path.join(save_dir, file_name)
    
    # set the logger
    logger = setlogger(args.save_path  + '.log')

    # save the args
    for k, v in args.__dict__.items():
        if k != 'source_name':
            logging.info("{}: {}".format(k, v))
    return logger, args


def get_fault(name, args):
    dataset, condition, selected_list = utils.get_info_from_name(name)
    if condition is not None:
        data_root = os.path.join(args.data_dir, dataset)
        faults = np.array(sorted(os.listdir(os.path.join(data_root, 'condition_%d' % condition))))
    else:
        data_root = os.path.join(args.data_dir, dataset)
        faults = np.array(sorted(os.listdir(data_root)))
    if selected_list:
        faults = faults[selected_list]
    num_classes = len(faults)
    return faults, num_classes


def determine_da_scenario(label_sets):
    # Extract source and target labels
    source_labels = label_sets[:-1]
    target_labels = label_sets[-1]

    # Flatten the source labels and convert to a set to get unique labels
    source_labels_flat = set([label for sublist in source_labels for label in sublist])
    target_labels_set = set(target_labels)

    # Check conditions for different domain adaptation scenarios
    if source_labels_flat == target_labels_set:
        return 'closed-set'
    elif target_labels_set.issubset(source_labels_flat):
        return 'partial'
    elif source_labels_flat.issubset(target_labels_set):
        return 'open-set'
    else:
        return 'universal'


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '8'
    args = parse_args()
    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.backends.cudnn.deterministic=True
    
    args.source_name = [x.strip() for x in list(args.source.split(','))]
    if '' in args.source_name:
        args.source_name.remove('')

    if not args.load_path:
        if len(args.source_name) == 1:
            args.train_mode = 'single_source'
        if args.train_mode == 'single_source':
            assert len(args.source_name) == 1, "single_source mode needs one source"
        else:
            assert len(args.source_name) > 1, "source_combine and multi_source mode need more than one source"
    
    # creating directory
    logger, args = creat_file(args)
    
    # getting faults dictionary
    args.faults, args.num_classes = [], []
    for name in args.source_name + [args.target]:
        faults, num_classes = get_fault(name, args)
        args.faults.append(faults)
        args.num_classes.append(num_classes)
    for name, faults, nclasses in zip(args.source_name, args.faults[:-1], args.num_classes[:-1]):
        logging.info('Source {} detected {} classes: {}'.format(name, nclasses, faults))
    logging.info('Target {} detected {} classes: {}'.format(args.target, args.num_classes[-1], args.faults[-1]))
    
    # getting mapping of fault to label
    all_faults = set()
    for faults in args.faults:
        for item in faults:
            all_faults.add(item)
    args.fault_label = {}
    for i, fault in enumerate(sorted(all_faults)):
        args.fault_label[fault] = i
    if args.train_mode == 'source_combine':
        source_faults_flat = sorted(list(set([fault for sublist in args.faults[:-1] for fault in sublist])))
        args.faults.insert(0, source_faults_flat)
        args.num_classes.insert(0, len(source_faults_flat))

    # getting sets of labels
    args.label_sets = list()
    for faults in args.faults:
        args.label_sets.append([args.fault_label[item] for item in faults])
    
    # determine current DA scenario
    args.da_scenario = determine_da_scenario(args.label_sets)
    logging.info('The scenario is: {} domain adaptation'.format(args.da_scenario))

    # training
    trainer = importlib.import_module(f"models.{args.model_name}").Trainer(args)
    if args.load_path:
        trainer.load_model()
        trainer.test()
        os.remove(args.save_path + '.log')
    else:
        trainer.train()
        if args.save:
            trainer.save_model()
        else:
            os.remove(args.save_path + '.log')
    logger.handlers.clear()
