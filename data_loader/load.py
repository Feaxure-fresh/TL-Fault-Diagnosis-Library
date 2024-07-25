import os
import pandas as pd

import aug
import data_utils
import load_methods


def get_files(root, dataset, faults, fault_label, signal_size):
    data, actual_labels = [], []
    data_load = getattr(load_methods, dataset)
    
    for _, name in enumerate(faults):
        data_dir = os.path.join(root, name)

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            signal = data_load(item_path)

            start, end = 0, signal_size
            while end <= signal.shape[0]:
                data.append(signal[start:end])
                actual_labels.append(fault_label[name])
                start += signal_size
                end += signal_size

    return data, actual_labels


def data_transforms(normlize_type="-1-1"):
    transforms = {
        'train': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()

        ]),
        'val': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    }
    return transforms


class dataset(object):
    
    def __init__(self, args, dataset, source_idx, balance_data=False, test_size=0.2):
        self.args = args
        data_root = os.path.join(args.data_dir, dataset)
        faults = args.faults[source_idx]
        signal_size = args.signal_size
        normlize_type = args.normlize_type
        fault_label = args.fault_label
        self.label_set = args.label_sets[source_idx]
        self.random_state = args.random_state
        self.balance_data = balance_data
        self.test_size = test_size

        self.data, self.actual_labels = get_files(data_root, dataset, faults, fault_label, signal_size)
        self.transform = data_transforms(normlize_type)

    def data_preprare(self, is_src=False):
        data_pd = pd.DataFrame({"data": self.data, "actual_labels": self.actual_labels})
        data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
        if is_src:
            train_dataset = data_utils.dataset(list_data=data_pd, transform=self.transform['train'])
            return train_dataset
        else:
            train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size, label_set=self.label_set, random_state=self.random_state)
            train_dataset = data_utils.dataset(list_data=train_pd, transform=self.transform['train'])
            val_dataset = data_utils.dataset(list_data=val_pd, transform=self.transform['val'])
            return train_dataset, val_dataset
