import os
import importlib
import pandas as pd
from scipy.io import loadmat

import aug
import data_utils
import load_methods


def get_files(root, dataset, faults, signal_size, condition=3):
    '''
    root: The location of the data set.
    condition: The working condition.
    num_classes: The number of classes.
    condition: The working condition.
    '''
    data, labels = [], []
    data_load = getattr(load_methods, dataset)
    
    for index, name in enumerate(faults):
        data_dir = os.path.join(root, 'condition_%d' % condition, name)

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            signal = data_load(item_path)

            start, end = 0, signal_size
            while end <= signal.shape[0]:
                data.append(signal[start:end])
                labels.append(index)
                start += signal_size
                end += signal_size

    return data, labels


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
    
    def __init__(self, data_dir, dataset, faults, signal_size, normlizetype, condition=2,
                 balance_data=True, test_size=0.2):
        self.num_classes = len(faults)
        self.balance_data = balance_data
        self.test_size = test_size
        self.data, self.labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition)
        self.transform = data_transforms(normlizetype)

    def data_preprare(self, source_label=None, is_src=False, random_state=1):
        data_pd = pd.DataFrame({"data": self.data, "labels": self.labels})
        data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
        if is_src:
            train_dataset = data_utils.dataset(list_data=data_pd, source_label=source_label, transform=self.transform['train'])
            return train_dataset
        else:
            train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size, num_classes=self.num_classes, random_state=random_state)
            train_dataset = data_utils.dataset(list_data=train_pd, source_label=source_label, transform=self.transform['train'])
            val_dataset = data_utils.dataset(list_data=val_pd, source_label=source_label, transform=self.transform['val'])
            return train_dataset, val_dataset