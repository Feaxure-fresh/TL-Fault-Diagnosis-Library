import random
import numpy as np
import pandas as pd
from scipy.signal import resample
from torch.utils.data import Dataset
from collections import defaultdict

import aug


def train_test_split_(data_pd, test_size=0.2, num_classes=3, random_state=10):
    train_pd = pd.DataFrame(columns=('data', 'labels'))
    val_pd = pd.DataFrame(columns=('data', 'labels'))
    rng = np.random.default_rng(random_state)
    for i in range(num_classes):
        if type(test_size) == float:
            data_pd_tmp = data_pd[data_pd['labels'] == i].reset_index(drop=True)
            size = [i for i in range(data_pd_tmp.shape[0])]
            rng.shuffle(size)
            train_pd = pd.concat([train_pd, data_pd_tmp.loc[size[:int((1-test_size)*data_pd_tmp.shape[0])],
                                                   ['data', 'labels']]], ignore_index=True)
            val_pd = pd.concat([val_pd, data_pd_tmp.loc[size[int((1-test_size)*data_pd_tmp.shape[0]):],
                                                   ['data', 'labels']]], ignore_index=True)
        elif type(test_size) == list:
            assert len(test_size) == num_classes
            data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
            train_pd = pd.concat([train_pd, data_pd_tmp.loc[size[:int((1-test_size[i])*data_pd_tmp.shape[0])],
                                                   ['data', 'labels']]], ignore_index=True)
            val_pd = pd.concat([val_pd, data_pd_tmp.loc[size[int((1-test_size[i])*data_pd_tmp.shape[0]):],
                                                   ['data', 'labels']]], ignore_index=True)
        else:
            raise Exception("unknown test size type")
    return train_pd,val_pd


def balance_data(data_pd):
    count = data_pd.value_counts(subset='labels')
    min_len = min(count) - 1
    df = pd.DataFrame(columns=('data', 'labels'))
    for i in count.keys():
        data_pd_tmp = data_pd[data_pd['labels'] == i].reset_index(drop=True)
        df = pd.concat([df, data_pd_tmp.loc[:min_len, ['data', 'labels']]], ignore_index=True)
    return df
        

class dataset(Dataset):
    def __init__(self, list_data, source_label=None, test=False, transform=None):
        self.test = test
        self.source_label = source_label
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['labels'].tolist()
                
        if transform is None:
            self.transforms = aug.Compose([
                aug.Reshape(),
                aug.Retype()
            ])
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label, self.source_label
    
    def summary(self):
        total = defaultdict(int)
        for i in self.labels:
            total[i] += 1
            
        for key in sorted(total.keys()):
            print('Label {} has samples: {}'.format(key, total[key]))
