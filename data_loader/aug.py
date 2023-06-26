import random
import numpy as np
import pandas as pd
from scipy.signal import resample
from torch.utils.data import Dataset
from collections import defaultdict


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq


class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","-1-1","mean-std"
        self.type = type
        
    def __call__(self, seq):
        if  self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')
        return seq


def train_test_split_(data_pd, test_size=0.2, num_classes=3, random_state=10):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    rng = np.random.default_rng(random_state)
    for i in range(num_classes):
        if type(test_size) == float:
            data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
            size = [i for i in range(data_pd_tmp.shape[0])]
            rng.shuffle(size)
            train_pd = pd.concat([train_pd, data_pd_tmp.loc[size[:int((1-test_size)*data_pd_tmp.shape[0])],
                                                   ['data', 'label']]], ignore_index=True)
            val_pd = pd.concat([val_pd, data_pd_tmp.loc[size[int((1-test_size)*data_pd_tmp.shape[0]):],
                                                   ['data', 'label']]], ignore_index=True)
        elif type(test_size) == list:
            assert len(test_size) == num_classes
            data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
            train_pd = pd.concat([train_pd, data_pd_tmp.loc[size[:int((1-test_size[i])*data_pd_tmp.shape[0])],
                                                   ['data', 'label']]], ignore_index=True)
            val_pd = pd.concat([val_pd, data_pd_tmp.loc[size[int((1-test_size[i])*data_pd_tmp.shape[0]):],
                                                   ['data', 'label']]], ignore_index=True)
        else:
            raise Exception("unknown test size type")
    return train_pd,val_pd


def balance_data(data_pd):
    count = data_pd.value_counts(subset='label')
    min_len = min(count) - 1
    df = pd.DataFrame(columns=('data', 'label'))
    for i in count.keys():
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        df = pd.concat([df, data_pd_tmp.loc[:min_len, ['data', 'label']]], ignore_index=True)
    return df
        

class dataset(Dataset):
    def __init__(self, list_data, source_label=None, test=False, transform=None):
        self.test = test
        self.source_label = source_label
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
                
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
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
