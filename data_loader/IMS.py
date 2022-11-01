import os
import pandas as pd
import numpy as py
from scipy.io import loadmat
from sequence_aug import *

signal_size = 1024

datasetname = ["normal", "inner_race_c5", "outer_race_c1"]#, "ball_c7"
channel = [0, 4, 0, 6]

def get_files(root):
    '''
    root: The location of the data set.
    '''
    data, lab = [], []
    
    for idx, name in enumerate(datasetname):
        data_dir = os.path.join(root, name)

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            data_load(item_path, idx, channel[idx], data, lab)

    return data, lab


def data_load(item_path, label, channel, data, lab):
    '''
    This function is mainly used to generate test data and training data.
    '''    
    fl = np.loadtxt(item_path)[:, channel]
    start, end = 0, signal_size
    
    while end <= fl.shape[0]:
        data.append(fl[start:end].reshape(-1,1))
        lab.append(label)
        start += signal_size
        end += signal_size


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


class IMS(object):
    
    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, is_src=False):
        data, lab = get_files(self.data_dir)
        data_pd = pd.DataFrame({"data": data, "label": lab})
        # data_pd = balance_data(data_pd)
        if is_src:
            train_dataset = dataset(list_data=data_pd, transform=data_transforms('train', self.normlizetype))
            return train_dataset
        else:
            train_pd, val_pd = train_test_split_(data_pd, test_size=0.2, num_classes=len(datasetname))
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
            return train_dataset, val_dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    root = '../dataset/IMS'
    train_dataset, val_dataset = IMS(root, "-1-1").data_preprare()
    train_dataset.summary()
    
    '''
    midpoint = len(train_dataset) / 4
    for i in range(4):
        fig = plt.figure(figsize=(25, 12))
        for j in range(8):
            ax = fig.add_subplot(241 + j)
            ax.plot(train_dataset[int(midpoint*i+j*100)][0].reshape(-1,1))
            ax.set_xlabel('time')
            plt.legend('%d' % train_dataset[int(midpoint*i+j)][1])
    '''
   
