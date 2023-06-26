import os
import pandas as pd
from scipy.io import loadmat

import aug

signal_size = 1024

datasetname = ["inner_07", "ball_07", "outer_07", "inner_14", "ball_14",
               "outer_14", "inner_21", "ball_21", "outer_21"]

axis = ["_DE_time", "_FE_time", "_BA_time"]

def get_files(root, op=3):
    '''
    root: The location of the data set.
    '''
    data, lab = [], []
    
    for idx, name in enumerate(datasetname):
        data_dir = os.path.join(root, 'op_%d' % op, name)

        for item in os.listdir(data_dir):
            if item.endswith('.mat'):
                item_path = os.path.join(data_dir, item)
                data_load(item_path, idx, data, lab)

    return data, lab


def data_load(item_path, label, data, lab):
    '''
    This function is mainly used to generate test data and training data.
    '''    
    datanumber = os.path.basename(item_path).split(".")[0]
    if eval(datanumber) < 100:
        realaxis = "X0" + datanumber + axis[0]
    else:
        realaxis = "X" + datanumber + axis[0]
    fl = loadmat(item_path)[realaxis]
    start, end = 0, signal_size
    
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size


def data_transforms(dataset_type="train", normlize_type="-1-1"):
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
    return transforms[dataset_type]


class dataset(object):
    
    def __init__(self, data_dir, normlizetype, random_state=10, op=2):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.op = int(op)
        self.random_state = random_state

    def data_preprare(self, source_label=-1, is_src=False):
        data, lab = get_files(self.data_dir, self.op)
        data_pd = pd.DataFrame({"data": data, "label": lab})
        data_pd = aug.balance_data(data_pd)
        if is_src:
            train_dataset = aug.dataset(list_data=data_pd, source_label=source_label, transform=data_transforms('train', self.normlizetype))
            return train_dataset
        else:
            train_pd, val_pd = aug.train_test_split_(data_pd, test_size=0.2, num_classes=len(datasetname), random_state=self.random_state)
            train_dataset = aug.dataset(list_data=train_pd, source_label=source_label, transform=data_transforms('train', self.normlizetype))
            val_dataset = aug.dataset(list_data=val_pd, source_label=source_label, transform=data_transforms('val', self.normlizetype))
            return train_dataset, val_dataset


if __name__ == '__main__':
    root = '../dataset/CWRU'
    train_dataset, val_dataset = dataset(root, "-1-1").data_preprare()
    train_dataset.summary()
    val_dataset.summary()