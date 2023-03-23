import os
import pandas as pd
from scipy.io import loadmat

import aug

signal_size = 1024

datasetname = ["inner_race_01", "outer_race_01", "normal_001"]
sub_dir_nor = ["K001"]
sub_dir_ir = ["KI01"]
sub_dir_or = ["KA01"]

#working condition
WC = ["N9M7F10", "N15M7F10", "N15M1F10", "N15M7F04"]
state = WC[0]

def get_files(root):
    '''
    root: The location of the data set.
    '''
    data, lab = [], []
    for i in sub_dir_nor:
        data_normal = os.path.join(root, datasetname[2], i)
        for item in os.listdir(data_normal):
            if item.endswith('.mat') and state in item:
                item_path = os.path.join(data_normal, item)
        
                # The label for normal data is 0
                data_load(item_path, label=0)
            
    for i in sub_dir_ir:
        data_ir = os.path.join(root, datasetname[0], i)
        for item in os.listdir(data_ir):
            if item.endswith('.mat') and state in item:
            
                item_path = os.path.join(data_ir, item)
        
                # The label for inner race data is 1
                data_load(item_path, label=1)
    
    for i in sub_dir_or:
        data_or = os.path.join(root, datasetname[1], i)
        for item in os.listdir(data_or):
            if item.endswith('.mat') and state in item:
                item_path = os.path.join(data_or, item)
        
                # The label for outer race data is 2
                data_load(item_path, label=2)
                
    return data, lab

def data_load(item_path, label, data, lab):
    '''
    This function is mainly used to generate test data and training data.
    '''
    name = os.path.basename(item_path).split(".")[0]
    fl = loadmat(item_path)[name]
    fl = fl[0][0][2][0][6][2]  #Take out the data
    fl = fl.reshape(-1,1)

    start,end = 0, signal_size
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


class PU(object):

    def __init__(self, data_dir, normlizetype, random_state=10):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.random_state = random_state

    def data_preprare(self, is_src=False):
        data, lab = get_files(self.data_dir)
        data_pd = pd.DataFrame({"data": data, "label": lab})
        data_pd = aug.balance_data(data_pd)
        if is_src:   
            train_dataset = aug.dataset(list_data=data_pd, transform=data_transforms('train', self.normlizetype))
            return train_dataset
        else:
            train_pd, val_pd = aug.train_test_split_(data_pd, test_size=0.20, num_classes=len(datasetname), random_state=self.random_state)
            train_dataset = aug.dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            val_dataset = aug.dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
            return train_dataset, val_dataset
        
        
if __name__ == '__main__':
    root = '../dataset/PU'
    train_dataset = PU(root, "-1-1").data_preprare(True)
    train_dataset.summary()
