import os
import pandas as pd
from scipy.io import loadmat
import sys  
sys.path.append('/home/zjy/Course/Diagnosis/data_loader')
from sequence_aug import *

signal_size = 1024

datasetname = ["inner_race_3_021", "outer_race_3_021_6", "normal_3"]

axis = ["_DE_time", "_FE_time", "_BA_time"]

data = []
lab = []

def get_files(root):
    '''
    root: The location of the data set.
    '''
    data_normal = os.path.join(root, datasetname[2])
    data_ir = os.path.join(root, datasetname[0])
    data_or = os.path.join(root, datasetname[1])

    for item in os.listdir(data_normal):
        if item.endswith('.mat'):
            item_path = os.path.join(data_normal, item)
    
            # The label for normal data is 0
            data_load(item_path, label=0)
            
    for item in os.listdir(data_ir):
        if item.endswith('.mat'):
            item_path = os.path.join(data_ir, item)
    
            # The label for inner race data is 1
            data_load(item_path, label=1)
    
        for item in os.listdir(data_or):
            if item.endswith('.mat'):
                item_path = os.path.join(data_or, item)
        
                # The label for outer race data is 2
                data_load(item_path, label=2)


def data_load(item_path, label):
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


class CWRU(object):
    
    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, is_src=False):
        get_files(self.data_dir)
        data_pd = pd.DataFrame({"data": data, "label": lab})
        data_pd = balance_data(data_pd)
        if is_src:
            train_dataset = dataset(list_data=data_pd, transform=data_transforms('train', self.normlizetype))
            return train_dataset
        else:
            train_pd, val_pd = train_test_split_(data_pd, test_size=0.2)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
            return train_dataset, val_dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    root = '/home/zjy/Course/Diagnosis/Dataset/CWRU'
    train_dataset = CWRU(root, "-1-1").data_preprare(True)
    train_dataset.summary()
    print(len(train_dataset))
    
    # midpoint = len(train_dataset) / 3
    # for i in range(3):
    #     fig = plt.figure(figsize=(25, 12))
    #     for j in range(8):
    #         ax = fig.add_subplot(241 + j)
    #         ax.plot(train_dataset[int(midpoint*i+j*50)][0].reshape(-1,1))
    #         ax.set_xlabel('time')
    #         plt.legend('%d' % train_dataset[int(midpoint*i+j)][1])  
