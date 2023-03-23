import os
import pandas as pd

import aug

signal_size=1024

datasetname = ["normal", "inner_race", "outer_race"]

#working condition
WC = ["35Hz12kN","37.5Hz11kN","40Hz10kN"]

#generate Training Dataset and Testing Dataset
def get_files(root, op=1):
    '''
    root:The location of the data set
    '''
    data, lab = [], []
    
    for idx, name in enumerate(datasetname):
        data_dir = os.path.join(root, WC[op], name)

        for item in os.listdir(data_dir):
            if item.endswith('.csv'):
                item_path = os.path.join(data_dir, item)
                data_load(item_path, idx, data, lab)

    return data, lab


def data_load(filename, label, data, lab):
    '''
    This function is mainly used to generate test data and training data.
    '''
    fl = pd.read_csv(filename)
    fl = fl["Horizontal_vibration_signals"]
    fl = fl.values
    fl = fl.reshape(-1,1)

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
            aug.etype()
        ])
    }
    return transforms[dataset_type]


class XJTU_op(object):

    def __init__(self, data_dir, normlizetype, random_state=10, op=1):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.op = int(op)
        self.random_state = random_state

    def data_preprare(self, is_src=False):
        data, lab = get_files(self.data_dir, self.op)
        data_pd = pd.DataFrame({"data": data, "label": lab})
        data_pd = aug.balance_data(data_pd)
        if is_src:
            train_dataset = aug.dataset(list_data=data_pd, transform=data_transforms('train', self.normlizetype))
            return train_dataset
        else:
            train_pd, val_pd = aug.train_test_split_(data_pd, test_size=0.2, num_classes=len(datasetname), random_state=self.random_state)
            train_dataset = aug.dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            val_dataset = aug.dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
            return train_dataset, val_dataset


if __name__ == '__main__':
    root = '../dataset/XJTU'
    train_dataset = XJTU_op(root, "-1-1").data_preprare(True)
    train_dataset.summary()
