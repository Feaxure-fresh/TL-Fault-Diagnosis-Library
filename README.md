# Transfer learning for Fault Diagnosis

## Introduction
This repository is an open-source library for cross-domain fault diagnosis, including Single-source Unsupervised Domain Adaptation (SUDA) and Multi-source Unsupervised Domain Adaptation (MUDA) methods.

## Supported Methods
- **ACDANN** - Integrating expert knowledge with domain adaptation for unsupervised fault diagnosis. [[TIM 2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9612159) [[Code]](/models/ACDANN.py)
- **ADACL** - Adversarial domain adaptation with classifier alignment for cross-domain intelligent fault diagnosis of multiple source domains. [[MST 2020]](https://iopscience.iop.org/article/10.1088/1361-6501/abcad4/pdf) [[Code]](/models/ADACL.py)
- **BSP** - Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation. [[ICML 2019]](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) [[Code]](/models/BSP.py) 
- **CDAN** - Conditional adversarial domain adaptation. [[NIPS 2018]](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) [[Code]](/models/CDAN.py) 
- **CORAL** - Deep coral: Correlation alignment for deep domain adaptation. [[ECCV 2016]](https://arxiv.org/abs/1607.01719) [[Code]](/models/CORAL.py)
- **DAN** - Learning transferable features with deep adaptation networks. [[ICML 2015]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) [[Code]](/models/DAN.py)
- **DANN** - Unsupervised domain adaptation by backpropagation. [[ICML 2015]](http://proceedings.mlr.press/v37/ganin15.pdf) [[Code]](/models/DANN.py)
- **IRM** - Invariant risk minimization. [[ArXiv]](https://arxiv.org/abs/1907.02893) [[Code]](/models/IRM.py)
- **MCD** - Maximum classifier discrepancy for unsupervised domain adaptation. [[CVPR 2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) [[Code]](/models/MCD.py)
- **MDD** - Bridging theory and algorithm for domain adaptation. [[ICML 2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Code]](/models/MDD.py)
- **MFSAN** - Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources. [[AAAI 2019]](https://ojs.aaai.org/index.php/AAAI/article/view/4551) [[Code]](/models/MFSAN.py) 
- **MSSA** - A multi-source information transfer learning method with subdomain adaptation for cross-domain fault diagnosis. [[Knowledge-Based Systems 2022]](https://reader.elsevier.com/reader/sd/pii/S0950705122001927?token=03BD384CA5D6E0E7E029B23C739C629913DE8F8BB37F6331F7D233FB6C57599BFFC86609EE63BE2F9FC43871D96A2F61&originRegion=us-east-1&originCreation=20230324021230) [[Code]](/models/MSSA.py)
- **MixStyle** - Domain generalization with mixstyle. [[ICLR 2021]](https://arxiv.org/abs/2104.02008) [[Code]](/models/MixStyle.py)

## Installation
### Prerequisites
*  python3 (>=3.8)
*  Pytorch (>=1.10)
*  numpy (>=1.21.2)
*  pandas (>=1.5.3)
*  tqdm (>=4.46.1)
*  scipy (>=1.10)

## Data Preparation
### Download datasets
Download the data from a public bearing or gearbox fault dataset. Loading code for the following datasets is provided referring to [this](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark).
- **[CWRU](https://engineering.case.edu/bearingdatacenter)** - Case Western Reserve University dataset.
- **[MFPT](https://www.mfpt.org/fault-data-sets)** - Machinery Failure Prevention Technology dataset.
- **[PU](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter)** - Paderborn University dataset.
- **[XJTU](https://biaowang.tech/xjtu-sy-bearing-datasets)** - Xi’an Jiaotong University dataset.
- **[IMS](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset?resource=download)** - Intelligent Maintenance Systems dataset.

### Within-dataset transfer
According to different operation conditions, divide a specific dataset into folders like "op_0", "op_1" and so on. In each "op_?" folder, use subfolders for different categories, which contain the fault data.

For example, CWRU dataset can be divided into 4 folders according to 4 motor speed. In each folder, data of this operation condition can be classified into 9 fault classes, such as 7 mil Inner Race fault, 14 mil Inner Race fault, 7 mil Outer Race fault and so on (referring to [this article](https://ieeexplore.ieee.org/abstract/document/9399341)). Then, the dataset folder is organized as
```
.
└── dataset
    └── CWRU
        ├── op_0
        │   ├── ball_07
        │   │   └── 122.mat
        │   ├── inner_07
        │   │   └── 109.mat
        │   ...
        ├── op_1
        │   ├── ball_07
        │   │   └── 123.mat
        │   ...
        ├── op_2
        ...
```

### Cross-dataset transfer
You can also try to implement transfer among different datasets. In this case, the categories of faults contained in each dataset must be the same.

For example, organize CWRU and MFPT datasets as follows for one-to-one transfer.
```
.
└── dataset
    ├── CWRU
    │   ├── inner
    |   |    ├── ***.mat
    |   |    |   ***.mat
    |   |    ...
    │   ├── normal
    │   └── outer
    └── MFPT
        ├── inner
        ├── normal
        └── outer
```
Note: It is highly recommended to modify the dataset loading code based on custom training. Make sure that `datasetname` in the loading code is consistent with names of your subfolders. The sampling length can also be changed by adjusting the `signal_size` inside.

## Usage
### Load trained weights
```shell
python train.py --model_name CNN --load_path ./CNN/single_source/model.pth --target_name CWRU_3 --num_classes 9 --cuda_device 0
```
### Within-dataset transfer
One-to-one transfer (such as CWRU operation condition 0 to condition 1).
```shell
python train.py --model_name CNN --source_name CWRU_0 --target_name CWRU_1 --train_mode single_source --num_classes 9 --cuda_device 0
``` 
Many-to-one transfer. 
```shell
python train.py --model_name MFSAN --source_name CWRU_0,CWRU_1 --target_name CWRU_2 --train_mode multi_source --num_classes 9 --cuda_device 0
``` 
### Cross-dataset transfer
One-to-one transfer.
```shell
python train.py --model_name CNN --source_name CWRU --target_name MFPT --train_mode single_source --num_classes 3 --cuda_device 0
``` 
Many-to-one transfer. 
```shell
python train.py --model_name MFSAN --source_name CWRU,PU --target_name MFPT --train_mode multi_source --num_classes 3 --cuda_device 0
``` 

## Contact
If you have any problem with our code or have some suggestions, feel free to contact Jinyuan Zhang (feaxure@outlook.com) or describe it in Issues.

## Citation
If you use this toolbox or benchmark in your research, please cite this project. 
```latex
@misc{dafd,
    author = {Jinyuan Zhang},
    title = {TL-Bearing-Fault-Diagnosis},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/Feaxure-fresh/TL-Bearing-Fault-Diagnosis}},
}
```

