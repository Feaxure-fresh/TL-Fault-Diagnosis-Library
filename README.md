# Transfer Learning for Fault Diagnosis

## Overview
Welcome to our repository dedicated to advancing the field of cross-domain fault diagnosis through transfer learning. Our focus lies in Single-source Unsupervised Domain Adaptation (SUDA) and Multi-source Unsupervised Domain Adaptation (MUDA), offering a rich collection of resources and methodologies.

## Featured Methods
Our repository includes a diverse range of state-of-the-art methods, each accompanied by relevant publications and direct access to implementation code in `models`. Highlights include:

- **ACDANN** - Integrating expert knowledge with domain adaptation for unsupervised fault diagnosis. [Published in TIM 2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9612159) | [View Code](/models/ACDANN.py)
- **ADACL** - Adversarial domain adaptation with classifier alignment for cross-domain intelligent fault diagnosis of multiple source domains. [Published in Measurement Science and Technology 2020](https://iopscience.iop.org/article/10.1088/1361-6501/abcad4/pdf) | [View Code](/models/ADACL.py)
- **BSP** - Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation. [Published in ICML 2019](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) | [View Code](/models/BSP.py) 
- **CDAN** - Conditional adversarial domain adaptation. [Published in NIPS 2018](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) | [View Code](/models/CDAN.py) 
- **CORAL** - Deep coral: Correlation alignment for deep domain adaptation. [Published in ECCV 2016](https://arxiv.org/abs/1607.01719) | [View Code](/models/CORAL.py)
- **DAN** - Learning transferable features with deep adaptation networks. [Published in ICML 2015](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) | [View Code](/models/DAN.py)
- **DANN** - Unsupervised domain adaptation by backpropagation. [Published in ICML 2015](http://proceedings.mlr.press/v37/ganin15.pdf) | [View Code](/models/DANN.py)
- **IRM** - Invariant risk minimization. [Published in ArXiv](https://arxiv.org/abs/1907.02893) | [View Code](/models/IRM.py)
- **MCD** - Maximum classifier discrepancy for unsupervised domain adaptation. [Published in CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) | [View Code](/models/MCD.py)
- **MDD** - Bridging theory and algorithm for domain adaptation. [Published in ICML 2019](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) | [View Code](/models/MDD.py)
- **MFSAN** - Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources. [Published in AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4551) | [View Code](/models/MFSAN.py) 
- **MSSA** - A multi-source information transfer learning method with subdomain adaptation for cross-domain fault diagnosis. [Published in Knowledge-Based Systems 2022](https://reader.elsevier.com/reader/sd/pii/S0950705122001927?token=03BD384CA5D6E0E7E029B23C739C629913DE8F8BB37F6331F7D233FB6C57599BFFC86609EE63BE2F9FC43871D96A2F61&originRegion=us-east-1&originCreation=20230324021230) | [View Code](/models/MSSA.py)
- **MixStyle** - Domain generalization with mixstyle. [Published in ICLR 2021](https://arxiv.org/abs/2104.02008) | [View Code](/models/MixStyle.py)

## Getting Started
### Requirements
Our code runs fine with the following prerequisites:
*  Python 3 (>=3.8)
*  Pytorch (>=1.10)
*  Numpy (>=1.21.2)
*  Pandas (>=1.5.3)
*  tqdm (>=4.46.1)
*  Scipy (>=1.10)

### Repository Access
You can access our repository either by direct download or using git clone. Here’s how:
#### Direct Download
1. Click on the 'Code' button and select 'Download ZIP'.
2. Extract the ZIP file to your desired location.
#### Using Git Clone
1. Open your command line interface.
2. Navigate to the directory where you wish to clone the repository.
3. Run the command: `git clone https://github.com/Feaxure-fresh/TL-Bearing-Fault-Diagnosis.git`

## Accessing Datasets
### Supported datasets
Our repository supports several public datasets for fault diagnosis, with accompanying loading code. These include:
- **[CWRU](https://engineering.case.edu/bearingdatacenter)** - Case Western Reserve University dataset.
- **[MFPT](https://www.mfpt.org/fault-data-sets)** - Machinery Failure Prevention Technology dataset.
- **[PU](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter)** - Paderborn University dataset.
- **[XJTU](https://biaowang.tech/xjtu-sy-bearing-datasets)** - Xi’an Jiaotong University dataset.
- **[IMS](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset?resource=download)** - Intelligent Maintenance Systems dataset.

### Setting Up Dataset Directory
1. Create a folder named "datasets" in the root directory of the cloned repository.
2. Download the desired datasets and place them into this "datasets" folder, follow the steps below:

#### Within-dataset Transfer
For analyzing a specific dataset under different working conditions:
1. Divide the dataset into separate folders named "condition_0", "condition_1", etc., each representing a unique operational condition.
2. Within each "condition_?" folder, create subfolders for different fault categories containing the respective fault data.

For example, for the CWRU dataset:
   - Organize the dataset into folders based on motor speed (four speeds as four folders).
   - Within each speed folder, categorize data into 9 fault classes, such as '7 mil Inner Race fault', '14 mil Inner Race fault', '7 mil Outer Race fault', etc., as detailed in [this IEEE article](https://ieeexplore.ieee.org/abstract/document/9399341).
Example folder structure for CWRU dataset:
```
.
└── datasets
    └── CWRU
        ├── condition_0
        │   ├── ball_07
        │   │   └── 122.mat
        │   ├── inner_07
        │   │   └── 109.mat
        │   ...
        ├── condition_1
        │   ├── ball_07
        │   │   └── 123.mat
        │   ...
        ├── condition_2
        ...
```

#### Cross-dataset transfer
For implementing transfer between different datasets:
1. Ensure that the categories of faults contained in each dataset are consistent.
2. Organize the datasets such that there is a direct correspondence between the fault categories across datasets.

For instance, when organizing CWRU and MFPT datasets for one-to-one transfer:
```
.
└── datasets
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
### Custom Dataset Integration
For incorporating other public datasets or your custom datasets, navigate to `data_loader/load_methods.py` in the repository. Implement your data loading function following this template:
```python
def your_dataset_name(item_path):
    # Your code to extract the signal or data from the file
    signal = take_out_data_from_file(item_path)
    return signal
```
This process allows for the seamless integration and utilization of diverse datasets within our fault diagnosis framework.

## Usage
### Within-dataset transfer
One-to-one transfer (such as CWRU operation condition 0 to condition 1).
```shell
python train.py --model_name DAN --source CWRU_0 --target CWRU_1 --faults inner_07,ball_07,outer_07,inner_14,ball_14,outer_14,inner_21,ball_21,outer_21 --train_mode single_source --cuda_device 0
``` 
Many-to-one transfer. 
```shell
python train.py --model_name MFSAN --source CWRU_0,CWRU_1 --target CWRU_2 --faults inner_07,ball_07,outer_07,inner_14,ball_14,outer_14,inner_21,ball_21,outer_21 --train_mode multi_source --cuda_device 0 
``` 
### Cross-dataset transfer
One-to-one transfer.
```shell
python train.py --model_name DAN --source_name CWRU --target_name MFPT --faults normal,inner,outer --train_mode single_source --cuda_device 0
``` 
Many-to-one transfer. 
```shell
python train.py --model_name MFSAN --source_name CWRU,PU --target_name MFPT  --faults normal,inner,outer --train_mode multi_source --cuda_device 0
```
### Load trained weights
```shell
python train.py --model_name MFSAN --load_path ./ckpt/MFSAN/multi_source/***.pth --faults inner_07,ball_07,outer_07,inner_14,ball_14,outer_14,inner_21,ball_21,outer_21 --source CWRU_0,CWRU_1 --target CWRU_3 --cuda_device 0
```
Note: "--source" does not need to be defined for some models when loading weight testing. But for some models, definition is required, but the definition is only to know the number of source fields to define the model structure, and it does not matter what the sources are.

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

