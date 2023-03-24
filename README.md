# Domain Adaptation for Fault Diagnosis

## Introduction
This repository is an open-source library for cross-domain fault diagnosis, including Single-source Unsupervised Domain Adaptation (SUDA) and Multi-source Unsupervised Domain Adaptation (MUDA) methods.

## Supported Methods
- **ACDANN** - Integrating expert knowledge with domain adaptation for unsupervised fault diagnosis. [[TIM 2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9612159) [[Code]](/models/ACDANN.py)
- **ADACL** - Adversarial domain adaptation with classifier alignment for cross-domain intelligent fault diagnosis of multiple source domains. [[MST 2020]](https://iopscience.iop.org/article/10.1088/1361-6501/abcad4/pdf) [[Code]](/models/ADACL.py)
- **BSP** - Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation. [[ICML 2019]](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) [[Code]](/models/BSP.py) 


# Installation
### Prerequisites
*  python3 (>=3.8)
*  Pytorch (>=1.10)
*  numpy (=1.21.2)
*  pandas (=1.3.5)
*  tqdm (=4.62.3)
*  matplotlib (=3.5.0)
