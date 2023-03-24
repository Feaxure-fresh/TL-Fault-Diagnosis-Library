# Domain Adaptation for Fault Diagnosis

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
- **MFSAN** - Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources. [[AAAI 2019]]([http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4551/4429)) [[Code]](/models/MFSAN.py) 
- **MSSA** - A multi-source information transfer learning method with subdomain adaptation for cross-domain fault diagnosis. [[Knowledge-Based Systems 2022]](https://reader.elsevier.com/reader/sd/pii/S0950705122001927?token=03BD384CA5D6E0E7E029B23C739C629913DE8F8BB37F6331F7D233FB6C57599BFFC86609EE63BE2F9FC43871D96A2F61&originRegion=us-east-1&originCreation=20230324021230) [[Code]](/models/MSSA.py)
- **MixStyle** - Domain generalization with mixstyle. [[ICLR 2021]](https://arxiv.org/abs/2104.02008) [[Code]](/models/MixStyle.py)

# Installation
### Prerequisites
*  python3 (>=3.8)
*  Pytorch (>=1.10)
*  numpy (=1.21.2)
*  pandas (=1.3.5)
*  tqdm (=4.62.3)
*  matplotlib (=3.5.0)
