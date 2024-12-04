# Compromise Privacy in Large Batch Federated Learning via Model Poisoning

This repository is an implementation of a novel gradient inversion attack which exploits model poisoning to compromise privacy in large batch federated learning.

Related works are published on:
- ICA3PP 2024 (<https://doi.org/10.1007/978-3-031-22677-9_4>)
```
@inproceedings{zhang2022compromise,
  title={Compromise Privacy in Large-Batch Federated Learning via Malicious Model Parameters},
  author={Zhang, Shuaishuai and Huang, Jie and Zhang, Zeping and Qi, Chunyang},
  booktitle={International Conference on Algorithms and Architectures for Parallel Processing},
  pages={63--80},
  year={2022},
  organization={Springer}
}
```
- Information Sciences (<https://doi.org/10.1016/j.ins.2023.119421>)
```
@article{zhang2023compromise,
  title={Compromise privacy in large-batch Federated Learning via model poisoning},
  author={Zhang, Shuaishuai and Huang, Jie and Zhang, Zeping and Li, Peihao and Qi, Chunyang},
  journal={Information Sciences},
  volume={647},
  pages={119421},
  year={2023},
  publisher={Elsevier}
}
```

## Abstract
Federated Learning (FL) is a distributed learning paradigm in which clients collaboratively train a global model with shared gradients while preserving the privacy of local data. Recent researches found that an adversary can reveal private local data with gradient inversion attacks. However, setting a large batchsize in local training can defense against these attacks effectively by confusing gradients computed on private data. Although some advanced methods have been proposed to improve the performance of gradient inversion attacks on the large-batch training, they are limited to specific model architectures (e.g., fully connected neural networks (FCNNs) with ReLU layers). To address these problems, we propose a novel gradient inversion attack to compromise privacy in large-batch FL via model poisoning. We poison clients’ models with malicious parameters, which are constructed purposely to mitigate the confusion of aggregated gradients. For FCNNs, the private data can be perfectly recovered by analyzing gradients of the first fully connected (FC) layer. For convolutional neural networks (CNNs), we extend our proposed method as a hybrid approach, consisting of the analytic method and the optimization-based method. We first recover the feature maps after the convolutional layers and then reconstruct private data by minimizing the loss of data-wise feature map matching. We demonstrate the effectiveness of our proposed method on four datasets and show that it outperforms previous methods for large-batch FL (e.g., 64, 128, 256) and models with different activation layers (e.g., ReLU, Sigmoid and Tanh).
<p align="center">
      <img width="600" height="400" src="./figs/intro.png" alt>
</p>
<p align="center">
    <em>Figure 1: The illustration of our attack method. The server (as an adversary) poisons clients’ models by delivering malicious gradients $\bar{g}$. Then the server can reconstruct private trainsets successfully by inverting shared gradients $g_i$ computed on poisoned models.</em>
</p>

## Environment
This code is implemented in PyTorch, and we have tested the code under the following environment settings:
- python=3.10.13
- pytorch=2.1.0
- pytorch-cuda=11.8
- tensorboard=2.12.1
- torchvision=0.16.0
- matplotlib=3.7.2
- numpy=1.26.0

## Run
### Gradient Inversion Attack via Model Poisoning (GIAvMP)
We have prepared 4 trained malicious parameters saved at "```./model-saved```". To run these attack experiments, follow these instructions:

1. GIAvMP attack on a FCNN training with FashionMNIST
```
python GIAviaMP.py --DATA FashionMNIST --BATCHSIZE 64 --MPfile MP-saved/FCNN-FashionMNIST/FC1_MP.pth
```
Here we show the recovered results of one batch with 64 training images.
<p align="center">
      <img width="242" height="482" src="./figs/FCNN-FMNIST-B64.png" alt>
</p>
<p align="center">
    <em>Figure 2: The recovered data of our attack on a FCNN training with FashionMNIST. The batchsize is 64. The odd columns show the original images and the even columns
show the recovered images..</em>
</p>

2. GIAvMP attack on a FCNN training with CIFAR100
```
python GIAviaMP.py --DATA CIFAR100 --BATCHSIZE 64 --MPfile MP-saved/FCNN-CIFAR100/FC1_MP.pth
```
Here we show the recovered results of one batch with 64 training images.
<p align="center">
      <img width="274" height="546" src="./figs/FCNN-CIFAR100-B64.png" alt>
</p>
<p align="center">
    <em>Figure 2: The recovered data of our attack on a FCNN training with CIFAR100. The batchsize is 64. The odd columns show the original images and the even columns
show the recovered images..</em>
</p>

3. GIAvMP attack on a CNN model training with CIFAR100
```
python GIAviaMP.py --DATA CIFAR100 --BATCHSIZE 64 --CNN --MPfile MP-saved/CNN-CIFAR100/FC1_MP.pth
```
Here we show the recovered results of one batch with 64 training images.
<p align="center">
      <img width="274" height="546" src="./figs/CNN-CIFAR100-B64.png" alt>
</p>
<p align="center">
    <em>Figure 2: The recovered data of our attack on a CNN model training with CIFAR100. The batchsize is 64. The odd columns show the original images and the even columns
show the recovered images..</em>
</p>

### Train malicious parameters for the 1st FC layer in models
The file ```trainMP.py``` is used to train malicious parameters for the 1st FC layer in models.

1. Train malicious parameters for FCNN on FashionMNIST
```
python TrainMP.py --DATA FashionMNIST --BATCHSIZE 64 --FCNum 1024 --K 2
```

The trained malicious parameters will be saved at  "```./outputs/MP-FashionMNIST-K2-B64-FCNN/mdoels/..```".

2. Train malicious parameters for CNN models on CIFAR100
```
python TrainMP.py --DATA CIFAR100 --CNN --BATCHSIZE 64 --FCNum 1024 --K 4
```

3. Train malicious parameters for other models and datasets
- Set ```f_dim``` in the function ```get_datset()```, which is the dimensions of the inputs to the 1st FC layer in models. the value of ```f_dim``` will be different according to different models and datasets. 
- For any CNN models, they should be splited into two parts, the **features module** and the **classifier module**.
- In our method, we keep the parameters of **features module** in CNN models unchanged. These freezed parameters should be saved at ```./model-saved/..``` and reload them before training malicious parameters or attacking models.