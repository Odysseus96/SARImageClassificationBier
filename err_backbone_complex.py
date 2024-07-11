import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from dataset import BaseDataSet
from backbone import *
from collate_batch import collate_fn
from random_identity_sampler import RandomIdentitySampler
from loss import *
from utils import *

def KL_divergence_loss(preds, labels):
    loss = preds * (torch.log(preds) - labels)
    return loss

def log_cosh(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return loss.sum()

def onehot_encoding(label, n_classes):
    """Conduct one-hot encoding on a label vector."""
    label = label.view(-1)
    onehot = torch.zeros(label.size(0), n_classes).float().to(device)
    onehot.scatter_(1, label.view(-1, 1), 1)

    return onehot

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    import matplotlib
    from matplotlib import rcParams
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    plt.style.use(['science', 'no-latex', 'ieee'])
    config = {
        "font.family": 'serif',
        "font.size": 9,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    x = torch.linspace(0, 4, 30)
    x1 = torch.linspace(0, 10, 20)
    # err = torch.randn(30)
    # print(x**2)
    variance = torch.exp(x) - 4.0
    bias = lambda x : torch.exp(x) - 4.0
    total_err = bias(4.0-x) + variance + 10.0

    plt.plot(x, bias(4.0-x), 'b', label='偏差',linewidth=2.5)
    plt.plot(x, variance, 'r', label='方差',linewidth=2.5)
    plt.plot(x, total_err, 'g--', label='总误差',linewidth=2.5)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("模型复杂度")
    plt.ylabel("误差")
    plt.text(1.55, 22.0, '最优平衡点')
    plt.scatter([2.0], [16.5], c='r')
    plt.yticks(fontproperties='Times New Roman')
    plt.legend()
    plt.tight_layout()
    plt.savefig('E:/CodeforFuture/error vs backbone complex.png')
    plt.show()
    # import os
    # save_dir = 'runs/a-bier/'
    #
    # file_idx = []
    # for file in os.listdir(save_dir):
    #     file_idx.append(int(file.strip('exp')))
    #
    # save_dir += 'exp' + str(max(file_idx) + 1) + '/'
    # print(save_dir)



