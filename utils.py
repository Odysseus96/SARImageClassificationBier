import os.path

import matplotlib.pyplot as plt
import torch
import os.path as osp
import numpy as np
import torch.nn.functional as F
import cv2 as cv

from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image(img_path, mode='RGB'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while not got_img:
        try:
            # img = Image.open(img_path).convert("RGB")
            img = Image.fromarray(cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB))
            if mode == "BGR":
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))
            got_img = True
        except IOError:
            print(f"IOError incurred when reading '{img_path}'. Will redo.")
            pass
    return img


def load_file(filename):
    with open(filename, 'r') as f:
        f.readline()
        train_loss = []; val_acc = []
        for line in f.readlines():
            curline = line.strip().split(',')
            if len(curline) < 2: continue

            train_loss.append(float(curline[1]))
            val_acc.append(float(curline[2]))
            # adv_loss.append(float(curline[3]))
            # booster1_loss.append(float(curline[3]))
            # booster1_acc.append(float(curline[4]))
            # booster2_loss.append(float(curline[5]))
            # booster2_acc.append(float(curline[6]))
        # return train_loss, val_acc, adv_loss, booster1_loss, booster1_acc, booster2_loss, booster2_acc
        return train_loss, val_acc


def plot_training(filename, save_dir):
    train_loss, val_acc = load_file(filename)

    epochs = [i+1 for i in range(len(train_loss))]

    plt.figure()
    plt.plot(epochs, train_loss, label='training loss')
    plt.ylabel("Training Loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.title("Training Loss")
    plt.savefig(save_dir + "train_loss.png")

    plt.figure()
    plt.plot(epochs, val_acc, label='val acc')
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.title("Val Accuracy")
    plt.savefig(save_dir + "val_acc.png")

def time_record(t, save_dir):
    with open(save_dir+'time.txt', 'a') as f:
        f.write("Training span time: {:.4f} s".format(t))


def get_mean_and_std(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    cmp_bar = tqdm(train_loader)
    for X, _ in cmp_bar:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def checkpoints(save_dir, acc, epoch, model, name):
    # Save checkpoint.
    state = {
        'backbone' : model,
        'acc' : acc,
        'epoch' : epoch
    }
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, save_dir + name)



def onehot_encoding(label, n_classes):
    """Conduct one-hot encoding on a label vector."""
    label = label.view(-1)
    onehot = torch.zeros(label.size(0), n_classes).float().to(device)
    onehot.scatter_(1, label.view(-1, 1), 1)

    return onehot

def pseudo_residual_classification(target, output, n_classes):
    """Compute the pseudo residual for classification with cross-entropyloss."""

    onehot_y = F.one_hot(target, n_classes)

    return torch.max(onehot_y - F.softmax(output, dim=1), dim=1)[0]









