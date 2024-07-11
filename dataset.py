import matplotlib.pyplot as plt
import torch
import os
import random
from torchvision import datasets
import numpy as np
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from utils import read_image


def data_split(data_path, split_rate=0.2):

    sar_data = datasets.ImageFolder(data_path)

    random.seed(666)

    eval_index = random.sample(sar_data.imgs, int(split_rate * len(sar_data)))

    train_info = []
    val_info = []
    for idx, info in enumerate(sar_data.imgs):
        if info in eval_index:
            path, label = info
            val_info.append(path + ' ' + str(label))
        else:
            path, label = info
            train_info.append(path + ' ' + str(label))

    with open(data_path+'train_info.txt', 'a') as f:
        for info in train_info:
            f.write(info + '\n')

    with open(data_path+'val_info.txt', 'a') as f:
        for info in val_info:
            f.write(info + '\n')

# class SARClassificationDataset(Dataset):
#     def __init__(self, data_dir, transform=None, train=True):
#         self.transform = transform
#         self.train = train
#         self.label_list = list()
#
#         # if train:
#         #     self.dataset = datasets.ImageFolder(data_dir+'train')
#         # else:
#         #     self.dataset = datasets.ImageFolder(data_dir+'val')
#         self.dataset = datasets.ImageFolder(data_dir)
#
#     def __getitem__(self, idx):
#         images_path, label = self.dataset.imgs[idx]
#         images = cv.imread(images_path)
#         images = cv.cvtColor(images, cv.COLOR_BGR2RGB)
#         images = Image.fromarray(images)
#         if self.transform:
#             images = self.transform(images)
#         label = torch.tensor(label)
#         return transforms.ToTensor()(images), label
#
#     def __len__(self):
#         return len(self.dataset.imgs)

def preprocess(sar_data, train=None):
    positive = 0
    negative = 0
    data_info = sar_data.imgs
    random.shuffle(data_info)
    images_info = []
    if train == 'train':
        # data_info_train = data_info[:6000]
        data_info_train = data_info
        for i in range(len(data_info_train)):
            image1 = random.choice(data_info_train)
            for j in range(10):
                same_class = random.randint(0, 1)
                if same_class:
                    positive += 1
                    while True:
                        image2 = random.choice(data_info_train)
                        if image1[0] == image2[0]:
                            continue
                        elif image1[1] == image2[1]:
                            images_info.append([image1[0], image2[0], image1[1], image2[1]])
                            break
                else:
                    negative += 1
                    while True:
                        image2 = random.choice(data_info_train)
                        if image1[0] == image2[0]:
                            continue
                        elif image1[1] != image2[1]:
                            images_info.append([image1[0], image2[0], image1[1], image2[1]])
                            break
        print("postive: {}\t negative: {}".format(positive, negative))
    else:
        # data_info_test = data_info[6000:]
        data_info_test = data_info
        for i in range(len(data_info_test)):
            image1 = random.choice(data_info_test)
            for j in range(10):
                while True:
                    image2 = random.choice(data_info_test)
                    if image1[0] != image2[0]:
                        break
                    else:
                        continue
                images_info.append([image1[0], image2[0], image1[1], image2[1]])

    random.shuffle(images_info)
    return images_info


class SARClassificationDataset(Dataset):
    def __init__(self, ImageFolderData, transform=None, mode="train"):
        self.ImageFolderData = ImageFolderData
        self.transform = transform
        self.mode = mode
        self.images_info = []
        self.images = []
        self.labels = []

        if self.mode == 'train':
            self.images_info = preprocess(self.ImageFolderData, 'train')
            for i in range(len(self.images_info)):
                self.images.append(self.images_info[i][0])
                self.images.append(self.images_info[i][1])
                self.labels.append(self.images_info[i][2])
                self.labels.append(self.images_info[i][3])
        else:
            self.images_info = preprocess(self.ImageFolderData, 'test')
            for i in range(len(self.images_info)):
                self.images.append(self.images_info[i][0])
                self.images.append(self.images_info[i][1])
                self.labels.append(self.images_info[i][2])
                self.labels.append(self.images_info[i][3])

    def __getitem__(self, idx):
        images, labels = self.images[idx], self.labels[idx]
        images = cv.imread(images, cv.IMREAD_LOAD_GDAL)
        images = cv.cvtColor(images, cv.COLOR_BGR2RGB)
        # images = Image.open(images).convert('RGB')
        images = Image.fromarray(images)
        if self.transform:
            images = self.transform(images)
        labels = torch.tensor(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, transforms=None, mode="RGB"):
        self.mode = mode
        self.transforms = transforms
        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, 'r') as f:
            for line in f:
                _path, _label = line.strip().split(",")
                self.path_list.append(_path)
                self.label_list.append(int(_label))

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label


if __name__ == '__main__':
    from torchvision import datasets, transforms
    import os
    from collate_batch import collate_fn
    from random_identity_sampler import RandomIdentitySampler
    from loss.margin_loss import *
    from loss.triplet_loss import TripletLoss
    from loss.binomial_deviance_loss import BinDevianceLoss
    from backbone import *
    import torch.nn.functional as F
    from evaluate import evaluate
    from Mahalanobis import MahalanobisLayer

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    batch_size = 64
    # root = 'E:/CodeforFuture/Data/SAR_data/'

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    path = 'E:/CodeforFuture/Data/MSTAR_SOC/'

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    train_soc = datasets.ImageFolder(path + 'train/')
    test_soc = datasets.ImageFolder(path + 'test/')
    print(len(train_soc))
    print(len(test_soc))

    train_data = SARClassificationDataset(train_soc, transform_train)
    val_data = SARClassificationDataset(test_soc, transform_test, mode='test')

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size, shuffle=True, num_workers=8)
    images, labels = next(iter(train_loader))

    print(images.size())
