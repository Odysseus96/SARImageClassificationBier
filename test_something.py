import os
import json
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn

from ConfusionMatrix.confusion_matrix import ConfusionMatrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_metric_data():
    # read class_indict
    json_label_path = 'ConfusionMatrix/class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    matrix_name = []
    for name in os.listdir('ConfusionMatrix/'):
        if '.npy' in name:
            matrix_name.append(name[:-4])
    print(matrix_name)

    for name in matrix_name:
        confusion = ConfusionMatrix(num_classes=25, labels=labels, name=name)
        confusion.matrix = np.load(name + '.npy')
        confusion.summary()

def read_metrics(filename):
    precision = []; recall = []; f1_score = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for idx, row in enumerate(reader):
            if len(row) == 0 or idx == 0:
                continue
            precision.append(float(row[0]) * 100)
            recall.append(float(row[1]))
            f1_score.append(float(row[2]))

    return precision, recall, f1_score

classes = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09',
                   'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
                   'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#7FFF00', '#D2691E',
          '#6495ED', '#00008B', '#B8860B', '#A9A9A9',
          '#BDB76B', '#556B2F', '#E9967A', '#2F4F4F',
          '#9400D3', '#FF1493', '#1E90FF', '#FFD700', '#90EE90']

class SARDataSet2(Dataset):
    def __init__(self, ImageFolderData, transform=None):
        self.ImageFolderData = ImageFolderData
        self.transform = transform

    def __getitem__(self, idx):
        images, labels = self.ImageFolderData.imgs[idx]
        images = cv.imread(images, cv.IMREAD_LOAD_GDAL)
        images = cv.cvtColor(images, cv.COLOR_BGR2RGB)
        # images = Image.open(images).convert('RGB')
        images = Image.fromarray(images)
        if self.transform:
            images = self.transform(images)
        labels = torch.tensor(labels)
        return images, labels

    def __len__(self):
        return len(self.ImageFolderData.imgs)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(25, 2),
        )

    def forward(self, x):
        # output = self.convnet(x)
        # # print(output.size())
        # output = output.view(output.size()[0], -1)
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, colors=None):
    plt.figure(figsize=(10,10))
    for i in range(25):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.xticks([])
    plt.yticks([])
    # plt.savefig("images/cosine_image.png")
    plt.show()


def extract_embeddings(dataloader, net, model):
    with torch.no_grad():
        model.eval()
        net.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            embeddings[k:k+len(images)] = model.get_embedding(net(images)).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def checkpoint(save_dir, acc, epoch, model, name):
    # Save checkpoint.
    state = {
        'backbone' : model,
        'features':model.features,
        'acc' : acc,
        'epoch' : epoch
    }
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, save_dir + name)

if __name__ == '__main__':
    # from backbone import BaselineModel
    # img_size=100
    # batch_size = 64
    # transform_train = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])
    #
    # root = 'E:/CodeforFuture/Data/imagedata/'
    #
    # sar_data = datasets.ImageFolder(root)
    # train_data = SARDataSet2(sar_data, transform=transform_train)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    #
    # val_data = SARDataSet2(sar_data, transform=transform_test)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    #
    # ckt = torch.load('runs/baseline/exp12/BaselineModel.pt', map_location=device)
    #
    # net = ckt['backbone']
    # # net.load_state_dict(torch.load('../backbone/weight/Ma.mdl'))
    # backbone = EmbeddingNet()
    # train_embedding_baseline, train_labels_baseline = extract_embeddings(train_loader, net, backbone)
    # plot_embeddings(train_embedding_baseline, train_labels_baseline, colors=colors)

    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    # plt.style.use(['science', 'no-latex', 'ieee'])
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']

    # sar18 = cv.imread('E:/CodeforFuture/Data/imagedata/C18/C18.003.tif')
    sar = cv.imread('E:/CodeforFuture/Data/imagedata/C05/C05.006.tif')

    src_sar = cv.imread('C:/Users/wyz93/Pictures/ship_sar.jpg')
    src = cv.imread('C:/Users/wyz93/Pictures/oilcan.jpg')

    src_sar = cv.cvtColor(src_sar, cv.COLOR_BGR2GRAY)
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    sar = cv.cvtColor(sar, cv.COLOR_BGR2GRAY)

    dpi = 200

    fig = plt.figure(dpi=dpi)  # 定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')

    xx = np.arange(0, 200, 1)
    yy = np.arange(0, 200, 1)
    X, Y = np.meshgrid(xx, yy)
    ax3.view_init(32, 20)

    plt.tight_layout(pad=0)

    ax3.plot_surface(X, Y, sar, rstride = 1, cstride = 1, cmap='rainbow')
    # plt.savefig('E:/CodeforFuture/毕业进程/大论文/Image/sar.png')
    plt.show()


    # ax3.plot_surface(X, Y, src_sar, rstride = 1, cstride = 1, cmap='rainbow')
    # plt.savefig('E:/CodeforFuture/毕业进程/大论文/Image/ship_sar.png')
    # plt.show()











