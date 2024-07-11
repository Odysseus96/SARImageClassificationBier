import os
import math
import timeit
import argparse
import platform
import csv

import torch.optim.lr_scheduler
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from loss.Mahalanobis import MahalanobisCrossEntropyLoss
from loss.D_CNNLoss import DCNNLoss
from loss.triplet_fisher import TripletFisherLoss
from loss.triplet_loss import TripletLoss


from backbone.lenet5 import LeNet5
from backbone.dsnet import DSNet

from random_identity_sampler import RandomIdentitySampler
from dataset import *
from model import *
from utils import *
from collate_batch import collate_fn

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_learning_rate(epoch, epochs, optimizer, gamma=0.1):
    if epoch / epochs == 0.5:
        for param in optimizer.param_groups:
            param['lr'] *= gamma
    if epoch / epochs == 0.75:
        for param in optimizer.param_groups:
            param['lr'] *= gamma

def train(epoch, epochs, train_loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)

    train_steps = len(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data

        images, labels = images.to(device), labels.to(device)
        # images, labels_a, labels_b, lam = mixup_data(images, labels, args.mix_alpha, use_cuda=True)
        optimizer.zero_grad()
        # images, labels_a, labels_b = Variable(images), Variable(labels_a), Variable(labels_b)
        outputs = model(images)

        # loss_func = mixup_criterion(labels_a, labels_b, lam)
        # loss = loss_func(criterion, outputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        # adjust_learning_rate(epoch, epochs, optimizer)
        scheduler.step()

        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
    return running_loss / train_steps


def test(epoch, save_dir):
    best_acc = 0.0
    model.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_images, val_labels in val_bar:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            output = model(val_images)
            predict_y = torch.max(output, dim=1)[1]
            acc += torch.eq(predict_y, val_labels).sum().item()
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

    val_accurate = acc / val_num

    if val_accurate > best_acc:
        best_acc = val_accurate
        checkpoints(save_dir, best_acc, epoch, model, name=model.__class__.__name__+'.pt')

    return val_accurate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SAR Image Training')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--backbone', type=str, default='original')
    parser.add_argument('--max_iter', type=int, default=4000)
    parser.add_argument('--num_class', type=int, default=25)
    parser.add_argument('--save_dir', type=str, default='runs/trfisher/')
    parser.add_argument('--data_path', type=str, default='/root/wyz/SAR_data/')
    parser.add_argument('--lr_mode', type=str, default='cycle')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--lamda', '-la', type=float, default=5e-4)

    args = parser.parse_args()

    start_epoch = 0
    epochs = args.epochs
    batch_size = args.batch_size
    image_size = args.image_size

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 更新保存文件名：exp*
    file_idx = []
    for file in os.listdir(save_dir):
        if not 'exp' in file:
            continue
        file_idx.append(int(file.strip('exp')))

    if len(file_idx) == 0:
        save_dir += 'exp' + str(1) + '/'
    else:
        last_dir = save_dir + 'exp' + str(max(file_idx)) + '/'
        save_dir += 'exp' + str(max(file_idx) + 1) + '/'

    if platform.system() == "Windows":
        root = 'E:/CodeforFuture/Data/SAR_data/'
    else:
        root = args.data_path

    # data
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.467, 0.467, 0.467], [0.237, 0.237, 0.237])
        ]),
        "val": transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.467, 0.467, 0.467], [0.237, 0.237, 0.237])
        ])
    }

    train_data = BaseDataSet(root + 'train.txt', transforms=data_transforms['train'])
    val_data = BaseDataSet(root + 'val.txt', transforms=data_transforms['val'])

    train_num = len(train_data)
    val_num = len(val_data)

    train_sampler = RandomIdentitySampler(dataset=train_data,
                                          batch_size=batch_size,
                                          num_instances=5,
                                          max_iters=args.max_iter)

    train_loader = DataLoader(train_data,
                              collate_fn=collate_fn,
                              batch_sampler=train_sampler,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)
    # path = 'E:/CodeforFuture/Data/MSTAR_SOC/'
    #
    # transform_train = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # ])
    # train_soc = datasets.ImageFolder(path + 'train/')
    # test_soc = datasets.ImageFolder(path + 'test/')
    # print(len(train_soc))
    # print(len(test_soc))
    #
    # train_data = SARClassificationDataset(train_soc, transform_train)
    # val_data = SARClassificationDataset(test_soc, transform_test, mode='test')
    #
    # train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_data, batch_size, shuffle=True, num_workers=8)
    #
    # train_num = len(train_data)
    # val_num = len(val_data)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(last_dir)
        assert os.path.isdir(last_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(last_dir + 'BaselineModel.pt')
        model = checkpoint['backbone']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
    else:
        # model = DSNet().to(device)
        model = TripletFisherModel().to(device)
        # model = BaselineModel(num_classes=args.num_class, backbone=args.backbone).to(device)
        # model = LeNet5().to(device)
        # backbone = LeNetBN().to(device)
        # model = models.resnet18(pretrained=True).to(device)
        # model.classifier._modules['6'] = nn.Linear(4096, 25).to(device)
        # model = models.alexnet(pretrained=True).to(device)
        # model.classifier._modules['6'] = nn.Linear(4096, 25).to(device)
        # model = models.vgg16_bn(pretrained=True).to(device)
        # model.classifier._modules['6'] = nn.Linear(4096, 25).to(device)
    # # for param in backbone.features.parameters():
    # #     param.requires_grad = False

    # criterion = nn.CrossEntropyLoss()
    # criterion = MahalanobisCrossEntropyLoss(args.lamda)
    # criterion = DCNNLoss()
    criterion = TripletFisherLoss()
    # criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-4)

    if args.lr_mode == 'cosine':
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, gamma=0.9,
                                                         cycle_momentum=False,
                                                         step_size_up=1400, step_size_down=1400,
                                                      mode="triangular2")

    logname = save_dir + model.__class__.__name__ + '.csv'

    start = timeit.default_timer()

    for epoch in range(start_epoch, epochs):
        # adjust_learning_rate(optimizer, epoch, base_lr=args.base_lr)
        # train_loss = train(epoch, epochs, train_loader, criterion, optimizer, scheduler)
        test_acc = test(epoch, save_dir)
        # print("[epoch {}] train_loss {:.6f} val_accuracy: {:.4f}".
        #       format(epoch+1, train_loss, test_acc))
        #
        # with open(logname, 'a') as logfile:
        #     logwriter = csv.writer(logfile, delimiter=',')
        #     logwriter.writerow([epoch+1, train_loss, test_acc])


    print("Training Finished")

    end = timeit.default_timer()

    plot_training(logname, save_dir)
    time_record(end - start, save_dir)

    print("training span: {:.4f} s".format(end-start))