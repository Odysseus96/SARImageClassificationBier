import os
import timeit
import math
import torch
import torch.nn as nn
import argparse
import platform

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from trainer import train
from loss.cls_loss import ClassifiyLoss
from random_identity_sampler import RandomIdentitySampler
from dataset import *
from model import *
from evaluate import evaluate
from collate_batch import collate_fn

from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='runs/a-bier/')
    parser.add_argument('--data_path', type=str, default='/root/wyz/SAR_data/')
    parser.add_argument('--weight_path', type=str, default='runs/a-bier/')
    parser.add_argument('--embed_sizes', type=str, default='32,96')
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--max_iter', type=int, default=4000)
    parser.add_argument('--num_learners', type=int, default=2)
    parser.add_argument('--shrinkage_rate', type=float, default=0.06)
    parser.add_argument('--adv_hidden_dim', type=int, default=128)
    parser.add_argument('--sim_mode', type=str, default='pearson')
    parser.add_argument('--lambda_metric', type=float, default=5e-4)
    parser.add_argument('--lambda_div', type=float, default=1e-3)
    parser.add_argument('--metric_mode', type=str, default='triplet')
    parser.add_argument('--metric_loss', type=bool, default=False)
    parser.add_argument('--aux_loss', type=bool, default=False)
    parser.add_argument('--num_class', type=int, default=25)
    parser.add_argument('--backbone', type=str, default='original')
    parser.add_argument('--booster_show', type=bool, default=True)
    parser.add_argument('--freeze', '-fz', action='store_true', help='freeze the network')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    args = parser.parse_args()

    save_dir = args.save_dir
    image_size = args.image_size
    batch_size = args.batch_size
    start_epoch = 0

    if platform.system() == "Windows":
        root = 'E:/CodeforFuture/Data/SAR_data/'
    else:
        root = args.data_path

    embed_sizes = [int(x) for x in args.embed_sizes.split(',')]

    # 更新保存文件名：exp*
    file_idx = []
    for file in os.listdir(save_dir):
        file_idx.append(int(file.strip('exp')))

    if len(file_idx) == 0:
        save_dir += 'exp' + str(1) + '/'
    else:
        last_dir = save_dir + 'exp' + str(max(file_idx)) + '/'
        save_dir += 'exp' + str(max(file_idx) + 1) + '/'


    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.467, 0.467, 0.467], [0.237, 0.237, 0.237])
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.467, 0.467, 0.467], [0.237, 0.237, 0.237])
        ])
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    model = Model(embed_size=embed_sizes, num_classes=args.num_class, backbone=args.backbone).to(device)


    criterion = ClassifiyLoss(embed_dim=embed_sizes,
                              args=args)


    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(last_dir+'Model.ckt', map_location=device)
        model = checkpoint['backbone']


    if args.freeze:
        for param in model.features.parameters():
            param.requires_grad = False

        optimizers = torch.optim.Adam(model.embeddings.parameters(), lr=0.001, weight_decay=1e-4)
        optimizers.add_param_group({"params": model.classifiers.parameters(),
                                    "lr": 0.001, "weight_decay": 1e-4})

    else:
        optimizers = torch.optim.Adam(model.features.parameters(), lr=0.001, weight_decay=1e-4)
        optimizers.add_param_group({"params": model.embeddings.parameters(),
                                    "lr": 0.01, "weight_decay": 1e-4})
        optimizers.add_param_group({"params": model.classifiers.parameters(),
                                    "lr": 0.001, "weight_decay": 1e-4})

    if args.aux_loss:
        # 添加对抗损失
        optimizers.add_param_group({"params": criterion.adversarial_loss.regressors.parameters(),
                                    "lr": 0.01, "weight_decay": 1e-4})

    schedulers = lr_scheduler.CyclicLR(optimizers, base_lr=0.001, max_lr=0.01, gamma=0.9,
                                       cycle_momentum=False,
                                       step_size_up=1400, step_size_down=1400,
                                       mode="triangular2")
    # if not args.beta_constant:
    #     optimizers.add_param_group({"params": criterion.metric_loss.beta})


    # schedulers = lr_scheduler.CyclicLR(optimizers, base_lr=0.0001, max_lr=5e-4, gamma=0.9,
    #                                    cycle_momentum=False,
    #                                    step_size_up=1400, step_size_down=14000,
    #                                    mode="triangular2")

    start = timeit.default_timer()

    train(train_loader, val_loader, model, criterion, optimizers,
          schedulers, device, val_num, len(embed_sizes), save_dir, args)

    # for epoch in range(args.epochs):
    #     evaluate(val_loader, backbone, device, epoch, args.epochs,
    #              num_learners=len(embed_sizes))

    end = timeit.default_timer()

    # plot_training(logname, save_dir)
    time_record(end-start, save_dir)

    print("Finished Training")




