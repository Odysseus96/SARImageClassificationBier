import torch
from tqdm import tqdm

from utils import *

from evaluate import evaluate

def train(train_loader, val_loader, model, criterion, optimizers,
          schedulers, device, val_num, num_learners, save_dir, args):

    # optimizer, optimizerL = optimizers
    # sch_features, sch_embeddings = schedulers

    best_acc = 0.0
    train_steps = len(train_loader)
    logger_title = "epoch\ttrain_loss\tval_accuracy"

    # 写入文件result.txt的标题：epoch train_loss val_acc...
    with open(save_dir + 'result.txt', 'a') as f:
        f.write(logger_title + '\n')
        f.write(args.embed_sizes + '\n')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)

        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizers.zero_grad()

            embed_fvecs, outputs = model(images)

            loss = criterion(embed_fvecs, outputs, labels)


            loss.backward()
            optimizers.step()
            # optimizerL.step()
            schedulers.step()

            running_loss += loss.item()

            train_bar.desc = "Train epoch[{}/{}] loss:{:.4f}".\
                format(epoch + 1, args.epochs, loss)

        acc, boost_acc = evaluate(val_loader, model, device, epoch,
                       num_learners=num_learners, args=args)

        val_accurate = acc / val_num
        boost_acc = [bst_acc / val_num for bst_acc in boost_acc]

        desc = "[epoch {}] loss: {:.6f} val_acc: {:.4f}".format(epoch + 1, running_loss/train_steps, val_accurate)
        logger = "{}\t\t{:.4f}\t\t{:.4f}".format(epoch + 1, running_loss/train_steps,val_accurate)

        # 是否展示每个booster的情况
        if args.booster_show:
            for idx in range(len(boost_acc)):
                desc += " boost{}_acc: {:.4f}".format(idx, boost_acc[idx])
                logger_title += "\t\tboosts_acc"
                logger += "\t\t{:.4f}".format(boost_acc[idx])

        # 打印一个epoch的结果
        print(desc)

        # 写入文件result.txt的内容，模型运行情况
        with open(save_dir + 'result.txt', 'a') as f:
            f.write(logger + '\n')

        if val_accurate > best_acc:
            best_acc = val_accurate
            checkpoints(save_dir, best_acc, epoch, model, name=model.__class__.__name__+'.ckt')

    print("Best accuracy: {:.4f}".format(best_acc))







