import numpy as np
import torch
import timeit
import torch.nn.functional as F

from tqdm import tqdm
from functools import reduce

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import *
from backbone import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _alpha_value(num_learners):
    alphas = []
    eta = [2.0 / (i + 1 + 1) for i in range(num_learners)]
    for idx in range(num_learners):
        if idx == num_learners - 1:
            alphas.append(eta[idx])
        else:
            alphas.append(eta[idx] * reduce(lambda x, y: x * y, [1 - e for e in eta[idx + 1:]]))
    return alphas

def average_ensemble(pred_vecs):
    preds = torch.tensor(0.0)
    for vec in pred_vecs:
        pred = F.softmax(vec, dim=1)
        preds = preds + pred
    return preds / len(pred_vecs)

def weight_vote_ensemble(pred_vecs):
    alphas = _alpha_value(len(pred_vecs))
    preds = torch.tensor(0.0)
    for idx, vec in enumerate(pred_vecs):
        pred = F.softmax(vec, dim=1)
        preds = preds + alphas[idx] * pred

    return preds


def evaluate(val_loader, model, device, epoch, num_learners, args, training=True):
    model.eval()
    acc = 0.0
    booster_acc = [0.0] * num_learners

    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_images, val_labels in val_bar:

            val_images, val_labels = val_images.to(device), val_labels.to(device)

            _, outputs = model(val_images)

            cls_vecs = torch.chunk(outputs, chunks=num_learners, dim=1)

            preds = weight_vote_ensemble(cls_vecs)
            # preds = average_ensemble(cls_vecs)

            for idx, cls_vec in enumerate(cls_vecs):

                booster_predict_y = torch.max(cls_vec, dim=1)[1]
                booster_acc[idx] += torch.eq(booster_predict_y, val_labels).sum().item()

            predict_y = torch.max(preds, dim=1)[1]
            acc += torch.eq(predict_y, val_labels).sum().item()

            if training:
                val_bar.desc = "Valid epoch[{}/{}]".format(epoch + 1, args.epochs)
            else:
                val_bar.desc = "Testing".format(epoch + 1, args.epochs)

    return acc, booster_acc

if __name__ == '__main__':
    alphas = _alpha_value(num_learners=3)

    print(alphas)
    print(sum(alphas))








