import torch
import time
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset import *
from evaluate import weight_vote_ensemble
from backbone.lenet5 import LeNet5
from model import *

SAR_classes = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09',
                 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
                 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25']

def get_all_preds(model, loader, dataset, is_baseline, name, root=None):
    corrects = 0
    class_c = [0.0] * len(SAR_classes)
    class_total = [0.0] * len(SAR_classes)

    with torch.no_grad():
        test_bar = tqdm(loader)
        for images, labels in test_bar:
            val_images, labels = images.to(device), labels.to(device)
            if is_baseline:
                preds = model(val_images)
            else:
                _, outputs = model(val_images)
                # cls_vecs = torch.chunk(outputs, chunks=3, dim=1)
                # preds = weight_vote_ensemble(cls_vecs)
            #
            predict_y = torch.max(outputs, dim=1)[1]
            corrects += torch.eq(predict_y, labels).sum().item()

            c = (predict_y == labels.data).squeeze()
            for i in range(80):
                class_c[labels[i]] += c[i].item()
                class_total[labels[i]] += 1
            test_bar.desc = 'Testing'

    accuracy = corrects / len(dataset)
    # f1 = f1 / len(loader)
    # precision = precision / len(loader)
    # recall = recall / len(loader)

    # with open(root + 'pbier-4_acc.txt', 'a') as file:
    #     file.write("Total Accuracy: {:.2f}%".format(accuracy * 100) + '\n')
    print("Total Accuracy: {:.2f}%".format(accuracy * 100))
    # print("Precision: {:4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(f1, precision, recall))

    for i in range(len(SAR_classes)):
        print('Accuracy of {} : {:.4f}'.format(SAR_classes[i], 100 * class_c[i] / class_total[i]))
        with open(root + '.txt', 'a') as file:
            file.write('{}'.format(100 * class_c[i] / class_total[i]) + '\n')

    return accuracy

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transforms = {
        "val": transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
            transforms.Normalize([0.467, 0.467, 0.467], [0.237, 0.237, 0.237])
        ])
    }
    root = 'E:/CodeforFuture/Data/SAR_data/'

    val_data = BaseDataSet(root + 'val.txt', transforms=data_transforms['val'])

    batch_size = 80
    val_loader = torch.utils.data.DataLoader(val_data,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=8)
    name = 'p-bier'
    is_baseline = True if name == 'baseline' else False
    path = 'runs/trfisher/exp1/TripletFisherModel.pt'
    # model = LeNet5()
    checkpoint = torch.load(path, map_location=device)
    # model.load_state_dict(torch.load(path + 'SNCA_CE(MB).mdl', map_location=device))
    # print(checkpoint)
    model = checkpoint['backbone']
    model.to(device)
    #
    get_all_preds(model, val_loader, val_data, is_baseline, name, root=path)