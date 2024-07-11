import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from dataset import *
from backbone import *
from evaluate import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, val_loader, val_num):
    model.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_images, val_labels in val_bar:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            output = model(val_images)
            predict_y = torch.max(output, dim=1)[1]
            acc += torch.eq(predict_y, val_labels).sum().item()
            val_bar.desc = "evaluating"

    val_accurate = acc / val_num

    return val_accurate


if __name__ == '__main__':
    root = 'F:/CodeforFuture/data/SAR_data/'

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.467, 0.467, 0.467], [0.237, 0.237, 0.237])
        ]),
        "val": transforms.Compose([
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.467, 0.467, 0.467], [0.237, 0.237, 0.237])
        ])
    }

    val_data = BaseDataSet(root + 'val.txt', transforms=data_transforms['val'])
    val_loader = DataLoader(val_data,
                            batch_size=1,
                            shuffle=True,
                            num_workers=8)

    val_num = len(val_data)

    # backbone = BaselineModel().to(device)
    # ckpt = torch.load('runs/baseline/exp9/BaselineModel.pt', map_location=device)
    # backbone = ckpt['backbone']
    #
    #
    # torch.save(backbone.features.state_dict(), 'runs/baseline/exp9/features.pt')

    # backbone = Model()
    ckpt = torch.load('runs/a-bier/exp9/Model.ckt', map_location=device)
    model = ckpt['backbone']
    acc, booster_acc = evaluate(val_loader, model, device, 0, 0, training=False)
    val_accurate = acc / val_num
    booster1_acc = booster_acc[0] / val_num
    booster2_acc = booster_acc[1] / val_num

    print(val_accurate)
    print(booster1_acc, booster2_acc)

