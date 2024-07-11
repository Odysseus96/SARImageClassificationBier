import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from backbone import *

image_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

checkpoint = torch.load('runs/a-bier/exp23/Model.ckt', map_location=device)

features = checkpoint['features']

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = checkpoint['features']
    def forward(self, x):
        outputs = []
        for name, module in self.features.named_children():
            for idx, mdl in enumerate(module):
                x = mdl(x)
                if idx >= 3 and idx != len(module)-1:
                    outputs.append(x)
        return outputs

model = MyModel()

img = Image.open('F:/CodeforFuture/data/imagedata/C18/C18.001.tif').convert('RGB')
img = data_transforms['train'](img)
img = torch.unsqueeze(img, dim=0)

output = model(img)

for feature_map in output:
    print(feature_map.shape)
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    print(im.shape)
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(len(output)):
        ax = plt.subplot(1, 3, i + 1)
        # [H, W, C]
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()