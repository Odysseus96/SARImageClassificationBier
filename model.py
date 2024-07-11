import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from backbone import resnet, densenet, efficientnetv2
from efficientnet_pytorch import model as enet

INPUT_SIZE = {"original":128, "lenet":64, "resnet":512, "vgg":25088, "googlenet":1024, "densenet":1024}

class DWBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(DWBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=(3, 3),
                               stride=(stride, stride), padding=(1, 1), groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(1, 1),bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class Backbone(nn.Module):
    def __init__(self, num_classes=25):
        super(Backbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            DWBlock(32, 64),
            DWBlock(64, 128, stride=2),
            DWBlock(128, 128),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        return x

class LeNetBackbone(nn.Module):
    def __init__(self):
        super(LeNetBackbone, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16, 256),  # 100 * 100
            nn.Linear(256, 64),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x


class BaselineModel(nn.Module):
    def __init__(self, num_classes=25, backbone="original"):
        super(BaselineModel, self).__init__()

        if backbone == "original":
            self.features = Backbone()
            self.classifiers = nn.Sequential(
                nn.Dropout(0.2),
                # nn.Linear(64, num_classes)
                nn.Linear(128, num_classes)
            )
        elif backbone == 'lenet':
            self.features = LeNetBackbone()
            self.classifiers = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        elif backbone == "resnet":
            self.features = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(self.features.children())[:-1],
                                          nn.Flatten())
            self.classifiers = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, num_classes)
            )
        elif backbone == "efficientnet":
            # self.features = efficientnetv2.efficientnetv2_s(num_classes=num_classes)
            self.features = enet.EfficientNet.from_pretrained('efficientnet-b0')
            self.classifiers = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1000, num_classes)
            )
        else:
            self.features = models.densenet121(pretrained=True).features
            self.features.add_module('avg_pool2d', nn.AdaptiveAvgPool2d((1,1)))
            self.features.add_module('flatten', nn.Flatten())
            self.classifiers = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1024, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        return self.classifiers(x)


class TripletFisherModel(nn.Module):
    def __init__(self, embed_size=128, num_classes=25):
        super(TripletFisherModel, self).__init__()
        self.embed_size = embed_size
        self.num_classes= num_classes

        # self.features = Backbone()
        self.features = LeNetBackbone()
        self.embeddings = nn.Linear(128, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        embed = self.embeddings(x)
        x = self.classifier(embed)
        return embed, x


class Model(nn.Module):
    def __init__(self, embed_size, num_classes=25, backbone="original", embedding_size=128):
        super(Model, self).__init__()
        self.embed_sizes = embed_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        if backbone == "original":
            self.features = Backbone()
        elif backbone == 'lenet':
            self.features = LeNetBackbone()
        elif backbone == "resnet":
            self.features = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(self.features.children())[:-1],
                                          nn.Flatten())
        elif backbone == "vgg":
            # self.features = efficientnetv2.efficientnetv2_s(num_classes=num_classes)
            self.features = nn.Sequential(
                models.vgg16_bn(pretrained=True).features,
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
            )
        elif backbone == "googlenet":
            self.features = models.googlenet(pretrained=True)
            self.features = nn.Sequential(
                *list(self.features.children())[:-2],
                nn.Flatten()
            )
        else:
            self.features = models.densenet121(pretrained=True).features
            self.features.add_module('avg_pool2d', nn.AdaptiveAvgPool2d((1, 1)))
            self.features.add_module('flatten', nn.Flatten())

        self.embeddings = nn.ModuleList()
        self.classifiers = nn.ModuleList()

        for embed_size in self.embed_sizes:
            self.embeddings.append(
                nn.Linear(INPUT_SIZE[backbone], embed_size)
            )
            self.classifiers.append(
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(embed_size, self.num_classes)
                )
            )

    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)

        out_fvecs = []
        embed_fvecs = []

        for i in range(len(self.embed_sizes)):
            embed_fvecs.append(self.embeddings[i](x))
            out_fvecs.append(self.classifiers[i](self.embeddings[i](x)))

        return torch.cat(embed_fvecs, dim=1), torch.cat(out_fvecs, dim=1)


if __name__ == '__main__':
    x = torch.randn(80, 3, 128, 128)
    model = Backbone()
    out = model(x)
    print(out.size())
    # model = Model(backbone='vgg', embed_size=[32,32,32,32])
    # # print(model)
    # # model = LeNetBackbone()
    # print(model(x)[1].size())
    # # print(model(x).size())











