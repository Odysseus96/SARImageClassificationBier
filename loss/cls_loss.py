import torch
import torch.nn as nn
import torch.nn.functional as F

from joblib import Parallel, delayed


from utils import *
from loss.advLoss import AdversarialLoss
from loss.margin_loss import MarginLoss
from loss.triplet_loss import TripletLoss
from loss.binomial_deviance_loss import BinDevianceLoss
from loss.MSLoss import MultiSimilarityLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _parallel_compute_pseudo_residual(output, target, learns_idx, shrinkage_rate, n_classes):
    """
        Compute pseudo residuals in soft gradient boosting for each base estimator
        in a parallel fashion.
    """
    accumulated_output = torch.zeros_like(output[0], device=output[0].device)
    for i in range(learns_idx):
        eta = 2 / (i+1+1)
        accumulated_output = eta * output[i] + (1-eta)  * accumulated_output
    residual = pseudo_residual_classification(target, accumulated_output, n_classes)

    return residual

metric_func = {
    'bin' : BinDevianceLoss,
    'MS' : MultiSimilarityLoss,
    'triplet' : TripletLoss,
    'Margin' : MarginLoss
}

class ClassifiyLoss(nn.Module):
    def __init__(self, embed_dim, args):
        super(ClassifiyLoss, self).__init__()
        self.embed_dim = embed_dim
        self.aux_loss = args.aux_loss
        self.lambda_div = args.lambda_div
        self.lambda_metric = args.lambda_metric
        self.num_class = args.num_class
        self.n_jobs = None
        self.metric_loss = args.metric_loss

        self.shrinkage_rate = args.shrinkage_rate
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_metric = metric_func[args.metric_mode]()

        if self.aux_loss:
            self.hidden_dim = args.adv_hidden_dim
            self.adversarial_loss = AdversarialLoss(self.embed_dim, self.hidden_dim, args)


    def forward(self, embed_fvecs, inputs, target):

        outputs = torch.chunk(inputs, chunks=len(self.embed_dim), dim=1)

        # Compute pseudo residuals in parallel
        rets = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_pseudo_residual)(
                outputs, target, i, self.shrinkage_rate, self.num_class
            ) for i in range(len(self.embed_dim))
        )
        # Compute loss
        loss = torch.tensor(0.0).to(device)
        for idx, output in enumerate(outputs):
            loss += torch.mean(self.loss_cls(outputs[idx], target) * rets[idx])

        if self.aux_loss:
            adv_loss = self.adversarial_loss(embed_fvecs)
            loss = loss + self.lambda_div * adv_loss

        return loss




if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    from model import *

    # batch_size = 32
    # data_transforms = {
    #     "train": transforms.Compose([
    #         transforms.RandomResizedCrop(32),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     "val": transforms.Compose([
    #         transforms.Resize(32),
    #         transforms.CenterCrop(32),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    # }
    # train_data = datasets.CIFAR10(root='data', train=True, transform=data_transforms['train'], download=False)
    # val_data = datasets.CIFAR10(root='data', train=False, transform=data_transforms['val'], download=False)
    #
    # train_num = len(train_data)
    # val_num = len(val_data)
    #
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    # backbone = Model().to(device)
    # criterion = ClassifiyLoss(nums=3,
    #                           aux_loss=False,
    #                           hidden_dim=128,
    #                           lambda_div=1e-3,
    #                           num_class=25)
    #
    #
    # images, labels  = next(iter(train_loader))
    # pred = backbone(images.to(device))
    # loss = criterion(pred, labels.to(device))
    # print(loss)
    # embed_dim = [25] * 3
    # directions = []
    # for i in range(len(embed_dim)):
    #     for j in range(i + 1, len(embed_dim)):
    #         directions.append("{}-{}".format(j, i))
    #
    # print(directions)









