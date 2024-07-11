import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from batchminer.semihard import SemiHard


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    def __init__(self, embed_size=128, margin=1.0): # 画图时可设置为1.5
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.batchminer = SemiHard(self.margin)

    def triplet_distance(self, anchor, positive, negative):
        return F.relu(F.pairwise_distance(anchor, positive).pow(2).sum()
                      -F.pairwise_distance(anchor, negative).pow(2).sum()+self.margin)

    def forward(self, batch, inputs, labels, size_average=True):
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        sampled_triplets = self.batchminer(batch, labels)
        loss = torch.stack(
            [self.triplet_distance(batch[triplet[0], :], batch[triplet[1], :], batch[triplet[2], :]) for triplet in
             sampled_triplets])
        return loss.mean() if size_average else loss.sum()


    # def forward(self, dist_pos, neg_pos, size_average=True):
    #     distance_positive = dist_pos ** 2  # .pow(.5)
    #     distance_negative = neg_pos ** 2  # .pow(.5)
    #     losses = F.relu(distance_positive - distance_negative + self.margin)
    #     return losses.mean() if size_average else losses.sum()

if __name__ == '__main__':
    pass
    # 画出对比损失与样本距离的变化曲线
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    from matplotlib import rcParams

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    plt.style.use(['science', 'no-latex', 'ieee'])

    config = {
        "font.family": 'serif',
        "font.size": 9,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    distance = torch.linspace(0, 3.5, 30)
    loss_func = TripletLoss()
    loss_pos = []
    loss_neg = []
    for d in distance:
        l_pos = loss_func(d, distance[0]-loss_func.margin)
        loss_pos.append(np.array(l_pos))

    for d in distance:
        l_neg = loss_func(distance[0]+loss_func.margin, d)
        loss_neg.append(np.array(l_neg))

    distance = np.array(distance)
    plt.plot(distance, loss_pos, 'r', label='正样本对', linewidth=2.5)
    plt.plot(distance, loss_neg, 'g--', label='负样本对', linewidth=2.5)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.text(distance[5]-0.2, 0.15, r'$D_{an}-\alpha$')
    plt.text(distance[16], 0.15, r'$D_{ap}+\alpha$')
    plt.axis([0., 2.5, 0., 5.0])
    plt.ylabel('损失')
    # plt.xlabel(r'$\mathregular{||f({{x}_{i}})-f({{x}_{j}})||_{2}}$', fontdict={'family':'Times New Roman',
    #                                                                            'size':10
    #                                                                            })
    plt.xlabel(r'$||f(\mathbf{{x}}_{i})-f(\mathbf{{x}}_{j})||_{2}$', size=10)

    plt.yticks(fontproperties='Times New Roman')
    plt.legend(prop={'size': 9}, loc='upper left')
    plt.tight_layout()
    plt.savefig('E:/CodeforFuture/triplet.png')
    plt.show()