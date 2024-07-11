import torch
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive

    # 画 Loss v.s. distance 曲线专用
    # def forward(self, distance, label):
    #     pos = (label) * torch.pow(distance, 2)
    #     neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
    #     loss_contrastive = torch.mean( pos + neg )
    #     return loss_contrastive

if __name__ == '__main__':
    # pass

    # 画出对比损失与样本距离的变化曲线
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.style.use(['science', 'no-latex', 'ieee'])
    #
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    config = {
        "font.family": 'serif',
        "font.size": 9,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    distance = torch.linspace(0, 3.5, 30)
    loss_func = ContrastiveLoss()
    loss_pos = []
    loss_neg = []
    for d in distance:
        l_pos = loss_func(d, 1)
        loss_pos.append(np.array(l_pos))

    for d in distance:
        l_neg = loss_func(d, 0)
        loss_neg.append(np.array(l_neg))
        if l_neg == 0:
            break

    distance = np.array(distance)
    plt.plot(distance, loss_pos, 'r', label='正样本对', linewidth=2.5)
    plt.plot(distance[:len(loss_neg)], loss_neg, 'g--', label='负样本对', linewidth=2.5)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.text(1.5, 0.1, 'Margin', fontdict={'family':'Times New Roman', 'size':9})
    plt.axis([0., 2.0, 0., 3.5])
    plt.ylabel('损失')
    plt.xlabel(r'$||f(\mathbf{{x}}_{i})-f(\mathbf{{x}}_{j})||_{2}$', size=10)
    plt.yticks(fontproperties='Times New Roman')
    plt.legend()
    plt.tight_layout()
    plt.savefig('E:/CodeforFuture/contrast.png')
    plt.show()
