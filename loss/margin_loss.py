import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DistanceWeightedSampling(object):
    """
    """
    def __init__(self):
        super(DistanceWeightedSampling, self).__init__()
        self.cutoff = 0.5
        self.upper_cutoff = 1.4

    def sample(self, batch, labels):

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]
        distances = self.p_dist(batch.detach()).clamp(min=self.cutoff)

        positives, negatives = [], []

        for i in range(bs):
            pos = labels == labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            # sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            # sample negatives by distance
            negatives.append(np.random.choice(bs, p=q_d_inv))

        sampled_triplets = [[a, p, n] for a, p, n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    @staticmethod
    def p_dist(A, eps=1e-4):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        return res.clamp(min=eps).sqrt()

    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        bs, dim = len(dist), batch.shape[-1]
        # negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2)
                       * torch.log(1.0 - 0.25 * (dist.pow(2))))
        # set sampling probabilities of positives to zero
        log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

        q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))  # - max(log) for stability
        # set sampling probabilities of positives to zero
        q_d_inv[np.where(labels == anchor_label)[0]] = 0

        # NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist > self.upper_cutoff)[0]] = 0

        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()


class MarginLoss(nn.Module):
    """Margin based loss with DistanceWeightedSampling
    """
    def __init__(self):
        super(MarginLoss, self).__init__()
        self.beta_val = 1.25
        self.margin = 0.2
        self.nu = 0.0
        self.n_classes = 25
        self.beta_constant = True
        if self.beta_constant:
            self.beta = self.beta_val
        else:
            self.beta = torch.nn.Parameter(torch.ones(self.n_classes)*self.beta_val)
        self.sampler = DistanceWeightedSampling()

    # def forward(self, batch, labels):
    #     if isinstance(labels, torch.Tensor):
    #         labels = labels.detach().cpu().numpy()
    #     sampled_triplets = self.sampler.sample(batch, labels)
    #
    #     # compute distances between anchor-positive and anchor-negative.
    #     d_ap, d_an = [], []
    #     for triplet in sampled_triplets:
    #         train_triplet = {'Anchor': batch[triplet[0], :],
    #                          'Positive': batch[triplet[1], :], 'Negative': batch[triplet[2]]}
    #         pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
    #         neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)
    #
    #         d_ap.append(pos_dist)
    #         d_an.append(neg_dist)
    #     d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)
    #
    #     # group betas together by anchor class in sampled triplets (as each beta belongs to one class).
    #     if self.beta_constant:
    #         beta = self.beta
    #     else:
    #         beta = torch.stack([self.beta[labels[triplet[0]]] for
    #                             triplet in sampled_triplets]).to(device)
    #     # compute actual margin positive and margin negative loss
    #     pos_loss = F.relu(d_ap-beta+self.margin)
    #     neg_loss = F.relu(beta-d_an+self.margin)
    #
    #     # compute normalization constant
    #     pair_count = torch.sum((pos_loss > 0.)+(neg_loss > 0.)).to(device)
    #     # actual Margin Loss
    #     loss = torch.sum(pos_loss+neg_loss) if pair_count == 0. else torch.sum(pos_loss+neg_loss)/pair_count
    #
    #     # (Optional) Add regularization penalty on betas.
    #     # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)
    #     return loss
    def forward(self, dst, label):

        # group betas together by anchor class in sampled triplets (as each beta belongs to one class).
        # beta = self.beta
        # compute actual margin positive and margin negative loss
        loss = F.relu(label * (dst - self.beta) + self.margin)

        # compute normalization constant
        # pair_count = torch.sum((loss > 0.))
        # # actual Margin Loss
        # loss = torch.sum(pos_loss+neg_loss) if pair_count == 0. else torch.sum(pos_loss+neg_loss)/pair_count

        # (Optional) Add regularization penalty on betas.
        # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)
        return loss

if __name__ == '__main__':
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

    distance = torch.linspace(0, 2.5, 100)
    loss_func = MarginLoss()
    loss_pos = []
    loss_neg = []
    for d in distance:
        l_pos = loss_func(d, 1)
        loss_pos.append(np.array(l_pos))

    for d in distance:
        l_neg = loss_func(d, -1)
        loss_neg.append(np.array(l_neg))

    distance = np.array(distance)

    plt.plot(distance, loss_pos, 'r', label='正样本对', linewidth=2.5)
    plt.plot(distance, loss_neg, 'g--', label='负样本对', linewidth=2.5)
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.legend(prop={'size': 9}, loc='upper center')
    plt.xticks([])
    plt.yticks([])
    plt.text(loss_func.beta - loss_func.margin-0.35, 0.05, r'$\beta - \alpha$')
    plt.text(loss_func.beta + loss_func.margin+0.03, 0.05, r'$\beta + \alpha$')
    plt.text(loss_func.beta-0.05, 0.025, r'$\beta$')
    plt.ylim([0., 1.5])
    plt.xlim([0, 2.5])
    plt.ylabel('损失', size=9)
    plt.xlabel(r'$||f(\mathbf{{x}}_{i})-f(\mathbf{{x}}_{j})||_{2}$', size=10)
    plt.tight_layout()
    plt.savefig('E:/CodeforFuture/margin.png')
    plt.show()