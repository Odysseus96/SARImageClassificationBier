import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def similarity(inputs_, eps=1e-08):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t()) \
          / max(torch.norm(inputs_, p=2) * torch.norm(inputs_.T, p=2), eps)
    return sim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BinDevianceLoss(nn.Module):
    def __init__(self, alpha=20, margin=0.5):
        super(BinDevianceLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def _similarity(self, inputs_, eps=1e-08):
        sim = torch.matmul(inputs_, inputs_.t()) \
              / max(torch.norm(inputs_, p=2) * torch.norm(inputs_.T, p=2), eps)

        return sim

    def forward(self, inputs, targets, size_average=True):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = self._similarity(inputs)
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).to(device)
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) ^ pos_mask
        pos_mask = pos_mask ^ eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.reshape(len(pos_sim)//(num_instances-1), num_instances-1)
        neg_sim = neg_sim.reshape(
            len(neg_sim) // num_neg_instances, num_neg_instances)

        #  clear way to compute the loss first
        loss = list()

        for i, pos_pair in enumerate(pos_sim):
            # print(i)
            pos_pair = torch.sort(pos_pair)[0]
            neg_pair = torch.sort(neg_sim[i])[0]

            neg_pair = torch.masked_select(neg_pair, neg_pair > pos_pair[0] - 0.05)
            # pos_pair = pos_pair[1:]

            neg_pair = torch.sort(neg_pair)[0]

            pos_loss = torch.mean(torch.log(1 + torch.exp(-2*(pos_pair - self.margin))))
            neg_loss = 0.04*torch.mean(torch.log(1 + torch.exp(50*(neg_pair - self.margin))))
            loss.append(pos_loss + neg_loss)

        loss = torch.stack(loss)
        return loss.mean() if size_average else loss.sum()

if __name__ == '__main__':
    inputs = torch.randn(80, 32)
    sim = similarity(inputs)
    print(sim)
    print(sim.size())
    # sim_mat = F.cosine_similarity(inputs, inputs.t())
    # sim_mat = similarity(inputs)
    # print(sim_mat.size())
    # print(sim_mat)