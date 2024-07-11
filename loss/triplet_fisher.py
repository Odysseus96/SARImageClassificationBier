import torch
import torch.nn as nn
import torch.nn.functional as F

from batchminer.semihard import SemiHard

class TripletFisherLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    def __init__(self, lamda1=0.6, lamda2=0.005, margin=0.2, eps=1e-9): # 画图时可设置为1.5
        super(TripletFisherLoss, self).__init__()
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.margin = margin
        self.eps = eps
        self.batchminer = SemiHard(self.margin)
        self.cls_loss = nn.CrossEntropyLoss()

    def triplet_distance(self, anchor, positive, negative):
        return F.relu(F.pairwise_distance(anchor, positive).pow(2).sum()
                      -F.pairwise_distance(anchor, negative).pow(2).sum()+self.margin)

    def fisher_regular(self, batch, sampled_triplets):
        dist_ap = []
        dist_an = []
        for triplet in sampled_triplets:
            anchor, positive, negative = batch[triplet[0], :], batch[triplet[1], :], batch[triplet[2], :]
            dist_ap.append(F.pairwise_distance(anchor, positive))
            dist_an.append(F.pairwise_distance(anchor, negative))

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        m1 = dist_ap.mean()
        m2 = dist_an.mean()
        sigma1 = torch.var(dist_ap, unbiased=True).mean()
        sigma2 = torch.var(dist_an, unbiased=True).mean()
        return (sigma1 + sigma2) / max((m1 - m2), self.eps)

    def forward(self, batch, inputs, labels):
        log_softmax_loss = self.cls_loss(inputs, labels)
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        batch = F.normalize(batch)
        sampled_triplets = self.batchminer(batch, labels)
        triplet_loss = torch.stack(
            [self.triplet_distance(batch[triplet[0], :], batch[triplet[1], :], batch[triplet[2], :]) for triplet in
             sampled_triplets])
        fisher_reg = self.fisher_regular(batch, sampled_triplets)
        loss = self.lamda1 * log_softmax_loss + (1 - self.lamda1) * triplet_loss.mean() + self.lamda2 * fisher_reg.mean()

        return loss
