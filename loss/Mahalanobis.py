import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MahalanobisCrossEntropyLoss(nn.Module):
    def __init__(self, lamda=0.01):
        super(MahalanobisCrossEntropyLoss, self).__init__()
        self.lamda = lamda
        self.M = torch.eye(25, requires_grad=True).to(device)

    def metric_learning(self, inputs, labels):
        distance_p = torch.tensor(0.0)
        distance_n = torch.tensor(0.0)
        idx = 0
        while idx <= (inputs.size(0)-1):
            if labels[idx] == labels[idx + 1]:
                d_p = torch.abs(inputs[idx] - inputs[idx + 1])
                distance_p = distance_p + torch.norm((self.M.t() * torch.unsqueeze(d_p, -1)), 2)
            else:
                d_n = torch.abs(inputs[idx] - inputs[idx + 1])
                distance_n = distance_p + torch.norm((self.M.t() * torch.unsqueeze(d_n, -1)), 2)
            idx += 2
        metric_loss = F.relu(1. + distance_p - distance_n)
        return metric_loss

    def forward(self, inputs, labels):
        metric_loss = self.metric_learning(inputs, labels)
        cross_entropy_loss = F.cross_entropy(inputs, labels)
        loss = cross_entropy_loss + self.lamda * metric_loss / 2
        return loss