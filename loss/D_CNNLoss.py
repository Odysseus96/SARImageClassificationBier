import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DCNNLoss(nn.Module):
    def __init__(self, lamda=0.05, tau=0.44):
        super(DCNNLoss, self).__init__()
        self.lamda = lamda
        self.tau = tau

    def _hinge_loss(self, inputs, labels):
        inputs = F.normalize(inputs)
        idx = 0
        loss_hinge = torch.tensor(0.0).to(device)
        l = -1.0
        while idx <= (inputs.size(0) - 1):
            if labels[idx] == labels[idx + 1]:
                l = 1.0
                loss_hinge += F.relu(
                    0.05 - l * (self.tau - torch.pow(F.pairwise_distance(inputs[idx], inputs[idx + 1]), 2)))
            else:
                loss_hinge += F.relu(
                    0.05 - l * (self.tau - torch.pow(F.pairwise_distance(inputs[idx], inputs[idx + 1]), 2)))
            idx += 2
        return loss_hinge.mean()

    def forward(self, inputs, labels):

        return F.cross_entropy(inputs, labels) + self.lamda * self._hinge_loss(inputs, labels) / 2


if __name__ == '__main__':
    x = torch.randn(64, 25)
    y = torch.randn(64, 25)
    label = torch.randint(0, 25, [64])
    print(label)
    loss = DCNNLoss()
    print(loss(x, label))
