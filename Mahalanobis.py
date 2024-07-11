import torch
import torch.nn as nn
import numpy as np
import os
from scipy.spatial.distance import mahalanobis
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class MahalanobisLayer(nn.Module):

    def __init__(self, dim, decay = 0.1):
        super(MahalanobisLayer, self).__init__()
        self.register_buffer('S', torch.eye(dim))
        self.register_buffer('S_inv', torch.eye(dim))
        self.decay = decay

    def forward(self, x, x_fit):
        """
        Calculates the squared Mahalanobis distance between x and x_fit
        """
        self.cov = nn.Parameter(torch.ones_like(x))
        print(self.cov.size())
        print(torch.dot(self.cov, (x-x_fit)))
        m = mahalanobis(x, x_fit, self.cov.detach().numpy())
        return m

    def _mahalanobis(self, x, x_fit, cov):
        delta = x - x_fit
        m = torch.dot(delta, torch.dot(cov, delta).unsqueeze(-1))
        return m

    # def update(self, X, X_fit):
    #     delta = X - X_fit
    #     self.S = (1 - self.decay) * self.S + self.decay * self.cov(delta)
    #     self.S_inv = torch.pinverse(self.S)

if __name__ == '__main__':
    torch.manual_seed(666)
    x = torch.randn(32)
    y = torch.randn(32)

    delta = x - y
    cov = np.linalg.pinv(np.cov(x.numpy(), y.numpy()))
    cov = torch.from_numpy(cov).to(torch.float32).mean()
    dis = delta * cov * delta
    print(dis)
    # distance = MahalanobisLayer(dim=32)
    #
    # d1 = distance(x, y)
    # print(d1)
    # print(distance(x, y)[1])
