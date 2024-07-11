import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_corr(fake_Y, Y):  # 计算两个向量pearson相关系数
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr

#
# def _get_ranks(x: torch.Tensor) -> torch.Tensor:
#     tmp = x.argsort()
#     ranks = torch.zeros_like(tmp)
#     ranks[tmp] = torch.arange(len(x))
#     return ranks
#
#
# def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
#     """计算两个向量Spearman相关系数
#     Compute correlation between 2 1-D vectors
#     Args:
#         x: Shape (N, )
#         y: Shape (N, )
#     """
#     x_rank = _get_ranks(x)
#     y_rank = _get_ranks(y)
#
#     n = x.size(0)
#     upper = 6 * torch.sum((x_rank - y_rank).pow(2))
#     down = n * (n ** 2 - 1.0)
#     return 1.0 - (upper / down)

def spearman_correlation(att, grad_att):
    """
    Function that measures Spearman’s correlation coefficient between target logits and output logits:
    att: [n, m]
    grad_att: [n, m]
    """
    def _rank_correlation_(att_map, att_gd):
        n = torch.tensor(att_map.shape[1])
        upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1)

    att = att.sort(dim=1)[1]
    grad_att = grad_att.sort(dim=1)[1]
    correlation = _rank_correlation_(att.float(), grad_att.float())
    return correlation

def jaccard_index(x, y, smooth=100):
    inter = (x * y).abs().sum(dim=-1)
    sum_ = torch.sum(x.abs() + y.abs(), dim=-1)
    jac = (inter + smooth) / (sum_ - inter + smooth)
    return torch.mean(jac, dim=-1)

class AdversarialLoss(nn.Module):
    def __init__(self, embed_dim, hidden_dim, args, lamda_weight=1000):
        """

        :param embed_dim: [10, 10]
        :param proj_dim: [32]
        :param directions:
        """
        super(AdversarialLoss, self).__init__()
        self.embed_dim = embed_dim # [10, 10]
        self.hidden_dim = hidden_dim
        self.lamda_weight = lamda_weight
        self.sim_mode = args.sim_mode

        self.sim = None

        self.directions = self.get_pair_directions(embed_dim)# []

        # Projection network
        self.regressors = nn.ModuleDict()
        for direction in self.directions:
            # j-i 代表 learner j --> learner i
            j, i = direction.split('-')
            self.regressors[direction] = nn.Sequential(
                nn.Linear(self.embed_dim[int(j)], self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.embed_dim[int(i)])
            ).to(device)

    def forward(self, inputs):
        embeds = []
        sim_loss = torch.tensor(0.0, requires_grad=True).to(device)
        # reg_loss = torch.tensor(0.0, requires_grad=True).to(device)
        weight_loss = torch.tensor(0.0, requires_grad=True).to(device)
        bias_loss = torch.tensor(0.0, requires_grad=True).to(device)

        for i in range(len(self.embed_dim)):
            start = int(sum(self.embed_dim[:i]))
            end = int(start + self.embed_dim[i])
            # embeds.append(grad_reverse(F.normalize(inputs[:, start:end])))
            embeds.append(grad_reverse(inputs[:, start:end]))

        for direction in self.directions:
            j, i = direction.split('-')

            if self.sim_mode == 'sim':
                sim_loss = sim_loss + \
                       -torch.mean(torch.mean((embeds[int(i)] *
                                            torch.square(
                                                self.regressors[direction](embeds[int(j)])
                                            )), dim=-1),dim=-1)
            elif self.sim_mode == 'hadamard':
                sim_loss = sim_loss + \
                           -torch.mean(torch.mean((torch.square(torch.mul(embeds[int(i)],
                                                                                    self.regressors[direction](
                                                                                        embeds[int(j)]))
                                                                )), dim=-1), dim=-1)
            elif self.sim_mode == 'cosine':
                sim_loss = sim_loss + \
                           -torch.mean(torch.mean((torch.square(F.cosine_similarity(embeds[int(i)],
                                                                         self.regressors[direction](embeds[int(j)]))
                                                                )), dim=-1), dim=-1)
            elif self.sim_mode == 'jaccard':
                sim_loss = sim_loss + \
                           -torch.mean(torch.mean((torch.square(jaccard_index(embeds[int(i)],
                                                                         self.regressors[direction](embeds[int(j)]))
                                                                )), dim=-1), dim=-1)
            elif self.sim_mode == 'pearson':
                sim_loss = sim_loss + \
                           -torch.mean(torch.mean((torch.square(get_corr(embeds[int(i)],
                                                    self.regressors[direction](embeds[int(j)]))
                                                )), dim=-1),dim=-1)
            elif self.sim_mode == 'spearman':
                sim_loss = sim_loss + \
                           -torch.mean(torch.mean((torch.square(spearman_correlation(embeds[int(i)],
                                                                         self.regressors[direction](embeds[int(j)]))
                                                                )), dim=-1), dim=-1)

            # weights_loss ==> (weight_loss, bias_loss)
            weights_loss = self.pairs_weight_loss(self.regressors[direction])

            weight_loss = weight_loss + weights_loss[0]
            bias_loss = bias_loss + weights_loss[1]

        return sim_loss + (weight_loss  + bias_loss) * self.lamda_weight
        # return sim_loss

    def get_pair_directions(self, embed_dim):
        directions = []
        for i in range(len(embed_dim)):
            for j in range(i+1, len(embed_dim)):
                directions.append("{}-{}".format(j, i))
        return directions

    def pairs_weight_loss(self, regressor):
        weight = []
        bias = []
        for param in regressor.named_parameters():
            if 'weight' in param[0]:
                weight.append(param[1])
            else:
                bias.append(param[1])

        weight_loss = torch.tensor(0.0, requires_grad=True).to(device)
        bias_loss = torch.tensor(0.0, requires_grad=True).to(device)
        for w in weight:
            weight_loss = weight_loss + \
                          torch.mean(torch.sum(torch.square(torch.mm(w.t(), w) - 1.0), dim=-1))

        for b in bias:
            bias_loss = bias_loss + F.relu(torch.sum(b * b) - 1.0)

        return (weight_loss, bias_loss)


class GradRev(Function):
    """
    Implements an autograd class to flip gradients during backward pass.
    """
    @staticmethod
    def forward(self, x):
        """
        Container which applies a simple identity function.

        Input:
            x: any torch tensor input.
        """
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        """
        Container to reverse gradient signal during backward pass.

        Input:
            grad_output: any computed gradient.
        """
        return (grad_output * -1.)

def grad_reverse(x):

    return GradRev().apply(x)

if __name__ == '__main__':
    x = torch.randn(80, 32)
    y = torch.randn(80, 32)
    print(jaccard_index(x, y))
