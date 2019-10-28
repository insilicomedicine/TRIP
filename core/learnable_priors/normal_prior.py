import torch
from torch import nn
from torch.distributions import MultivariateNormal

class Normal(nn.Module):
    def __init__(self, num_vars=100):
        super(Normal, self).__init__()

        self.num_vars = num_vars

        self.means = nn.Parameter(torch.zeros(num_vars))
        self.std = nn.Parameter(torch.eye(num_vars))

    def log_prob(self, x):
        distr = MultivariateNormal(self.means, self.std)
        return distr.log_prob(x)

    def sample(self, num_samples):
        distr = MultivariateNormal(self.means, self.std)
        return distr.sample_n(num_samples)
