import torch
from torch import nn
from math import log, pi
from torch.distributions import Multinomial

class GMM(nn.Module):
    def __init__(self, num_vars=100, num_components=1000, log_std=-1.):
        super(GMM, self).__init__()

        self.num_vars = num_vars
        self.num_components = num_components

        self.means = nn.Parameter(torch.rand(num_components, num_vars) * 2 - 1)
        self.log_std = nn.Parameter(
            log_std * torch.ones(num_components, num_vars))

        self.log_weight = nn.Parameter(torch.zeros(num_components))

    def stabilize(self):
        pass
        
    def log_prob(self, x):
        norm_comps_log_prob = -0.5 * (self.num_vars * log(2 * pi) + (
                    self.log_std.unsqueeze(0) + ((x.unsqueeze(
                1) - self.means.unsqueeze(0)) ** 2) / torch.exp(
                self.log_std).unsqueeze(0)).sum(dim=-1))
        weight_log_prob = torch.log_softmax(self.log_weight, dim=0)

        return torch.logsumexp(
            weight_log_prob.unsqueeze(0) + norm_comps_log_prob, dim=-1)

    def sample(self, num_samples):
        noise = torch.randn(num_samples, self.num_vars).to(self.means.device)

        comp_sampler = Multinomial(logits=self.log_weight)

        components = comp_sampler.sample_n(num_samples).cuda()

        return (components.unsqueeze(-1) * (self.means.unsqueeze(0) + torch.exp(
            self.log_std / 2).unsqueeze(0) * noise.unsqueeze(1))).sum(dim=1)
