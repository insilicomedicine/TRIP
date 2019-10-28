import torch
from torch import nn
from torch.distributions import MultivariateNormal


class RealNVPModule(nn.Module):
    def __init__(self, mask, hidden=64):
        super(RealNVPModule, self).__init__()

        self.register_buffer('mask', mask)
        self.hidden = hidden

        self.num_vars = self.mask.shape[0]

        self.s_net = nn.Sequential(
            nn.Linear(self.num_vars,
                      self.hidden),
            nn.LeakyReLU(),
            nn.Linear(self.hidden,
                      self.hidden),
            nn.LeakyReLU(),
            nn.Linear(self.hidden,
                      self.num_vars),
            nn.Tanh()
        )

        self.t_net = nn.Sequential(
            nn.Linear(self.num_vars,
                      self.hidden),
            nn.LeakyReLU(),
            nn.Linear(self.hidden,
                      self.hidden),
            nn.LeakyReLU(),
            nn.Linear(self.hidden,
                      self.num_vars),
        )

    def f(self, z):
        '''
        :param X: tuple of tensors, first consisting nodes latent
            and other edge latents
        :return: tuple of tensors, first consisting nodes latent
            and other edge latents
        '''
        mask = self.mask[None, :]

        z_ = z * mask

        s_out = self.s_net(z_) * (1 - mask)
        t_out = self.t_net(z_) * (1 - mask)

        z_new = z_ + (1 - mask) * (z * torch.exp(s_out) + t_out)

        return z_new

    def g(self, z):
        '''
        :param X: tuple of tensors, first consisting nodes latent
            and other edge latents
        :return: tuple of tensors, first consisting nodes latent
            and other edge latents
        '''
        mask = self.mask[None, :]

        z_ = z * mask

        s_out = self.s_net(z_) * (1 - mask)
        t_out = self.t_net(z_) * (1 - mask)

        z_new = (1 - mask) * (z - t_out) * torch.exp(-s_out) + z_

        log_det_J_change = -s_out.sum(dim=-1)

        return z_new, log_det_J_change


class RealNVP(nn.Module):
    def __init__(self, num_vars=100, num_layers=4, prior=None):
        super(RealNVP, self).__init__()

        self.num_vars = num_vars
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        self.prior = prior

        masks = self.num_layers * \
                [[((i + shift) % 2) for i in range(num_vars)]
                 for shift in range(2)]

        for mask in masks:
            self.layers.append(RealNVPModule(torch.tensor(mask).float()))

    def log_prob(self, x):
        log_prob = 0
        for layer in self.layers[::-1]:
            x, log_prob_change = layer.g(x)
            log_prob = log_prob_change + log_prob

        if self.prior is None:
            norm_prior = MultivariateNormal(torch.zeros(self.num_vars).to(x.device),
                                            torch.eye(self.num_vars).to(x.device))

            log_prob += norm_prior.log_prob(x)
        else:
            log_prob += self.prior.log_prob(x)

        return log_prob

    def sample(self, num_samples):
        if self.prior is None:
            norm_prior = MultivariateNormal(torch.zeros(self.num_vars).to(self.s_net.device),
                                            torch.eye(self.num_vars).to(self.s_net.device))

            x = norm_prior.sample((num_samples, ))
        else:
            x = self.prior.sample(num_samples)

        for layer in self.layers:
            x = layer.f(x)

        return x

    def reinit_from_data(self, x):
        for layer in self.layers[::-1]:
            x, _ = layer.g(x)

        self.prior.reinit_from_data(x)