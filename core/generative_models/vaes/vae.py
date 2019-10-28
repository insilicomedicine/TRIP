import torch
import torch.nn as nn
import torch.optim as optim
from math import pi, log

from ...utils import TrainStats

import sys

class VAE(nn.Module):
    def __init__(self, enc, dec, prior, beta=1, device='cpu'):
        super(VAE, self).__init__()

        self.enc = enc
        self.dec = dec

        self.lp = prior

        self.beta = beta

        self.device = device

    def get_elbo(self, x):
        enc_output = self.enc(x)
        means, log_stds = torch.split(enc_output,
                                      enc_output.shape[1] // 2,
                                      dim=1)
        latvar_samples = (means + torch.randn_like(log_stds) *
                          torch.exp(0.5 * log_stds))

        rec_part = -((self.dec(latvar_samples) - x) ** 2).sum(dim=-1).sum(
            dim=-1).mean()

        normal_distr_hentropies = (-latvar_samples.shape[-1]/2) * log(2 * pi) + \
                                  (- 0.5 * (latvar_samples - means) ** 2 / (torch.exp(log_stds) + 1e-10) -
                                   0.5 * log_stds).sum(dim=1)

        log_p_z = self.lp.log_prob(latvar_samples)

        kldiv_part = (normal_distr_hentropies - log_p_z).mean()

        elbo = rec_part - self.beta * kldiv_part

        return elbo, {
            'loss': -elbo.detach().cpu().numpy(),
            'rec': rec_part.detach().cpu().numpy(),
            'kl': kldiv_part.detach().cpu().numpy()
        }

    def make_training(self, train_loader, global_stats=None, num_iterations=20000, verbose_step=50,
              train_lp=True, lr=1e-4, lp_lr=1e-4):

        enc_dec_optimizer = optim.Adam(nn.ParameterList(
            list(self.enc.parameters()) + list(self.dec.parameters())),
            lr=lr)

        lp_optimizer = optim.Adam(self.lp.parameters(), lr=lp_lr)

        local_stats = TrainStats()

        cur_iteration = 0

        epoch_i = 0
        while cur_iteration < num_iterations:
            i = 0

            print("Epoch", epoch_i, ":")
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(self.device)

                print("!", end='')
                sys.stdout.flush()

                i += 1

                elbo, cur_stats = self.get_elbo(x_batch)
                local_stats.update(cur_stats)
                global_stats.update(cur_stats)

                enc_dec_optimizer.zero_grad()
                if train_lp:
                    lp_optimizer.zero_grad()
                loss = -elbo
                loss.backward()
                enc_dec_optimizer.step()
                if train_lp:
                    lp_optimizer.step()

                cur_iteration += 1
                if cur_iteration >= num_iterations:
                    break

                if i % verbose_step == 0:
                    local_stats.print()
                    local_stats.reset()
                    i = 0

            epoch_i += 1
            if i > 0:
                local_stats.print()
                local_stats.reset()

        return global_stats

    def sample(self, num_samples):
        z = self.lp.sample(num_samples)
        smiles = self.dec.sample(z)

        return smiles
