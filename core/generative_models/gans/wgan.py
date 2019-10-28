import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim

from ...utils import TrainStats

class WGAN(nn.Module):
    def __init__(self, gen, discr, prior, n_critic=5, gamma=1, gp=True,
                 device='cpu'):
        super(WGAN, self).__init__()

        self.gen = gen
        self.discr = discr
        self.prior = prior

        self.gamma = gamma

        self.n_critic = n_critic

        self.gp = gp

        self.device = device

    def get_losses(self, x, compute_reinforce=False):
        # get generator samples
        sampled_latents = self.prior.sample(x.shape[0])
        sampled_latents = sampled_latents.detach()
        sampled_images = self.gen(sampled_latents)

        # get discriminator outputs
        real_discr = self.discr(x)
        fake_discr = self.discr(sampled_images)

        # compute gradient penalties
        if self.gp:
            alphas = torch.rand(x.shape[0], 1, 1, 1).repeat(1, x.shape[1],
                                                        x.shape[2],
                                                        x.shape[3])
            alphas = alphas.to(self.device)
            int_points = alphas * sampled_images + (1 - alphas) * x
            int_points_discr = self.discr(int_points)

            gradients = autograd.grad(outputs=int_points_discr, inputs=int_points,
                                  grad_outputs=torch.ones(
                                      int_points_discr.size()).to(self.device),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]

            grad_norm = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # compute reinforce loss
        if compute_reinforce:
            rews = (fake_discr - fake_discr.mean()).detach()
            rews = rews / rews.std()
            lp_loss = -(rews * self.prior.log_prob(sampled_latents)).mean()
        else:
            lp_loss = torch.zeros(1).mean()

        # compute losses
        gen_loss = -fake_discr.mean()
        discr_loss = -(
                    real_discr.mean() - fake_discr.mean())

        if self.gp:
            discr_loss = discr_loss + self.gamma * grad_norm

        return gen_loss, \
               discr_loss, \
               lp_loss, \
               {
                   'gen_loss': gen_loss.detach().cpu().numpy(),
                   'discr_loss': discr_loss.detach().cpu().numpy(),
                   'lp_loss': lp_loss.detach().cpu().numpy(),
                   'grad_norm': grad_norm.detach().cpu().numpy()
               }

    def make_training(self, train_loader, global_stats=None, num_iterations=20000, verbose_step=50,
              train_lp=True, lr=1e-4, lp_lr=1e-4):
        gen_optimizer = optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, .9))
        discr_optimizer = optim.Adam(self.discr.parameters(), lr=lr,
                                     betas=(0.5, .9))
        lp_optimizer = optim.Adam(self.prior.parameters(), lr=lp_lr)

        local_stats = TrainStats()

        cur_iteration = 0

        epoch_i = 0
        while cur_iteration < num_iterations:
            i = 0

            print("Epoch", epoch_i, ":")
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(self.device)

                print("!", end='')
                i += 1

                gen_loss, discr_loss, lp_loss, cur_stats = self.get_losses(
                    x_batch, (i % self.n_critic == 0) and train_lp)
                local_stats.update(cur_stats)
                if global_stats is not None:
                    global_stats.update(cur_stats)

                if i % self.n_critic == 0:
                    gen_optimizer.zero_grad()
                    gen_loss.backward()
                    gen_optimizer.step()

                    if train_lp:
                        lp_optimizer.zero_grad()
                        lp_loss.backward()
                        lp_optimizer.step()

                        self.prior.stabilize()
                else:
                    discr_optimizer.zero_grad()
                    discr_loss.backward()
                    discr_optimizer.step()

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
        z = self.prior.sample(num_samples)
        samples = self.gen(z)
        return samples.detach().cpu().numpy()