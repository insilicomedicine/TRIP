import sys

sys.path.append('..')

from core.learnable_priors import TRIP, GMM, Normal, RealNVP
from core.generative_models import VAE
from core.utils import TrainStats

import pickle as pkl

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

import argparse

import os

# ---------------------------------------------------

parser = argparse.ArgumentParser()
# model description
parser.add_argument('--num_latents', type=int, default=100)

parser.add_argument('--num_components', type=int, default=10)
parser.add_argument('--tt_int', type=int, default=20)

parser.add_argument('--prior', type=str, default='trip')

parser.add_argument('--the_same_num_params', dest='snp', action='store_true')

parser.add_argument('--train_prior', dest='train_prior', action='store_true')
parser.add_argument('--fixed_prior', dest='train_prior',
                    action='store_false')

parser.add_argument('--add_flows', dest='add_flows', action='store_true')

# dataset desciption
parser.add_argument('--dataset', type=str, default='celeba')
parser.add_argument('--batch_size', type=int, default=128)

# train description
parser.add_argument('--n_epochs', type=int, default=8)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--lp_lr', type=float, default=3e-4)
parser.add_argument('--gpu', type=int, default=-1,
                    help='if -1 then perform computations on cpu, else on gpu'
                         'with set number')

# dirs
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--save_model_dir', type=str, default='../saved_models/')
parser.add_argument('--save_hist_dir', type=str, default='../saved_hists/')

args = parser.parse_args()

# ---------------------------------------------------

os.makedirs(args.save_model_dir, exist_ok=True)
os.makedirs(args.save_hist_dir, exist_ok=True)

if args.gpu >= 0:
    device = 'cuda'
    torch.cuda.set_device(args.gpu)
else:
    device = 'cpu'

if args.dataset == 'celeba':
    from core.custom_datasets import CelebaDataset
    from core.networks import Encoder
    from core.networks import Decoder

    if args.snp:
        if args.add_flows:
            prior_params = {'normal': 63,
                        'trip': 61,
                        'gmm': 62,
                        'vampprior': 64
                        }
        else:
            prior_params = {'normal': 64,
                        'trip': 60,
                        'gmm': 63,
                        'vampprior': 64
                        }

            ndf = ngf = prior_params[args.prior] if args.train_prior else 64
    else:
        ndf = ngf = 64


    transform = transforms.Compose(
        [transforms.CenterCrop(160),
         transforms.Resize(64),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CelebaDataset(data_dir=args.data_dir, train=True,
                                             download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=10)

    enc = Encoder(final_output=args.num_latents * 2, ndf=ndf)
    dec = Decoder(num_inputs=args.num_latents, ngf=ngf)
elif args.dataset == 'cifar10':
    from core.networks import Encoder
    from core.networks import Decoder

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                             download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=10)

    enc = Encoder(final_output=args.num_latents * 2, img_size=32, ndf=64)
    dec = Decoder(num_inputs=args.num_latents, img_size=32, ngf=64)

if args.prior == 'trip':
    prior = TRIP(args.num_latents * (('c', args.num_components),),
                 tt_int=args.tt_int, distr_init='uniform')

elif args.prior == 'gmm':
    prior = GMM(num_vars=args.num_latents,
                   num_components=args.num_latents * args.num_components)
elif args.prior == 'normal':
    prior = Normal(num_vars=args.num_latents)

if args.add_flows:
    prior = RealNVP(num_vars=args.num_latents, prior=prior)

model = VAE(enc, dec, prior, device=device)
model.to(device)

model.train()

model_name = 'vae_' + str(args.num_latents) + '_' + args.prior

if args.prior == 'trip':
    model_name += str(args.num_components) + '_' + str(args.tt_int)

model_name += '_' + ('withflows' if args.add_flows else "noflows")

model_name += '_' + ('lp' if args.train_prior else 'nolp') + '_' + args.dataset

global_stats = TrainStats()

for i in range(args.n_epochs):
    if (i - 1) % 5 == 0:
        if args.prior == 'trip':
            latvar_samples = None

            with torch.no_grad():
                for x, _ in data_loader:
                    x = x.cuda()
                    enc_output = model.enc(x)
                    means, log_stds = torch.split(enc_output,
                                          enc_output.shape[1] // 2,
                                          dim=1)
                    cur_latvar_samples = (means + torch.randn_like(log_stds) *
                              torch.exp(0.5 * log_stds))

                    if latvar_samples is None:
                        latvar_samples = cur_latvar_samples
                    else:
                        latvar_samples = torch.cat((latvar_samples, cur_latvar_samples), dim=0)

                    if latvar_samples.shape[0] > 2000:
                        break

            model.lp.reinit_from_data(latvar_samples)
            model.lp.cuda()

    local_hist = model.make_training(data_loader, global_stats=global_stats,
                                     num_iterations=10000, lr=args.lr,
                                     lp_lr=args.lp_lr,
                                     train_lp=args.train_prior)

    model.cpu()
    torch.save(model.state_dict(),
               args.save_model_dir + model_name + '_latest.model')
    torch.save(model.state_dict(),
               args.save_model_dir + model_name + '_{}.model'.format(i))
    model.to(device)
    pkl.dump(global_stats.stats,
             open(args.save_hist_dir + model_name + '_latest.pkl', 'wb'))
