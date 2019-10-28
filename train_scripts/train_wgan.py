import sys

sys.path.append('..')

from core.networks import Generator_CNN, Discriminator_CNN

from core.learnable_priors import TRIP, GMM, Normal
from core.generative_models import WGAN
from core.utils import TrainStats

import pickle as pkl

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

import argparse

import os

#---------------------------------------------------

parser = argparse.ArgumentParser()
# model description
parser.add_argument('--gamma', type=float, default=10)

parser.add_argument('--gp', dest='gp', action='store_true')
parser.add_argument('--no_gp', dest='gp', action='store_false')

parser.add_argument('--prior', type=str, default='trip')
parser.add_argument('--train_prior', dest='train_prior', action='store_true')
parser.add_argument('--fixed_prior', dest='train_prior', action='store_false')

#dataset desciption
parser.add_argument('--dataset', type=str, default='celeba')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_latents', type=int, default=128)
parser.add_argument('--num_components', type=int, default=10)
parser.add_argument('--tt_int', type=int, default=40)

#train description
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lp_lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=int, default=-1,
                    help='if -1 then perform computations on cpu, else on gpu'
                         'with set number')

# dirs
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--save_model_dir', type=str, default='../saved_models/')
parser.add_argument('--save_hist_dir', type=str, default='../saved_hists/')

parser.add_argument('--seed', type=int, default=777)

args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#---------------------------------------------------

os.makedirs(args.save_model_dir, exist_ok=True)
os.makedirs(args.save_hist_dir, exist_ok=True)

torch.manual_seed(args.seed)

if args.gpu >= 0:
    device = 'cuda'
    torch.cuda.set_device(args.gpu)
else:
    device = 'cpu'

if args.dataset == 'cifar10':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)

    gen = Generator_CNN(128, out_side_size=32)
    dis = Discriminator_CNN(3, in_side_size=32, gan_type='wgangp')
elif args.dataset == 'celeba':
    from core.custom_datasets import CelebaDataset
    
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
    
    gen = Generator_CNN(128, out_side_size=64)
    dis = Discriminator_CNN(3, in_side_size=64, gan_type='wgangp')

if args.prior == 'trip':
    prior = TRIP(args.num_latents * (('c', args.num_components),),
                 tt_int=args.tt_int, distr_init='uniform')
elif args.prior == 'gmm':
    prior = GMM(num_vars=args.num_latents,
                   num_components=args.num_latents * args.num_components)
elif args.prior == 'normal':
    prior = Normal(num_vars=args.num_latents)


model = WGAN(gen, dis, prior, gamma=args.gamma, device=device, gp=args.gp)
model.to(device)

    
model.train()

model_name = 'wgangp_' + args.prior + '_' + \
             ('lp' if args.train_prior else 'nolp') + '_' + \
             args.dataset + 'seed_' + str(args.seed)

global_stats = TrainStats()

for i in range(args.n_epochs):
    local_hist = model.make_training(data_loader, global_stats=global_stats,
                                     num_iterations=1000, lr=args.lr, lp_lr=args.lp_lr,
                                     train_lp=args.train_prior)

    model.cpu()
    torch.save(model.state_dict(),
               args.save_model_dir+model_name+'_latest.model')
    model.to(device)
    pkl.dump(global_stats.stats,
             open(args.save_hist_dir+model_name+'_latest.pkl', 'wb'))
