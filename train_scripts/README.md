# Training scripts

## Usage

### Data
In the paper, we provide the experiments on CIFAR-10 and CelebA datasets. Both datasets will be downloaded automatically into `../data/` folder when you run the experiment for the first time. Please note that CelebA weights ~1.3G, so downloading it may take a while.

### WGAN and WGAN-GP

To train WGAN with learnable prior use

```
$ python train_wgan.py
```

#### Parameters:
`--gp` or `--nogp` - use gradient penalty or not

`--gamma` - coefficient of gradient penalty

`--prior` - can be `trip`, `gmm` or `normal`

`--train_prior` or `--fixed_prior`

`--dataset` – can be `celeba` or `cifar10`

`--batch_size` - default `128`

`--num_latents` - latent code size

##### Tensor Ring specific parameters:

`--num_components` - number of Gaussians in each one-dimensional mixture

`--tt_int` - internal core size

##### Parameters of training:

`--n_epochs` - number of epochs

`--lr` - learning rate for discriminator and generator

`--lr_lp` - learning rate for learnable prior 

##### Other parameters

`--gpu` - gpu for training

`--seed` - seed for reproducibility

`--data_dir`

`--save_model_dir` - directory for model checkpoints

`--save_hist_dir` - directory for logs

-------------------

### VAE

To train VAE with learnable prior use

```
$ python train_vae.py
```

#### Parameters:

`--add_flows` - whether to add normalizing flows before prior

`--prior` - can be `trip`, `gmm` or `normal`

`--train_prior` or `--fixed_prior`

`--dataset` –– can be `celeba` or `cifar10`

`--batch_size` - default `128`

`--num_latents` - latent code size

##### Tensor Ring specific parameters:

`--num_components` - number of Gaussians in each one-dimensional mixture

`--tt_int` - internal core size

##### Parameters of training:

`--n_epochs` - number of epochs

`--lr` - learning rate for encoder and decoder

`--lr_lp` - learning rate for learnable prior 

##### Other parameters

`--gpu` - gpu for training

`--seed` - seed for reproducibility

`--data_dir`

`--save_model_dir` - directory for model checkpoints

`--save_hist_dir` - directory for logs

-------------------

