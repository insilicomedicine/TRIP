from torch import nn

class Generator_CNN(nn.Module):
    def __init__(self, in_dim, dim=64, out_side_size=64):
        super(Generator_CNN, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.ReLU())

        if out_side_size==64:
            self.l1 = nn.Sequential(
                nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
                nn.ReLU())
            self.l2_5 = nn.Sequential(
                dconv_bn_relu(dim * 8, dim * 4),
                dconv_bn_relu(dim * 4, dim * 2),
                dconv_bn_relu(dim * 2, dim),
                nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
                nn.Tanh())
        elif out_side_size==32:
            self.l1 = nn.Sequential(
                nn.Linear(in_dim, dim * 4 * 4 * 4, bias=False),
                nn.ReLU())
            self.l2_5 = nn.Sequential(
                dconv_bn_relu(dim * 4, dim * 2),
                dconv_bn_relu(dim * 2, dim),
                nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
                nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator_CNN(nn.Module):
    def __init__(self, in_dim, dim=64, gan_type='gan', in_side_size=64):
        super(Discriminator_CNN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                LayerNorm2d(out_dim) if (gan_type == 'wgangp') else nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))

        if in_side_size==64:
            self.ls = nn.Sequential(
                nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
                conv_ln_lrelu(dim, dim * 2),
                conv_ln_lrelu(dim * 2, dim * 4),
                conv_ln_lrelu(dim * 4, dim * 8),
                nn.Conv2d(dim * 8, 1, 4))
        elif in_side_size==32:
            self.ls = nn.Sequential(
                nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
                conv_ln_lrelu(dim, dim * 2),
                conv_ln_lrelu(dim * 2, dim * 4),
                nn.Conv2d(dim * 4, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

#----------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2d, self).__init__()

        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.transpose(1, -1)).transpose(1, -1)