from torch import nn

class Encoder(nn.Module):
    def __init__(self, final_output=512, img_size=64, ndf = 64,
                 use_biases=True):
        super(Encoder, self).__init__()

        self.ndf = ndf
        self.img_size =img_size
        self.use_biases = use_biases

        if img_size == 64:
            self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1, bias=False)
            self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            self.activ = nn.LeakyReLU(0.2)
            self.fc_final = nn.Linear((ndf * 8) * 4 * 4, final_output)
        elif img_size == 32:
            self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1, bias=False)
            self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            self.activ = nn.LeakyReLU(0.2)
            self.fc_final = nn.Linear((ndf * 4) * 4 * 4, final_output)

        self.out_size = final_output

    def forward(self, x):
        cur_x = self.activ(self.conv1(x))
        cur_x = self.activ(self.conv2(cur_x))
        cur_x = self.activ(self.conv3(cur_x))
        if self.img_size == 64:
            cur_x = self.conv4(cur_x)

        cur_x = self.activ(cur_x).view(cur_x.shape[0], -1)

        return self.fc_final(cur_x)


class Decoder(nn.Module):
    def __init__(self, num_inputs=32, img_size=64, ngf = 64, use_biases=True):
        super(Decoder, self).__init__()

        self.ngf = ngf

        self.img_size = img_size
        self.use_biases = use_biases

        if self.img_size == 64:
            self.fc_first = nn.Linear(num_inputs, ngf * 8)

            self.deconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 1, 0, bias=False)
            self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            self.deconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            self.deconv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
            self.deconv5 = nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False)
        else:
            self.fc_first = nn.Linear(num_inputs, ngf * 8)

            self.deconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 0,
                                              bias=False)
            self.deconv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1,
                                              bias=False)
            self.deconv3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1,
                                              bias=False)
            self.deconv4 = nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False)

        self.activ = nn.LeakyReLU(0.2)
        self.fin_activ = nn.Tanh()

    def forward(self, x):
        cur_x = self.activ(self.fc_first(x)).view(x.shape[0], self.ngf * 8, 1,
                                                  1)

        cur_x = self.activ(self.deconv1(cur_x))
        cur_x = self.activ(self.deconv2(cur_x))
        cur_x = self.activ(self.deconv3(cur_x))
        cur_x = self.activ(self.deconv4(cur_x))

        if self.img_size == 64:
            cur_x = self.activ(self.deconv5(cur_x))

        return self.fin_activ(cur_x)
