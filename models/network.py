import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from .src.snlayers.snlayer import SpectralNorm 
from models.spectral_normalization import SpectralNorm

class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
        
        # SN
        #self.sn = SpectralNorm(nn.Module) 

        # Unet encoder
        self.conv1 = nn.Conv2d(7, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # Res Block
        self.res_block = ResidualBlock(512, 512)

        # Unet decoder
        '''
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)
        '''
        self.up = nn.Upsample(scale_factor = 2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor = 4, mode='nearest')
        self.pad = nn.ReflectionPad2d(1) 
        self.upconv1 = nn.Conv2d(d * 8, d * 8, 3, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.upconv2 = nn.Conv2d(d * 8 * 2, d * 8, 3, 1, 0)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.upconv3 = nn.Conv2d(d * 8 * 2, d * 8, 3, 1, 0)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.upconv4 = nn.Conv2d(d * 8 * 2, d * 4, 3, 1, 0)
        self.deconv4_bn = nn.BatchNorm2d(d * 4)
        self.upconv5 = nn.Conv2d(d * 4 * 2, d * 4, 3, 1, 0)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.upconv6 = nn.Conv2d(d * 4 * 2, d * 2, 3, 1, 0)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.upconv7 = nn.Conv2d(d * 2 * 2, d, 3, 1, 0)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.upconv8 = nn.Conv2d(d * 2, 3, 3, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        #e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        #e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        
        e6_res1 = self.res_block(e6)
        e6_res2 = self.res_block(e6_res1)
        e6_res3 = self.res_block(e6_res2)
        #e6_res4 = self.res_block(e6_res3)
        #e6_res5 = self.res_block(e6_res4)
        e6 = e6_res3

        '''
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        '''
        d1 = F.dropout(self.deconv1_bn(self.upconv1(self.pad(self.up(F.relu(e6))))), 0.5, training=True)
        #d1 = F.dropout(self.sn(self.upconv1(self.pad(self.up(F.relu(e6))))), 0.5, training=True)
        d1 = torch.cat([d1, self.up(e6)], 1)
        #print('d1 {}'.format(d1.shape))
        d2 = F.dropout(self.deconv2_bn(self.upconv2(self.pad(self.up(F.relu(d1))))), 0.5, training=True)
        #d2 = F.dropout(self.sn(self.upconv2(self.pad(self.up(F.relu(d1))))), 0.5, training=True)
        d2 = torch.cat([d2, self.up(e5)], 1)
        #print('d2 {}'.format(d2.shape))
        d3 = F.dropout(self.deconv3_bn(self.upconv3(self.pad(self.up(F.relu(d2))))), 0.5, training=True)
        #d3 = F.dropout(self.sn(self.upconv3(self.pad(self.up(F.relu(d2))))), 0.5, training=True)
        d3 = torch.cat([d3, self.up(e4)], 1)
        #print('d3 {}'.format(d3.shape))
        #print('e6 {}'.format(e6.shape))
        #print('e5 {}'.format(e5.shape))
        #print('e4 {}'.format(e4.shape))
        #print('e3 {}'.format(e3.shape))
        #print('e2 {}'.format(e2.shape))
        #print('e1 {}'.format(e1.shape))
        d4 = F.dropout(self.deconv4_bn(self.upconv4(self.pad(self.up(F.relu(d3))))))
        #d4 = F.dropout(self.deconv4_bn(self.upconv4(self.pad(self.up(F.relu(d3))))))
        #d4 = F.dropout(self.sn(self.upconv4(self.pad(self.up(F.relu(d3))))))
        d4 = torch.cat([d4, self.up(e3)], 1)
        ##print('d4 {}'.format(d4.shape))
        d5 = F.dropout(self.deconv5_bn(self.upconv5(self.pad(self.up(F.relu(d4))))))
        #d5 = F.dropout(self.sn(self.upconv5(self.pad(self.up(F.relu(d4))))))
        ##print('d5 {}'.format(d5.shape))
        d5 = torch.cat([d5, self.up2(e3)], 1)
        d6 = F.dropout(self.deconv6_bn(self.upconv6(self.pad(self.up(F.relu(d5))))))
        #d6 = F.dropout(self.sn(self.upconv6(self.pad(self.up(F.relu(d5))))))
        #d6 = torch.cat([d6, self.up(e1)], 1)
        #d7 = F.dropout(self.deconv7_bn(self.upconv7(self.pad(self.up(F.relu(d6))))))
        #d7 = torch.cat([d7, e1], 1)
        d8 = self.upconv8(self.pad(F.relu(d6)))
        o = F.tanh(d8)
        #print('output size{}'.format(o.shape))

        return o

class discriminator_snIns(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator_snIns, self).__init__()
        #self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv1 = SpectralNorm(nn.Conv2d(3, d, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(d, d, 4, 2, 1))
        self.conv2_in = nn.InstanceNorm2d(d)
        self.conv3 = SpectralNorm(nn.Conv2d(d, d * 2, 3, 1, 1))
        self.conv3_in = nn.InstanceNorm2d(d * 2)
        self.conv4 = SpectralNorm(nn.Conv2d(d *2 , d * 2, 4, 1, 1))
        self.conv4_in = nn.InstanceNorm2d(d * 2)
        self.conv5 = SpectralNorm(nn.Conv2d(d * 2, d * 4, 3, 1, 1))
        self.conv5_in = nn.InstanceNorm2d(d * 4)
        self.conv6 = SpectralNorm(nn.Conv2d(d * 4, d * 4, 2, 1, 1))
        self.conv6_in = nn.InstanceNorm2d(d * 4)
        self.conv7 = SpectralNorm(nn.Conv2d(d * 4, d * 4, 3, 1, 1))
        self.conv7_in = nn.InstanceNorm2d(d * 4)
        self.conv8 = SpectralNorm(nn.Conv2d(d * 4, d * 8, 4, 2, 1))
        self.fc = SpectralNorm(nn.Linear(4 * 4 * d * 8, 1))


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input_):
        #x = torch.cat([input, label], 1)
        x = input_ 
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_in(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_in(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.conv7_in(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.conv8(x))

        return self.fc(x.view(-1, 4 * 4 * 512)) 

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        #self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    #def forward(self, input, label):
    def forward(self, input_):
        #x = torch.cat([input, label], 1)
        x = input_ 
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                # here we change original stride=2 to stride=1, we think high resolution helps for detailed decision
                # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                #           kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 32, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(32, 128, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(128, 64, 4, stride=1, padding=0),  # b, 8, 3, 3
            nn.ReLU(True),

        )
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(64, 128, 5, stride=2, padding=1),  # b, 8, 15, 15
            nn.ConvTranspose2d(64, 128, 5, stride=2, padding=4),  # b, 8, 15, 15
            nn.ReLU(True),
            #nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),  # b, 1, 28, 28
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),  # b, 1, 28, 28
            nn.ReLU(True),
            #nn.ConvTranspose2d(64, 16, 5, stride=2, padding=1),  # b, 1, 28, 28
            nn.ConvTranspose2d(64, 16, 5, stride=2, padding=2),  # b, 1, 28, 28
            nn.ReLU(True),
            #nn.ConvTranspose2d(16, 3, 2, stride=3, padding=8),  # b, 1, 28, 28
            nn.ConvTranspose2d(16, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x): 
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x

def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=padding, dilation=dilation)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride,
                             padding=dilation[0])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride,
                             padding=dilation[1])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

