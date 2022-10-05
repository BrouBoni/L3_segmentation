import functools

import torch.nn as nn


def dice_loss(output, target):
    smooth = 1.
    loss = 0.
    for c in range(target.shape[1]):
        output_flat = output[:, c].contiguous().view(-1)
        target_flat = target[:, c].contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss += 1. - ((2. * intersection + smooth) /
                      (output_flat.sum() + target_flat.sum() + smooth))

    return loss / target.shape[1]


def dice_score(output, target):
    return 1 - dice_loss(output, target)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def build_conv_block(dim, norm_layer, use_dropout, use_bias):
    conv_block = []
    conv_block += [nn.ReflectionPad2d(1)]
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    conv_block += [nn.ReLU(True)]

    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    conv_block += [nn.ReflectionPad2d(1)]
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    conv_block += [norm_layer(dim)]

    return nn.Sequential(*conv_block)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9,
                 norm_layer=functools.partial(nn.BatchNorm2d, affine=True),
                 use_dropout=False):
        super(ResnetGenerator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2 * ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4 * ngf),
            nn.ReLU(True),
        ]

        for _ in range(n_blocks):
            model += [ResnetBlock(4 * ngf, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=True)]

        model += [

            nn.ConvTranspose2d(4 * ngf, 2 * ngf,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, ngf, kernel_size=3, padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        self.model = nn.Sequential(*model)
        self.conv_segmentation = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.model(x)
        x = self.conv_segmentation(x)
        return x

    def __str__(self):
        return "ResnetGenerator"


class ResnetBlock(nn.Module):

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = build_conv_block(dim, norm_layer, use_dropout, use_bias)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.relu(x + out)
        return out


def define_generator(input_nc, output_nc, ngf, n_blocks, device,
                     use_dropout=False):
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

    net_g = ResnetGenerator(input_nc, output_nc, ngf, n_blocks, norm_layer=norm_layer,
                            use_dropout=use_dropout)

    net_g.to(device)
    net_g.apply(weights_init)
    return net_g


def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()
    return num_params
