import torch
import torch.nn as nn
import nn_module as M


class StyleEncoder(nn.Module):
    def __init__(self, in_channels, init_type="xavier", activation=nn.ReLU(), norm_type="IN"):
        super(StyleEncoder, self).__init__()
        self.channel_list = [64, 128, 256, 256]
        self.layers = []
        self.layers += M.conv2d_block(
                in_channels=in_channels,
                out_channels=self.channel_list[0],
                kernel_size=7,
                stride=1,
                padding=3,
                init_type=init_type,
                activation=activation,
                norm_type=norm_type
                )
        for i in range(len(self.channel_list) - 1):
            self.layers += M.conv2d_block(
                    in_channels=self.channel_list[i],
                    out_channels=self.channel_list[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type=norm_type
                    )
            self.layers += [nn.MaxPool2d(kernel_size=2)]
        self.layers += [nn.AdaptiveAvgPool2d(output_size=1)]
        self.layers += M.conv2d_block(
                in_channels=self.channel_list[-1],
                out_channels=self.channel_list[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                init_type=init_type,
                activation=None,
                norm_type=None
                )
        self.main = nn.Sequential(*self.layers)
        self.output_dim = self.channel_list[-1]

    def forward(self, x):
        x = self.main(x)
        return x


class ConetentEncoder(nn.Module):
    def __init__(self, in_channels, res_num, init_type="xavier", activation=nn.ReLU(), norm_type="IN"):
        super(ConetentEncoder, self).__init__()
        self.channel_list = [64, 128, 256, 512]
        self.layers = []
        self.layers += M.conv2d_block(
                in_channels=in_channels,
                out_channels=self.channel_list[0],
                kernel_size=7,
                stride=1,
                padding=3,
                init_type=init_type,
                activation=activation,
                norm_type=norm_type
                )
        for i in range(len(self.channel_list) - 1):
            self.layers += M.conv2d_block(
                    in_channels=self.channel_list[i],
                    out_channels=self.channel_list[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type=norm_type
                    )
            self.layers += [nn.MaxPool2d(kernel_size=2)]
        for i in range(res_num):
            self.layers += [M.ResidualBlock(
                    in_channels=self.channel_list[-1],
                    out_channels=self.channel_list[-1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type=norm_type,
                    is_bottleneck=False
                    )]
        self.main = nn.Sequential(*self.layers)
        self.output_dim = self.channel_list[-1]

    def forward(self, x):
        x = self.main(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, res_num, init_type="xavier", activation=nn.ReLU()):
        super(Decoder, self).__init__()
        self.channel_list = [256, 128, 64]
        self.layers = []
        for i in range(res_num):
            self.layers += [M.ResidualBlock(
                in_channels=self.channel_list[0],
                out_channels=self.channel_list[0],
                kernel_size=3,
                stride=1,
                padding=1,
                init_type=init_type,
                activation=activation,
                norm_type='IN',
                is_bottleneck=False
                )]
        for i in range(len(self.channel_list) - 1):
            self.layers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            self.layers += M.conv2d_block(
                    in_channels=self.channel_list[i],
                    out_channels=self.channel_list[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    init_type=init_type,
                    activation=activation,
                    norm_type='LN'
                    )

        self.layers += [M.conv2d_block(
            in_channels=self.channel_list[-1],
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            init_type=init_type,
            activation=nn.Tanh(),
            norm_type=None
            )]
        self.main = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.main(x)
        return x


class CariStyleGAN(nn.Module):
    def __init__(self, in_channels, init_type="xavier", norm_type='IN'):
        super(CariStyleGAN, self).__init__()

    def forward(self, x):
        return x


def unit_test_encoder():
    style_encoder = StyleEncoder(in_channels=3)
    print(style_encoder)
    content_encoder = ConetentEncoder(in_channels=3, res_num=2)
    print(content_encoder)


def unit_test_decoder():
    decoder = Decoder(out_channels=3, res_num=2)
    print(decoder)


if __name__ == "__main__":
    # unit_test_encoder()
    unit_test_decoder()
