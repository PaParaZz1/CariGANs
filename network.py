import torch
import torch.nn as nn
import nn_module as M


class StyleEncoder(nn.Module):
    def __init__(self, in_channels, style_dim, init_type="xavier", activation=nn.ReLU(), norm_type="IN"):
        super(StyleEncoder, self).__init__()
        self.channel_list = [64, 128, 256, 256, 256]
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
                out_channels=style_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                init_type=init_type,
                activation=None,
                norm_type=None
                )
        self.main = nn.Sequential(*self.layers)
        self.output_dim = style_dim

    def forward(self, x):
        x = self.main(x)
        return x


class ContentEncoder(nn.Module):
    def __init__(self, in_channels, res_num, init_type="xavier", activation=nn.ReLU(), norm_type="IN"):
        super(ContentEncoder, self).__init__()
        self.channel_list = [64, 128, 256]
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
    def __init__(self, res_num, out_channels, res_norm="AdaptiveIN", init_type="xavier", activation=nn.ReLU()):
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
                norm_type=res_norm,
                is_bottleneck=False
                )]
        for i in range(len(self.channel_list) - 1):
            self.layers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            self.layers += M.conv2d_block(
                    in_channels=self.channel_list[i],
                    out_channels=self.channel_list[i+1],
                    kernel_size=5,
                    stride=1,
                    padding=2,
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
            pad_type='reflect',
            activation=nn.Tanh(),
            norm_type=None
            )]
        self.main = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.main(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, block_num, norm_type=None, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = []
        self.layers.append(M.fc_block(input_dim, dim, norm_type=norm_type, activation=activation))
        for i in range(block_num):
            self.layers.append(M.fc_block(dim, dim, norm_type=norm_type, activation=activation))
        self.layers.append(M.fc_block(dim, output_dim, norm_type=None, activation=None))
        self.main = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.main(x.view(x.size()[0], -1))


class CariStyleGAN(nn.Module):
    def __init__(self, in_channels, res_num, style_dim, mlp_dim, activation=nn.ReLU(), init_type='xavier'):
        super(CariStyleGAN, self).__init__()
        self.res_num = res_num
        self.style_encoder = StyleEncoder(in_channels, style_dim)
        self.content_encoder = ContentEncoder(in_channels, res_num)
        self.decoder = Decoder(out_channels=in_channels, res_num=res_num)
        self.mlp = MLP(style_dim, self.get_param_num(self.decoder), mlp_dim, block_num=1, norm_type=None, activation=activation)

    def encode(self, x):
        return self.content_encoder(x), self.style_encoder(x)

    def decode(self, content, style):
        adain_param = self.mlp(style)
        self.assign_param(adain_param, self.decoder)
        img = self.decoder(content)
        return img

    def assign_param(self, adain_param, network):
        for m in network.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_param[:, :m.num_features]
                std = adain_param[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_param.size()[1] > 2*m.num_features:
                    adain_param = adain_param[:, 2*m.num_features:]

    def get_param_num(self, network):
        num = 0
        for m in network.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num += 2*m.num_features
        return num

    def forward(self, x):
        content, style = self.encode(x)
        x = self.decode(content, style)
        return x


def unit_test_encoder():
    style_encoder = StyleEncoder(in_channels=3)
    print(style_encoder)
    content_encoder = ContentEncoder(in_channels=3, res_num=2)
    print(content_encoder)


def unit_test_decoder():
    decoder = Decoder(out_channels=3, res_num=2)
    print(decoder)

def unit_test_cari_style():
    net = CariStyleGAN(in_channels=3, res_num=2, style_dim=8, mlp_dim=256).cuda()
    print(net)
    inputs = torch.randn(1, 3, 800, 600).cuda()
    output = net(inputs)
    print(output.shape)

if __name__ == "__main__":
    # unit_test_encoder()
    # unit_test_decoder()
    unit_test_cari_style()
