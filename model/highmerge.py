import torch
import torch.nn as nn
import torch.nn.functional as F
class fusion_h(nn.Module):
    def __init__(self, dim=3, block_depth=3):
        super(fusion_h, self).__init__()
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv_fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        x = self.sig(x)
        y = self.conv1(y)
        return self.conv_fuse(x * y)


class Conv1x1(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inc, outc, 1)

    def forward(self, x):
        return self.conv(x)
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.weight is not None:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class Highmerge(nn.Module):
    def __init__(self, ngf=64, bn=False):
        super(Highmerge, self).__init__()
        # 下采样
        self.down1 = ResnetBlock(3, first=True)
        self.down2 = ResnetBlock(ngf, levels=2)
        self.down3 = ResnetBlock(ngf * 2, levels=2, bn=bn)

        self.down1_high = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(3, ngf, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(ngf),
                                        nn.ReLU(True))

        self.down2_high = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True))

        self.down3_high = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 4),
                                        nn.ReLU(True))

        self.res = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.fusion_layer = nn.ModuleList([fusion_h(dim=2 ** i * ngf) for i in range(0, 3)])
        self.skfusion = SKConv(features=2 ** 3 * ngf)
        self.conv1 = Conv1x1(inc=2 ** 3 * ngf, outc=2 ** (3 - 1) * ngf)
        
        # 上采样
        self.up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2), #if not bn else nn.BatchNorm2d(ngf * 2),
            CALayer(ngf * 2),
            PALayer(ngf * 2))

        self.up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            CALayer(ngf),
            PALayer(ngf))

        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh())

        self.info_up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2) #if not bn else nn.BatchNorm2d(ngf * 2, eps=1e-5),
        )

        self.info_up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf)  # if not bn else nn.BatchNorm2d(ngf, eps=1e-5),
        )

        self.fam1 = FE_Block(ngf, ngf)
        self.fam2 = FE_Block(ngf, ngf * 2)
        self.fam3 = FE_Block(ngf * 2, ngf * 4)

        self.att1 = Fusion_Block(ngf)
        self.att2 = Fusion_Block(ngf * 2)
        self.att3 = Fusion_Block(ngf * 4, bn=bn)

        self.merge2 = nn.Sequential(
            ConvBlock(ngf * 2, ngf * 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.merge3 = nn.Sequential(
            ConvBlock(ngf, ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )

        self.upsample = F.interpolate
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(init_weights)

    def forward(self, hazy, high):
        ###############   high-frequency encode   ###################
        x_down1_high = self.down1_high(high)  # [bs, 64, 256, 256]
        # print("x_down1_high:", x_down1_high.size())
        x_down2_high = self.down2_high(x_down1_high)  # [bs, 128, 128, 128]
        # print("x_down2_high:", x_down2_high.size())
        x_down3_high = self.down3_high(x_down2_high)  # [bs, 256, 64, 64]
        # print("x_down3_high:", x_down3_high.size())

        ###############   hazy encode   ###################
        x_down1 = self.down1(hazy)  # [bs, ngf, ngf * 4, ngf * 4]
        # print("x_down1:", x_down1.size())
        att1 = self.att1(x_down1_high, x_down1)
        x_down2 = self.down2(x_down1)  # [bs, ngf*2, ngf*2, ngf*2]
        # print("x_down2:", x_down2.size())
        att2 = self.att2(x_down2_high, x_down2)
        fuse2 = self.fam2(att1, att2)
        x_down3 = self.down3(x_down2)  # [bs, ngf * 4, ngf, ngf]
        # print("x_down3:", x_down3.size())
        att3 = self.att3(x_down3_high, x_down3)
        fuse3 = self.fam3(fuse2, att3)

        ###############   dehaze   ###################
        x6 = self.res(x_down3)
        fuse_up2 = self.info_up1(fuse3)
        fuse_up2 = self.merge2(fuse_up2 + x_down2)
        fuse_up3 = self.info_up2(fuse_up2)
        fuse_up3 = self.merge3(fuse_up3 + x_down1)
        x_up2 = self.up1(x6 + fuse3)
        x_up3 = self.up2(x_up2 + fuse_up2)
        x_up4 = self.up3(x_up3 + fuse_up3)
        # print("x_up4:", x_up4.size(), x_up4.min().item(), x_up4.max().item())

        ##############   output dehazed image   #################
        # dehaze = x_up4  # 去雾后的图像
        dehaze = self.upsample(x_up4, size=hazy.shape[2:], mode='bilinear', align_corners=True)
        # Adjust the shape of dehaze to match hazy
        dehaze = dehaze[:hazy.shape[0], :, :, :]
        # print("dehaze:", dehaze.size(), dehaze.min().item(), dehaze.max().item())


        return dehaze

if __name__ == '__main__':
    G = Highmerge()
    a = torch.randn(1, 3, 512, 768)
    b = torch.randn(1, 3, 512, 768)
    output = G(a, b)
    print(output.shape)


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=1, device=5):
        super(Lap_Pyramid_Conv, self).__init__()
        self.num_high = num_high
        self.device = device
        self.kernel = self.gauss_kernel().to(self.device)

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel = kernel
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel
        return kernel.to(self.device)

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        x = x.to(self.device)
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(self.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2).to(self.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        x_up = self.conv_gauss(x_up, 4 * self.kernel)

        return x_up

    def conv_gauss(self, img, kernel):
        img = img.to(self.device)
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        img = img.to(self.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)

        return pyr


class ResnetBlock(nn.Module):

    def __init__(self, dim, down=True, first=False, levels=3, bn=False):
        super(ResnetBlock, self).__init__()
        blocks = []
        for i in range(levels):
            blocks.append(Block(dim=dim, bn=bn))
        self.res = nn.Sequential(
            *blocks
        ) if not first else None
        self.downsample_layer = nn.Sequential(
            nn.InstanceNorm2d(dim, eps=1e-6),# if not bn else nn.BatchNorm2d(dim, eps=1e-6),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2)
        ) if down else None
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, 64, kernel_size=7),
            nn.InstanceNorm2d(64, eps=1e-6)
        ) if first else None

    def forward(self, x):
        if self.stem is not None:
            out = self.stem(x)
            return out
        out = x + self.res(x)
        if self.downsample_layer is not None:
            out = self.downsample_layer(out)
        return out


class Block(nn.Module):

    def __init__(self, dim, bn=False):
        super(Block, self).__init__()

        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim),# if not bn else nn.BatchNorm2d(dim, eps=1e-6),
                       nn.LeakyReLU()]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim)]# if not bn else nn.BatchNorm2d(dim, eps=1e-6)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PALayer(nn.Module):

    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.LeakyReLU(inplace=False) if relu else None
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) 
        #if not bn else nn.BatchNorm2d(  out_planes, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        if self.relu is not None:
            x = self.relu(x)
        x = self.pad(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class Fusion_Block(nn.Module):

    def __init__(self, channel, bn=False, res=False):
        super(Fusion_Block, self).__init__()
        self.bn = nn.InstanceNorm2d(channel, eps=1e-5, momentum=0.01, affine=True) 
        # if not bn else nn.BatchNorm2d( channel, eps=1e-5, momentum=0.01, affine=True)
        self.merge = nn.Sequential(
            ConvBlock(channel, channel, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        ) if not res else None
        self.block = ResnetBlock(channel, down=False, levels=2, bn=bn) if res else None
        self.mix = Mix()

    def forward(self, o, s):
        o_bn = self.bn(o) if self.bn is not None else o
        x = 2 * self.mix(o_bn, s)
        if self.merge is not None:
            x = self.merge(x)
        if self.block is not None:
            x = self.block(x)
        return x

class AdaptiveFusion(nn.Module):
    def __init__(self, high_channel, low_channel):
        super(AdaptiveFusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion_factor = nn.Sequential(
            nn.Conv2d(high_channel + low_channel, 1, kernel_size=1),  # 输入通道为 high_channel + low_channel
            nn.Sigmoid()
        )

    def forward(self, high_feature, low_feature):
        combined = torch.cat([high_feature, low_feature], dim=1)  # 拼接后的通道数为 high_channel + low_channel
        avg_pooled = self.global_pool(combined)
        mix_factor = self.fusion_factor(avg_pooled)  # 得到融合因子
        out = high_feature * mix_factor + low_feature * (1 - mix_factor)
        return out

class FE_Block(nn.Module):

    def __init__(self, plane1, plane2, res=True):
        super(FE_Block, self).__init__()
        
        self.dsc = ConvBlock(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1, relu=False)

        self.merge = nn.Sequential(
            ConvBlock(plane2, plane2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        ) if not res else None
        self.block = ResnetBlock(plane2, down=False, levels=2) if res else None

    def forward(self, p, s):
        
        x = s + self.dsc(p)
        if self.merge is not None:
            x = self.merge(x)
        if self.block is not None:
            x = self.block(x)
        return x


class Iter_Downsample(nn.Module):
    def __init__(self, ):
        super(Iter_Downsample, self).__init__()
        self.ds1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.ds2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        return x, x1, x2


class CALayer(nn.Module):

    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ConvGroups(nn.Module):

    def __init__(self, in_planes, bn=False):
        super(ConvGroups, self).__init__()
        self.iter_ds = Iter_Downsample()
        self.lcb1 = nn.Sequential(
            ConvBlock(in_planes, 16, kernel_size=(3, 3), padding=1), ConvBlock(16, 16, kernel_size=1, stride=1),
            ConvBlock(16, 16, kernel_size=(3, 3), padding=1, bn=bn),
            ConvBlock(16, 64, kernel_size=1, bn=bn, relu=False))
        self.lcb2 = nn.Sequential(
            ConvBlock(in_planes, 32, kernel_size=(3, 3), padding=1), ConvBlock(32, 32, kernel_size=1),
            ConvBlock(32, 32, kernel_size=(3, 3), padding=1), ConvBlock(32, 32, kernel_size=1, stride=1, bn=bn),
            ConvBlock(32, 32, kernel_size=(3, 3), padding=1, bn=bn),
            ConvBlock(32, 128, kernel_size=1, bn=bn, relu=False))
        self.lcb3 = nn.Sequential(
            ConvBlock(in_planes, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 64, kernel_size=1, bn=bn),
            ConvBlock(64, 64, kernel_size=(3, 3), padding=1, bn=bn),
            ConvBlock(64, 256, kernel_size=1, bn=bn, relu=False))

    def forward(self, x):
        img1, img2, img3 = self.iter_ds(x)
        s1 = self.lcb1(img1)
        s2 = self.lcb2(img2)
        s3 = self.lcb3(img3)
        return s1, s2, s3


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=8, r=2, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.InstanceNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


def padding_image(image, h, w):
    assert h >= image.size(2)
    assert w >= image.size(3)
    padding_top = (h - image.size(2)) // 2
    padding_down = h - image.size(2) - padding_top
    padding_left = (w - image.size(3)) // 2
    padding_right = w - image.size(3) - padding_left
    out = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top, padding_down), mode='reflect')
    return out, padding_left, padding_left + image.size(3), padding_top, padding_top + image.size(2)\

import torch.nn.functional as F
def resize_and_pad_image(image, target_h, target_w):
    # Calculate the scaling factor to resize the image
    h, w = image.shape[2], image.shape[3]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize the image
    image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2

    # Apply padding
    padded_image = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    return padded_image, pad_w, pad_w + new_w, pad_h, pad_h + new_h