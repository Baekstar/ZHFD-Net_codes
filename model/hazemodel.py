import torch
import torch.nn as nn
import torch.nn.functional as F
from .highmerge import PALayer, ConvGroups, FE_Block, Fusion_Block, ResnetBlock, ConvBlock, CALayer, SKConv,fusion_h,Conv1x1,AdaptiveFusion
from .Parameter_test import atp_cal,Dense
from .Residual_Enhancer import DFREModule,DualDepthwiseSeparableConvModule
from .Retinex_enhance import RetinexFormer

def make_model():
    return MainModel()


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class InDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 9, stride=4, padding=4, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride = 4, padding=3,  bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=64, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
        self.convf = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=64, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
    def forward(self, x):
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        xR = torch.unsqueeze(self.conv(R), 1)
        xG = torch.unsqueeze(self.conv(G), 1)
        xB = torch.unsqueeze(self.conv(B), 1)
        x = torch.cat([xR, xG, xB], 1)
        x, _ = torch.min(x, dim=1)
        return self.convf(x)

class SKConv1(nn.Module):
    def __init__(self, outfeatures=64, infeatures=1, M=4 ,L=32):

        super(SKConv1, self).__init__()
        self.M = M
        self.convs = nn.ModuleList([])
        in_conv = InConv(in_ch=infeatures, out_ch=outfeatures)
        for i in range(M):
            if i==0:
                self.convs.append(in_conv)
            else:
                self.convs.append(nn.Sequential(
                    nn.Upsample(scale_factor=1/(2**i), mode='bilinear', align_corners=True),
                    in_conv,
                    nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
                ))
        self.fc = nn.Linear(outfeatures, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(L, outfeatures)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


# class estimation(nn.Module):
#     def __init__(self):
#         super(estimation, self).__init__()
                
#         self.InConv = SKConv1(outfeatures=64, infeatures=1, M=3 ,L=32)

#         self.convt = DoubleConv(64, 64)
#         self.OutConv = nn.Conv2d(64, 1, 3, padding = 1, stride=1, bias=False, padding_mode='reflect')
#         self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

#         self.conv1 = InDoubleConv(3, 64)
#         self.conv2 = DoubleConv(64, 64)
#         self.maxpool = nn.MaxPool2d(15, 7)
        
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.dense = nn.Linear(64, 3, bias=False)
        
        
#     def forward(self, x):

#         xmin = self.InConv(x)

#         trans = self.OutConv(self.up(self.convt(xmin)))
#         trans = torch.sigmoid(trans) + 1e-12

#         atm = self.conv1(x)
#         atm = torch.mul(atm, xmin)
#         atm = self.pool(self.conv2(self.maxpool(atm)))
#         atm = atm.view(-1, 64)
#         atm = torch.sigmoid(self.dense(atm))
        
#         return trans, atm

class estimation(nn.Module):
    def __init__(self):
        super(estimation, self).__init__()
                
        self.InConv = SKConv1(outfeatures=64, infeatures=1, M=3 ,L=32)
        # 新增MixStructureBlock用于强化特征提取
        self.mix_block1 = MixStructureBlock(64)

        self.convt = DoubleConv(64, 64)
        self.OutConv = nn.Conv2d(64, 1, 3, padding = 1, stride=1, bias=False, padding_mode='reflect')
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
 
        

        self.conv1 = InDoubleConv(3, 64)
        # Channel Attention
        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        self.conv2 = DoubleConv(64, 64)
        # Channel Attention
        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.maxpool = nn.MaxPool2d(15, 7)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(64, 3, bias=False)
        # Channel Attention
        self.ca3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 3, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(3, 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # 增加MixStructureBlock用于大气光分支增强
        # self.mix_block3 = MixStructureBlock(3) 
         
        
        
    def forward(self, x):

        # InConv 提取初始特征
        xmin = self.InConv(x)
        # 通过MixStructureBlock增强特征
        xmin = self.mix_block1(xmin)  

        #  convt双卷积模块（DoubleConv），对输入特征进行非线性变换
        # OutConv: 使用卷积层将通道数从 64 压缩到 1，输出透过率图像
        # up: 上采样层，放大分辨率到原始图像大小（放大 4 倍
        trans = self.OutConv(self.up(self.convt(xmin)))
        #  torch.sigmoid 将输出限制在 [0, 1] 范围
        trans = torch.sigmoid(trans) + 1e-12

        # 双卷积模块（InDoubleConv），从原始输入提取大气光的初始特征
        atm = self.conv1(x)
        #  提取通道级注意力权重，并与原始特征逐元素相乘（增强有效特征）
        atm=self.ca1(atm)*atm
        # 加权后的特征与透过率分支的特征xmin相乘，融合两条分支的信息
        atm = torch.mul(atm, xmin)
        atm=self.conv2(self.maxpool(atm))
        #  下采样特征，并通过 self.ca2 提取通道注意力后进行加权
        atm=atm*self.ca2(atm)
        # 全局平均池化将特征降维至通道描述符
        atm = self.pool(atm)
        atm = atm.view(-1, 64)
        # 使用全连接层将 64 维特征映射到 3 维
        atm=self.dense(atm)
        if atm.dim() == 2:  # 如果张量是2D，扩展为4D
            atm = atm.unsqueeze(2).unsqueeze(3)  # 增加高度和宽度维度
        #  生成全局通道注意力权重，并与特征逐元素相乘，最终输出大气光值
        atm = torch.sigmoid(self.ca3(atm)*atm)
        # 检查并调整 atp 的维度
        atm = atm.squeeze(-1).squeeze(-1)  # 去除多余维度，使其变为4D

        # # 不加CA
        # atm = self.conv1(x)
        # atm = torch.mul(atm, xmin)
        # atm = self.pool(self.conv2(self.maxpool(atm)))
        # atm = atm.view(-1, 64)
        # atm = torch.sigmoid(self.dense(atm))


        
        return trans, atm



# MixDehazeNet的MixDehazeNet
class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x




# class MainModel(nn.Module):
#     def __init__(self):
#         super().__init__()
                
#         self.estimation = estimation()		
        
#     def forward(self, x, flag):

#         trans, atm = self.estimation(x)

#         atm = torch.unsqueeze(torch.unsqueeze(atm, 2), 2)
#         atm = atm.expand_as(x)
#         trans = trans.expand_as(x)

#         if flag == 'train':
#             out = (x - (1 - trans)*atm)/trans
#             return trans, atm, out
#         elif flag == 'test':
#             return trans, atm

class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out



class MainModel(nn.Module):
    def __init__(self, ngf=64, bn=False):
        super(MainModel, self).__init__()
        # 下采样
        self.down1 = ResnetBlock(3, first=True)
        self.ddscm1 = DualDepthwiseSeparableConvModule(ngf, ngf)  # 加入 DDSCM
        # 加SCConv
        self.scconv1=SCBottleneck(ngf,ngf)
        self.down2 = ResnetBlock(ngf, levels=2)
        self.ddscm2 = DualDepthwiseSeparableConvModule(ngf * 2, ngf * 2)  # 加入 DDSCM
        # 加SCConv
        self.scconv2=SCBottleneck(ngf*2,ngf*2)
        self.down3 = ResnetBlock(ngf * 2, levels=2, bn=bn)
        self.ddscm3 = DualDepthwiseSeparableConvModule(ngf * 4, ngf * 4)  # 加入 DDSCM
        # 加SCConv
        self.scconv3=SCBottleneck(ngf*4,ngf*4)

        self.down1_high = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(3, ngf, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(ngf),
                                        nn.ReLU(True),
                                        SCBottleneck(ngf,ngf))

        self.down2_high = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True),
                                        SCBottleneck(ngf*2,ngf*2))

        self.down3_high = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 4),
                                        nn.ReLU(True),
                                        SCBottleneck(ngf*4,ngf*4))

        self.res = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.res_atp = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.res_tran = nn.Sequential(
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
            # SCBottleneck(ngf * 4,ngf * 4 ),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2),
            CALayer(ngf * 2),
            PALayer(ngf * 2))

        self.up2 = nn.Sequential(
            nn.LeakyReLU(True),
            # SCBottleneck(ngf * 2,ngf * 2 ),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            CALayer(ngf),
            PALayer(ngf))

        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            # SCBottleneck(ngf,ngf ),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh())
        
        # # 加多头注意力
        # self.mchead=mscheadv5(3)

        self.info_up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2, eps=1e-5),
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
        self.adaptive_fusion1 = AdaptiveFusion(ngf * 4,ngf * 4)  # 用于 x6 和 fuse3 的融合
        self.adaptive_fusion2 = AdaptiveFusion(ngf * 2,ngf * 2)  # 用于 fuse_up2 和 x_down2 的融合
        self.adaptive_fusion3 = AdaptiveFusion(ngf,ngf)      # 用于 fuse_up3 和 x_down1 的融合


        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.estimation=estimation()
        self.atp = atp_cal()
        self.tran = Dense()

        # 图像增强
        # self.enhancement_model=  RetinexFormer()

        # 添加 DFRE 模块
        # self.dfre = DFREModule(in_channels=3)  # DFRE 模块的输入通道数设置为 3

    def forward(self, hazy, high,flag):
        ###############   high-frequency encode   ###################
        # high = self.ffa(high)

        x_down1_high = self.down1_high(high)  # [bs, 64, 256, 256]

        x_down2_high = self.down2_high(x_down1_high)  # [bs, 128, 128, 128]

        x_down3_high = self.down3_high(x_down2_high)  # [bs, 256, 64, 64]

        ###############   hazy encode   ###################

        x_down1 = self.down1(hazy)  # [bs, ngf, ngf * 4, ngf * 4]
        x_down1 = self.ddscm1(x_down1)  # 应用 DDSCM
        # 加SCconv
        x_down1 =self.scconv1(x_down1)

        att1 = self.att1(x_down1_high, x_down1)

        x_down2 = self.down2(x_down1)  # [bs, ngf*2, ngf*2, ngf*2]
        x_down2 = self.ddscm2(x_down2)  # 应用 DDSCM
        # 加SCconv
        x_down2 =self.scconv2(x_down2)

        att2 = self.att2(x_down2_high, x_down2)
        fuse2 = self.fam2(att1, att2)

        x_down3 = self.down3(x_down2)  # [bs, ngf * 4, ngf, ngf]
        x_down3 = self.ddscm3(x_down3)  # 应用 DDSCM
        # 加SCconv
        x_down3=self.scconv3(x_down3)

        att3 = self.att3(x_down3_high, x_down3)
        fuse3 = self.fam3(fuse2, att3)

        ###############   dehaze   ###################

        x6 = self.res(x_down3)

        fuse_up2 = self.info_up1(fuse3)
        fuse_up2 = self.merge2(fuse_up2 + x_down2)
        # 自适应融合
        # fuse_up2 = self.merge2(self.adaptive_fusion2(fuse_up2, x_down2))

        fuse_up3 = self.info_up2(fuse_up2)
        fuse_up3 = self.merge3(fuse_up3 + x_down1)
        # 自适应融合
        # fuse_up3 = self.merge3(self.adaptive_fusion3(fuse_up3, x_down1))

        x_up2 = self.up1(x6 + fuse3)

        x_up3 = self.up2(x_up2 + fuse_up2)

        x_up4 = self.up3(x_up3 + fuse_up3)

        # 加多头注意力
        # x_up4 = self.mchead(x_up4)

        # 添加 DFRE 模块
        # x_up4=self.dfre(x_up4_1) 

        ###############   atp   ###################
        tran, atp = self.estimation(x_up4)
        atp = torch.unsqueeze(torch.unsqueeze(atp, 2), 2)
        atp = atp.expand_as(x_up4)
        tran = tran.expand_as(x_up4)
       
        # ##############  Atmospheric scattering model  #################

        zz = torch.abs((tran)) + (10 ** -10)  # t
        
        haze = (x_up4 * zz) + atp * (1 - zz)
        dehaze = (hazy - atp) / zz + atp  # 去雾公式

        # # 去雾后的图像进行增强
        # scale=0.05
        # enhance=self.enhancement_model(dehaze)
        # dehaze=(dehaze+enhance*scale)/(1+scale)


        if flag == 'train':
            out=x_up4
            # return zz, atp, dehaze
            return tran, atp, dehaze
        elif flag == 'test':
            # return zz, atp
            return tran, atp
        # return haze, dehaze, x_up4, tran, atp
