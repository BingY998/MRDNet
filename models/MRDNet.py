import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [Wavelet_ResBlock(out_channel, out_channel) for _ in range(num_res)]  # 8次残差，蓝色块

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [Wavelet_ResBlock(channel, channel) for _ in range(num_res)]  # 8次残差
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MFCF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFCF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),  # 1x1卷积
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 3x3卷积
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SFE(nn.Module):  # 浅卷积模块--从下采样图像中提取特征
    def __init__(self, out_plane):  # 浅卷积 # 调用就会执行这个，并且需要输出的通道数
        super(SFE, self).__init__()
        self.main = nn.Sequential(  # 顺序容器，按照加入的顺序构建单路径网络
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),  # 普通的卷积+relu
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True)  # 此1x1卷积的特征和输入Bk连接
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)  # 进一步细化连接之后的特征

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):  # 继承Module类  #特征注意模块--主动强调或抑制先前尺度的特征，并从SCM中学习特征的空间/通道重要性
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)  # 初始化属性,小型残差网络

    def forward(self, x1, x2):
        x = x1 * x2  # x1=EBout,x2=SCMout
        out = x1 + self.merge(x)  # EBout和卷积之后的特征进行融合
        return out


'''
网络中加入小波变换
'''


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2#4,3,128,256
    x02 = x[:, :, 1::2, :] / 2#4,3,128,256
    x1 = x01[:, :, :, 0::2]#4,3,128,128
    x2 = x02[:, :, :, 0::2]#4,3,128,128
    x3 = x01[:, :, :, 1::2]#4,3,128,128
    x4 = x02[:, :, :, 1::2]#4,3,128,128
    x_LL = x1 + x2 + x3 + x4 #4,3,128,128
    x_HL = -x1 - x2 + x3 + x4#4,3,128,128
    x_LH = -x1 + x2 - x3 + x4#4,3,128,128
    x_HH = x1 - x2 - x3 + x4#4,3,128,128

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# def dwt_init(x):
#     x01 = x[:, :, 0::2, :] / 2#4,3,128,256
#     x02 = x[:, :, 1::2, :] / 2#4,3,128,256
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)
#
#
# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     # print([in_batch, in_channel, in_height, in_width])
#     out_batch, out_channel, out_height, out_width = in_batch, int(
#         in_channel / (r ** 2)), r * in_height, r * in_width
#     x1 = x[:, 0:out_channel, :, :] / 2
#     x2 = x[:, out_channel:out_channel * 2, :, :] / 2
#     x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
#     x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h




# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# BA模块---里面包含MKSP(多内核条带池模块)+AR(注意力细化模块)---提取全局和局部模糊信息
class RSAM(nn.Module):  # 继承自Module类，里面包括各种深度学习方法
    def __init__(self, inplanes, outplanes):
        super(RSAM, self).__init__()  # 对继承自父类的属性初始化
        midplanes = int(outplanes // 2)  # 将输出特征通道数减半？？？BA模块中也进行了下采样，减小计算量吗?

        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))  # 二元自适应均值池化层,HxW形式,H=None意味着H大小与输入的大小相同。
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))  # W=None意味着W大小与输入的大小相同
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)  # 高--二维卷积不添加偏置项
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)  # 宽--二维卷积不添加偏置项
        # padding=(1,0),左右各填充一列，上下不填充
        self.pool_3_h = nn.AdaptiveAvgPool2d((None, 3))
        self.pool_3_w = nn.AdaptiveAvgPool2d((3, None))
        self.conv_3_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)  # padding=1上下左右都填充
        self.conv_3_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_5_h = nn.AdaptiveAvgPool2d((None, 5))
        self.pool_5_w = nn.AdaptiveAvgPool2d((5, None))
        self.conv_5_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_5_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_7_h = nn.AdaptiveAvgPool2d((None, 7))
        self.pool_7_w = nn.AdaptiveAvgPool2d((7, None))
        self.conv_7_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_7_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        # 进行特征融合，一共有四个输出midplanes，输出的通道数*4作为特征融合的输入：
        # 第一次卷积，特征融合
        self.fuse_conv = nn.Conv2d(midplanes * 4, midplanes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        # 第二次卷积，输出通道数增加一倍，变回原来的大小(BA模块处理前的大小)
        self.conv_final = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        # 下面是AR模块的卷积过程(两次卷积)--输出的是掩码，细化的过程
        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)

    def forward(self, x):  # 将输入层、网络层、输出层连接起来，实现信息的前向传导
        _, _, h, w = x.size()
        # print("x.size()", x.size())  # 输出结果:x.size() torch.Size([1, 256, 360, 640])

        x_1_h = self.pool_1_h(x)  # 高度--池化层结果
        x_1_h = self.conv_1_h(x_1_h)  # 高度--卷积结果
        x_1_h = x_1_h.expand(-1, -1, h, w)  # 扩展到h,w
        # print("x_1_h", x_1_h)
        # x1 = F.interpolate(x1, (h, w))

        x_1_w = self.pool_1_w(x)  # 宽度--池化层结果
        x_1_w = self.conv_1_w(x_1_w)  # 宽度--卷积结果
        x_1_w = x_1_w.expand(-1, -1, h, w)
        # x2 = F.interpolate(x2, (h, w))

        x_3_h = self.pool_3_h(x)
        x_3_h = self.conv_3_h(x_3_h)
        x_3_h = F.interpolate(x_3_h, (h, w))  # 上采样到和内核为1时的相同大小

        x_3_w = self.pool_3_w(x)
        x_3_w = self.conv_3_w(x_3_w)
        x_3_w = F.interpolate(x_3_w, (h, w))

        x_5_h = self.pool_5_h(x)
        x_5_h = self.conv_5_h(x_5_h)
        x_5_h = F.interpolate(x_5_h, (h, w))

        x_5_w = self.pool_5_w(x)
        x_5_w = self.conv_5_w(x_5_w)
        x_5_w = F.interpolate(x_5_w, (h, w))

        x_7_h = self.pool_7_h(x)
        x_7_h = self.conv_7_h(x_7_h)
        x_7_h = F.interpolate(x_7_h, (h, w))

        x_7_w = self.pool_7_w(x)
        x_7_w = self.conv_7_w(x_7_w)
        x_7_w = F.interpolate(x_7_w, (h, w))  # 进行上采样

        # 上采样之后，特征融合，进行张量拼接,维度为1
        hx = self.relu(self.fuse_conv(torch.cat((x_1_h + x_1_w, x_3_h + x_3_w, x_5_h + x_5_w, x_7_h + x_7_w), dim=1)))
        mask_1 = self.conv_final(hx).sigmoid()  # 反向传播，输出掩码
        out1 = x * mask_1  # 经过MKSP之后的输出

        hx = self.mask_relu(self.mask_conv_1(out1))
        mask_2 = self.mask_conv_2(hx).sigmoid()  # 经过AR模块，反向传播，输出掩码
        hx = out1 * mask_2  # 经过AR之后的输出

        return hx


class MRDNet(nn.Module):
    def __init__(self, num_res=8):
        super(MRDNet, self).__init__()

        base_channel = 32

        # self.DWT = DWT()  # 小波变换
        # self.IWT = IWT()  # 小波逆变换
        self.Encoder = nn.ModuleList([  # 编码器，8个残差模块
            EBlock(base_channel, num_res),  # base_channel是输出通道数，8个残差连接
            EBlock(base_channel * 2, num_res),  # 通道数扩大二倍
            EBlock(base_channel * 4, num_res),  # 通道数扩大四倍
        ])
        """
        加BA模块
        """
        self.en_layer1 = nn.Sequential(  # 改变通道数，同时下采样一半
            BasicConv(base_channel, base_channel * 4, kernel_size=3, relu=True, stride=2)

        )

        self.en_layer2 = nn.Sequential(  # 改变通道数，同时下采样一半
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2)

        )
        # Blur-aware Attention #模糊感知注意力
        self.BA = RSAM(base_channel * 8, base_channel * 8)  # 经过BA模块之后，通道数不变，将BA_Block类的计算方法赋给了self.BA  4,256,64,64

        self.de_layer3 = nn.Sequential(  # 改变通道数，同时上采样
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2, transpose=True)
        )

        self.de_layer4 = nn.Sequential(  # 改变通道数，同时上采样
            BasicConv(base_channel * 4, base_channel, kernel_size=4, relu=True, stride=2, transpose=True)
        )

        self.feat_extract = nn.ModuleList([  # 特征提取，利用浅卷积，Unet
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),  # 第一层卷积，将通道数从3变成32
            # 直接特征提取了，省去了通道数从3到32的这步卷积操作
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),  # 下采样
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),  # 下采样
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            # 转置卷积，上采样
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),  # 转置卷积，上采样
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([  # 解码器--蓝色块
            DBlock(base_channel * 4, num_res),  # 每一层都进行解码
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([  # 不改变大小，改变通道数，每一层都经历这个过程，从DB3-DB2，DB2-DB1一共两次
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(  # 经历此卷积层进行输出,应该一共三次
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),  # 通道数从128-->3
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),  # 通道数从64-->3
            ]
        )

        self.AFFs = nn.ModuleList([  # 两次的非对称特征融合
            MFCF(base_channel * 7, base_channel * 1),  # 7倍变成一倍，32+32*2+32*4=7*32
            MFCF(base_channel * 7, base_channel * 2)  # 7倍变成两倍
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SFE(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SFE(base_channel * 2)

    def forward(self, x):  # x是输入#4,3,256,256

        # 在RGB图上进行下采样的
        x_2 = F.interpolate(x, scale_factor=0.5)  # 下采样，第二层#4,3,128,128
        x_4 = F.interpolate(x_2, scale_factor=0.5)  # 下采样，最小尺寸的，第三层#4,3,64,64

        z2 = self.SCM2(x_2)  # 4,64,128,128----16,64,64,64
        z4 = self.SCM1(x_4)  # 4,128,64,64----16,128,32,32

        # '''加入小波变换'''
        # x1 = x
        # x = self.DWT(x) # 16,3,128,128
        # z2 = self.DWT(x_2)  # 16,3,64,64
        # z4 = self.DWT(x_4) # 16,3,32,32

        # z2 = self.SCM2(z2)  # 4,64,128,128----16,64,64,64
        # z4 = self.SCM1(z4)  # 4,128,64,64----16,128,32,32


        outputs = list()
        x_ = self.feat_extract[0](x)  # 4,32,256,256 # 第一层的特征提取
        res1 = self.Encoder[0](x_)  # 4,32,256,256 # 第一层编码

        z = self.feat_extract[1](res1)  # 4,64,128,128  # 第二层的，对第一层的结果进行特征提取
        z = self.FAM2(z, z2)  # 4,64,128,128  # 第二层的特征注意力模块，已经经历过SCM了
        res2 = self.Encoder[1](z)  # 4,64,128,128  # 第二层编码

        z = self.feat_extract[2](res2)  # 4,128,64,64  # 第三层，对第二层的结果进行特征提取
        z = self.FAM1(z, z4)  # 4,128,64,64  # 第三层的特征注意力模块
        z = self.Encoder[2](z)  # 4,128,64,64  # 第三层编码
        # a = self.IWT(z)

        z12 = F.interpolate(res1, scale_factor=0.5)  # 4,32,128,128  # 第一层下采样，输入到AFF2中，大小到(h/2)x(w/2)
        z21 = F.interpolate(res2, scale_factor=2)  # 4,64,256,256  # 中间层的，上采样一次恢复原始大小hxw
        z42 = F.interpolate(z, scale_factor=2)  # 4,128,128,128  # 最底层的，上采样一次，输入到AFF2中，大小到(h/2)x(w/2)
        z41 = F.interpolate(z42, scale_factor=2)  # 4,128,256,256  # 最底层的，上采样两次恢复原始大小hxw

        res2 = self.AFFs[1](z12, res2, z42)  # 第二层#4,64,128,128#最后通过AFF时进行融合，7*base_channels
        res1 = self.AFFs[0](res1, z21, z41)  # 第一层#4,32,256,256
        # res1 = self.IWT(res1)#4,32,256,256
        # res2 = self.IWT(res2)#4,64,128,128

        '''
        加入RSAM模块(模糊感知模块)--加在MFCF1之后
        原始MIMO-UNet中，EB1输出tensor为：4,32,256,256 后面输入tensor为：4,32,256,256
        EB1输出需要下采样并扩大通道数
        原始BANet中，输入tensor为：8,256,64,64  输出tensor为：8,256,64,64
        输出tensor需要上采样并缩小通道数
        '''

        res1 = self.en_layer1(res1)  # 4,128,128,128 # 经过第一层卷积，通道数改变到64
        in_feature = self.en_layer2(res1)  # 4,256,64,64

        BA_out = self.BA(in_feature)  # 4,256,64,64

        res1 = self.de_layer3(BA_out)  # 4,128,128,128
        res1 = self.de_layer4(res1)  # 4,32,256,256

        # z = self.DWT(a)
        z = self.Decoder[0](z)  # 4,128,64,64----16,128,32,32
        # z = self.IWT(z)  # 最底层加入小波逆变换 4,128,64,64

        z_ = self.ConvsOut[0](z)  # 4,3,64,64
        z = self.feat_extract[3](z)  # 4,64,128,128
        outputs.append(z_ + x_4)  # 最底层的输出

        z = torch.cat([z, res2], dim=1)#4,128,128,128
        # z = self.DWT(z)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        # z = self.IWT(z)  # 中间层加入小波变换#4,64,128,128
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)  # 4,32,256,256

        outputs.append(z_ + x_2)  # 中间层的输出

        z = torch.cat([z, res1], dim=1)  # 4,64,256,256
        # z = self.DWT(z)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)

        # z = self.IWT(z)  # 最上层加入小波变换
        z = self.feat_extract[5](z)  # 4,3,256,256
        outputs.append(z + x)  # 最上层的输出

        return outputs


'''
此部分原来在outputs = list()后面
        x1 = self.en_layer1(x)  # 4,64,256,256 # 经过第一层卷积，通道数改变到64
        x1 = self.en_layer2(x1)  # 4,128,256,256
        x1 = self.en_layer3(x1)  # 4,128,64,64
        x1 = self.BAM_1(x1)  # 经历BAM模块
        x1 = self.BAM_2(x1)  # 4,256,64,64
        x1 = self.BAM_3(x1)

        x1 = F.interpolate(x1, scale_factor=4)  # 4,256,256,256
        x1 = self.en_layer4(x1)  # 4,32,256,256
'''


class MRDNetPlus(nn.Module):
    def __init__(self, num_res=20):
        super(MRDNetPlus, self).__init__()
        base_channel = 32

        # self.DWT = DWT()  # 小波变换
        # self.IWT = IWT()  # 小波逆变换
        self.Encoder = nn.ModuleList([  # 编码器，20个残差模块
            EBlock(base_channel, num_res),  # base_channel是输出通道数，20个残差连接
            EBlock(base_channel * 2, num_res),  # 通道数扩大二倍
            EBlock(base_channel * 4, num_res),  # 通道数扩大四倍
        ])
        """
        加RSAM模块
        """
        self.en_layer1 = nn.Sequential(  # 改变通道数，同时下采样一半
            BasicConv(base_channel, base_channel * 4, kernel_size=3, relu=True, stride=2)

        )

        self.en_layer2 = nn.Sequential(  # 改变通道数，同时下采样一半
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2)

        )
        # Blur-aware Attention #模糊感知注意力
        self.BA = RSAM(base_channel * 8, base_channel * 8)  # 经过BA模块之后，通道数不变，将BA_Block类的计算方法赋给了self.BA  4,256,64,64

        self.de_layer3 = nn.Sequential(  # 改变通道数，同时上采样
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2, transpose=True)
        )

        self.de_layer4 = nn.Sequential(  # 改变通道数，同时上采样
            BasicConv(base_channel * 4, base_channel, kernel_size=4, relu=True, stride=2, transpose=True)
        )

        self.feat_extract = nn.ModuleList([  # 特征提取，利用浅卷积，Unet
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),  # 第一层卷积，将通道数从3变成32
            # 直接特征提取了，省去了通道数从3到32的这步卷积操作
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),  # 下采样
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),  # 下采样
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            # 转置卷积，上采样
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),  # 转置卷积，上采样
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([  # 解码器--蓝色块
            DBlock(base_channel * 4, num_res),  # 每一层都进行解码
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([  # 不改变大小，改变通道数，每一层都经历这个过程，从DB3-DB2，DB2-DB1一共两次
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(  # 经历此卷积层进行输出,应该一共三次
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),  # 通道数从128-->3
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),  # 通道数从64-->3
            ]
        )

        self.AFFs = nn.ModuleList([  # 两次的非对称特征融合
            MFCF(base_channel * 7, base_channel * 1),  # 7倍变成一倍，32+32*2+32*4=7*32
            MFCF(base_channel * 7, base_channel * 2)  # 7倍变成两倍
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SFE(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SFE(base_channel * 2)

    def forward(self, x):  # x是输入#4,3,256,256

        # 在RGB图上进行下采样的
        x_2 = F.interpolate(x, scale_factor=0.5)  # 下采样，第二层#4,3,128,128
        x_4 = F.interpolate(x_2, scale_factor=0.5)  # 下采样，最小尺寸的，第三层#4,3,64,64

        z2 = self.SCM2(x_2)  # 4,64,128,128----16,64,64,64
        z4 = self.SCM1(x_4)  # 4,128,64,64----16,128,32,32

        # '''加入小波变换'''
        # x1 = x
        # x = self.DWT(x) # 16,3,128,128
        # z2 = self.DWT(x_2)  # 16,3,64,64
        # z4 = self.DWT(x_4) # 16,3,32,32

        # z2 = self.SCM2(z2)  # 4,64,128,128----16,64,64,64
        # z4 = self.SCM1(z4)  # 4,128,64,64----16,128,32,32

        outputs = list()
        x_ = self.feat_extract[0](x)  # 4,32,256,256 # 第一层的特征提取
        res1 = self.Encoder[0](x_)  # 4,32,256,256 # 第一层编码

        z = self.feat_extract[1](res1)  # 4,64,128,128  # 第二层的，对第一层的结果进行特征提取
        z = self.FAM2(z, z2)  # 4,64,128,128  # 第二层的特征注意力模块，已经经历过SCM了
        res2 = self.Encoder[1](z)  # 4,64,128,128  # 第二层编码

        z = self.feat_extract[2](res2)  # 4,128,64,64  # 第三层，对第二层的结果进行特征提取
        z = self.FAM1(z, z4)  # 4,128,64,64  # 第三层的特征注意力模块
        z = self.Encoder[2](z)  # 4,128,64,64  # 第三层编码
        # a = self.IWT(z)

        z12 = F.interpolate(res1, scale_factor=0.5)  # 4,32,128,128  # 第一层下采样，输入到AFF2中，大小到(h/2)x(w/2)
        z21 = F.interpolate(res2, scale_factor=2)  # 4,64,256,256  # 中间层的，上采样一次恢复原始大小hxw
        z42 = F.interpolate(z, scale_factor=2)  # 4,128,128,128  # 最底层的，上采样一次，输入到AFF2中，大小到(h/2)x(w/2)
        z41 = F.interpolate(z42, scale_factor=2)  # 4,128,256,256  # 最底层的，上采样两次恢复原始大小hxw

        res2 = self.AFFs[1](z12, res2, z42)  # 第二层#4,64,128,128#最后通过AFF时进行融合，7*base_channels
        res1 = self.AFFs[0](res1, z21, z41)  # 第一层#4,32,256,256
        # res1 = self.IWT(res1)#4,32,256,256
        # res2 = self.IWT(res2)#4,64,128,128

        '''
        加入RSAM模块(模糊感知模块)--加在MFCF1之后
        原始MIMO-UNet中，EB1输出tensor为：4,32,256,256 后面输入tensor为：4,32,256,256
        EB1输出需要下采样并扩大通道数
        原始BANet中，输入tensor为：8,256,64,64  输出tensor为：8,256,64,64
        输出tensor需要上采样并缩小通道数
        '''

        res1 = self.en_layer1(res1)  # 4,128,128,128 # 经过第一层卷积，通道数改变到64
        in_feature = self.en_layer2(res1)  # 4,256,64,64

        BA_out = self.BA(in_feature)  # 4,256,64,64

        res1 = self.de_layer3(BA_out)  # 4,128,128,128
        res1 = self.de_layer4(res1)  # 4,32,256,256

        # z = self.DWT(a)
        z = self.Decoder[0](z)  # 4,128,64,64----16,128,32,32
        # z = self.IWT(z)  # 最底层加入小波逆变换 4,128,64,64

        z_ = self.ConvsOut[0](z)  # 4,3,64,64
        z = self.feat_extract[3](z)  # 4,64,128,128
        outputs.append(z_ + x_4)  # 最底层的输出

        z = torch.cat([z, res2], dim=1)  # 4,128,128,128
        # z = self.DWT(z)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        # z = self.IWT(z)  # 中间层加入小波变换#4,64,128,128
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)  # 4,32,256,256

        outputs.append(z_ + x_2)  # 中间层的输出

        z = torch.cat([z, res1], dim=1)  # 4,64,256,256
        # z = self.DWT(z)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)

        # z = self.IWT(z)  # 最上层加入小波变换
        z = self.feat_extract[5](z)  # 4,3,256,256
        outputs.append(z + x)  # 最上层的输出

        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MRDNetPlus":
        return MRDNetPlus()
    elif model_name == "MRDNet":
        return MRDNet()
    raise ModelError('Wrong Model!\nYou should choose MRDNetPlus or MRDNet.')
