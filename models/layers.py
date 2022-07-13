import torch
import torch.nn as nn


# from models.MRDNet import DWT, IWT


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2  # 填充1/2卷积核大小
        layers = list()
        if transpose:  # 如果转置的话，就执行下面的
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:  # 根据默认设置，执行下面句子
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                          bias=bias))  # 一般情况下，padding=1
        if norm:  # 如果进行归一化，就执行下面的
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:  # 默认执行下面句子
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)
        # main的容器中是如下两个操作
        # nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
        # nn.ReLU(inplace=True)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):  # 残差网络
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(  # 两个卷积层
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 最后不能有relu，残差网络
        )

    def forward(self, x):
        return self.main(x) + x  # +x，残差块，不会出现梯度消失的现象


'''
网络中加入小波变换
'''


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2  # 4,3,128,256
    x02 = x[:, :, 1::2, :] / 2  # 4,3,128,256
    x1 = x01[:, :, :, 0::2]  # 4,3,128,128
    x2 = x02[:, :, :, 0::2]  # 4,3,128,128
    x3 = x01[:, :, :, 1::2]  # 4,3,128,128
    x4 = x02[:, :, :, 1::2]  # 4,3,128,128
    x_LL = x1 + x2 + x3 + x4  # 4,3,128,128
    x_HL = -x1 - x2 + x3 + x4  # 4,3,128,128
    x_LH = -x1 + x2 - x3 + x4  # 4,3,128,128
    x_HH = x1 - x2 - x3 + x4  # 4,3,128,128

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
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


'''
小波变换残差块，将小波变换和小波逆变换加入到残差块中，进行高频特征的提取
'''


class Wavelet_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Wavelet_ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.DWT = DWT()
        self.IWT = IWT()

        self.Conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )

    def forward(self, x):  # 4,32,256,256
        res2 = self.DWT(x)  # 16,32,128,128
        res2 = self.Conv(res2)  # 16,32,128,128
        res2 = self.IWT(res2)  # 4,32,256,256
        return self.main(x) + res2 + x
