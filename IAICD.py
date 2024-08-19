import torch
import math
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class my_IAICD(nn.Module):
    def __init__(self, in_channels, out_channels, Resolution, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(my_IAICD, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.conv_rgbd = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       )

        if Resolution == 1:
            self.transfer = nn.Sequential(nn.Conv2d(1, 9, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(9),
                                          nn.ReLU(),
                                          )
            self.att = PAM_Module(in_channels, 320, 640)

        elif Resolution == 2:
            self.transfer = nn.Sequential(nn.Conv2d(1, 9, kernel_size=3, stride=2, padding=1),
                                          nn.BatchNorm2d(9),
                                          nn.ReLU(),
                                          )
            self.att = PAM_Module(in_channels, 160, 320)

        elif Resolution == 4:
            self.transfer = nn.Sequential(nn.Conv2d(1, 9, kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1),
                                          nn.BatchNorm2d(9),
                                          nn.ReLU(),
                                          )
            self.att = PAM_Module(in_channels, 80, 160)

        elif Resolution == 8:
            self.transfer = nn.Sequential(nn.Conv2d(1, 9, kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1),
                                          nn.BatchNorm2d(9),
                                          nn.ReLU(),
                                          )
            self.att = PAM_Module(in_channels, 40, 80)

        elif Resolution == 16:
            self.transfer = nn.Sequential(nn.Conv2d(1, 9, kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(9, 9, kernel_size=3, stride=2, padding=1),
                                          nn.BatchNorm2d(9),
                                          nn.ReLU(),
                                          )
            self.att = PAM_Module(in_channels, 20, 40)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.theta = theta
        self.theta = Parameter(torch.zeros(1))

    def forward(self, lidar, guidance, illu):
        # encode input features
        out_normal = self.conv(guidance)

        # calculate 9 centrals
        [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)   # C_out, C_in
        kernel_diff = kernel_diff[:, :, None, None]  # C_out, C_in, 1, 1
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = self.feature_padding(out_normal)
        out_diff0 = F.conv2d(input=x0, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff1 = F.conv2d(input=x1, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff2 = F.conv2d(input=x2, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff3 = F.conv2d(input=x3, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff4 = F.conv2d(input=x4, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff5 = F.conv2d(input=x5, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff6 = F.conv2d(input=x6, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff7 = F.conv2d(input=x7, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
        out_diff8 = F.conv2d(input=x8, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        # re-weight
        illu_trans = self.gap(self.transfer(illu))
        s0, s1, s2, s3, s4, s5, s6, s7, s8 = self.illu_normalization(illu_trans)
        out_diff = out_diff0 * s0 + out_diff1 * s1 + out_diff2 * s2 + out_diff3 * s3 + out_diff4 * s4 + out_diff5 * s5 + out_diff6 * s6 + out_diff7 * s7 + out_diff8 * s8

        new_normal = out_normal - self.theta * out_diff
        new_normal = self.conv_rgbd(torch.cat((new_normal, lidar), 1))
        att = self.att(new_normal)

        return att

        # return out_normal - self.theta * out_diff


    def feature_padding(self, input_feature):

        # 镜像填充：ReflectionPad2d
        # 重复填充：ReplicationPad2d

        B, C, H, W = input_feature.size()
        in_feat = input_feature.clone()
        x4 = in_feat

        # top pad
        left_top_pad = nn.ReflectionPad2d((1, 0, 1, 0))
        x0 = left_top_pad(in_feat)
        x0 = x0[:, :, 0:H, 0:W]

        center_top_pad = nn.ReflectionPad2d((0, 0, 1, 0))
        x1 = center_top_pad(in_feat)
        x1 = x1[:, :, 0:H, :]

        right_top_pad = nn.ReflectionPad2d((0, 1, 1, 0))
        x2 = right_top_pad(in_feat)
        x2 = x2[:, :, 0:H, 1:]

        # center pad
        left_center_pad = nn.ReflectionPad2d((1, 0, 0, 0))
        x3 = left_center_pad(in_feat)
        x3 = x3[:, :, :, 0:W]

        right_center_pad = nn.ReflectionPad2d((0, 1, 0, 0))
        x5 = right_center_pad(in_feat)
        x5 = x5[:, :, :, 1:]

        # bottom pad
        left_bottom_pad = nn.ReflectionPad2d((1, 0, 0, 1))
        x6 = left_bottom_pad(in_feat)
        x6 = x6[:, :, 1:, 0:W]

        center_bottom_pad = nn.ReflectionPad2d((0, 0, 0, 1))
        x7 = center_bottom_pad(in_feat)
        x7 = x7[:, :, 1:, :]

        right_bottm_pad = nn.ReflectionPad2d((0, 1, 0, 1))
        x8 = right_bottm_pad(in_feat)
        x8 = x8[:, :, 1:, 1:]

        return x0, x1, x2, x3, x4, x5, x6, x7, x8


    def illu_normalization(self, illu_trans):

        gate0 = illu_trans.narrow(1, 0, 1)
        gate1 = illu_trans.narrow(1, 1, 1)
        gate2 = illu_trans.narrow(1, 2, 1)
        gate3 = illu_trans.narrow(1, 3, 1)
        gate4 = illu_trans.narrow(1, 4, 1)
        gate5 = illu_trans.narrow(1, 5, 1)
        gate6 = illu_trans.narrow(1, 6, 1)
        gate7 = illu_trans.narrow(1, 7, 1)
        gate8 = illu_trans.narrow(1, 8, 1)

        gate = torch.cat((gate0, gate1, gate2, gate3, gate4, gate5, gate6, gate7, gate8), 1)
        gate_abs = torch.abs(gate)
        abs_weight = gate_abs.sum(1, keepdim=True) + 1e-8

        s0, s1, s2, s3, s4, s5, s6, s7, s8 = torch.div(gate0, abs_weight), torch.div(gate1, abs_weight), \
                                             torch.div(gate2, abs_weight), torch.div(gate3, abs_weight), \
                                             torch.div(gate4, abs_weight), torch.div(gate5, abs_weight), \
                                             torch.div(gate6, abs_weight), torch.div(gate7, abs_weight), \
                                             torch.div(gate8, abs_weight),

        return s0, s1, s2, s3, s4, s5, s6, s7, s8


class PAM_Module(nn.Module):

    """ Position attention module"""

    def __init__(self, in_dim, height, width, kernel_narrow=1):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_pool = nn.AvgPool2d(kernel_size=(kernel_narrow, width), stride=1,
                                             padding=((kernel_narrow - 1) // 2, 0))
        self.key_pool = nn.AvgPool2d(kernel_size=(height, kernel_narrow), stride=1,
                                           padding=(0, (kernel_narrow - 1) // 2))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W) B * 64 * 256 * 1216
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # 3 64 256 1216
        proj_query = self.query_conv(x)
        proj_query_pool = self.query_pool(proj_query)  # 3 * 64 * 256 * 1

        proj_key = self.key_conv(x)
        proj_key_pool = self.key_pool(proj_key)

        energy = proj_query_pool @ proj_key_pool
        energy_reshape = energy.view(m_batchsize, C, height * width)
        attention = self.softmax(energy_reshape)
        attention_reshape = attention.view(m_batchsize, C, height, width)

        proj_value = self.value_conv(x)
        out = proj_value * attention_reshape
        out = self.gamma * out + x

        return out


