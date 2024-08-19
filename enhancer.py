import torch
import torch.nn as nn
import torch.nn.functional as F


class RICD(nn.Module):
    def __init__(self, nChannels=1):
        super(RICD, self).__init__()

        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=5, padding=2)   #1
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1)   #3
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()

    def forward(self, x):

        y11 = self.relu1(self.conv1(x))
        y21 = self.relu2(self.conv2(x))
        diff = torch.sigmoid(y11 - y21)

        return diff


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        self.layers = layers
        self.diff_conv = RICD(nChannels=channels)

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.diff = RICD(nChannels=channels)

        self.final_out_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        ####################################################
        diff = self.diff_conv(input)
        fea = self.final_out_conv(torch.cat((fea, diff), 1))
        ####################################################

        illu = fea + input
        illu = torch.sigmoid(illu)
        illu_k = illu

        return illu, illu_k


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta


class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=3, channels=1)
        self.calibrate = CalibrateNetwork(layers=3, channels=1)

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input

        for _ in range(self.stage):
            inlist.append(input_op)
            i, i_k = self.enhance(input_op)
            r = input / i
            r = torch.sigmoid(r)
            att = self.calibrate(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist, i_k


