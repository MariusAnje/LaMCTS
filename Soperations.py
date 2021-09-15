# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from Functions import SCrossEntropyLossFunction
from modules import NConv2d, NSTPConv2d, NLinear, SConv2d, SSTPConv2d, SLinear, SBatchNorm2d, SReLU

OPS = {
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_2x2' : lambda C, stride, affine: nn.MaxPool2d(2, stride=stride, padding=0),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'max_pool_5x5': lambda C, stride, affine: nn.MaxPool2d(5, stride=stride, padding=2),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_1x1' : lambda C, stride, affine: NConv2d(C, C, (1,1), stride=(stride, stride), padding=(0,0), bias=False),
    'conv_3x3' : lambda C, stride, affine: NConv2d(C, C, (3,3), stride=(stride, stride), padding=(1,1), bias=False),
    'conv_5x5' : lambda C, stride, affine: NConv2d(C, C, (5,5), stride=(stride, stride), padding=(2,2), bias=False),
}

class SCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        self.function = SCrossEntropyLossFunction
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
    
    def forward(self, input, inputS, labels):
        output = self.function.apply(input, inputS, labels, self.weight, self.size_average, self.ignore_index, self.reduce, self.reduction)
        return output

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):

        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            SReLU(inplace=False),
            SSTPConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            SBatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            SReLU(inplace=False),
            SConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=C_in, bias=False),
            SSTPConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            SBatchNorm2d(C_in, affine=affine),
            SReLU(inplace=False),
            SConv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=C_in, bias=False),
            SSTPConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            SBatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            SReLU(inplace=False),
            SConv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            SSTPConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            SBatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):

        super(FactorizedReduce, self).__init__()

        assert C_out % 2 == 0

        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = NConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = NConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)

        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
