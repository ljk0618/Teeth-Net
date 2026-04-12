import math

from posixpath import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class SEGroupGate(nn.Module):

    def __init__(self, channels, num_groups=4, reduction=16):
        super().__init__()
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        hidden = max(channels // reduction, 16)
        self.num_groups = num_groups
        self.group_channels = channels // num_groups

        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.shape
        z = x.mean(dim=(2, 3))
        e = self.fc2(self.relu(self.fc1(z)))
        e = e.view(b, self.num_groups, self.group_channels)
        logits = e.mean(dim=-1)
        weights = self.softmax(logits)

        x_group = x.view(b, self.num_groups, self.group_channels, x.size(2), x.size(3))
        x_group = x_group * weights[:, :, None, None, None]
        out = x_group.view(b, c, x.size(2), x.size(3))
        return out, weights


class CrossScalePyramidFusion(nn.Module):

    def __init__(self, channels, stride=1, k1=1, k2=3, k3=5, num_scales=4):
        super().__init__()
        assert channels % num_scales == 0, f"channels must be divisible by {num_scales}, but got {channels}"
        self.num_scales = num_scales
        self.branch_channels = channels // num_scales

        self.branch0 = StdConv2d(
            channels, self.branch_channels,
            kernel_size=(k1, k1),
            stride=stride,
            padding=(k1 // 2, k1 // 2),
            bias=False
        )
        self.branch1 = StdConv2d(
            channels, self.branch_channels,
            kernel_size=(k1, k2),
            stride=stride,
            padding=(k1 // 2, k2 // 2),
            bias=False
        )
        self.branch2 = StdConv2d(
            channels, self.branch_channels,
            kernel_size=(k2, k2),
            stride=stride,
            padding=(k2 // 2, k2 // 2),
            bias=False
        )
        self.branch3 = StdConv2d(
            channels, self.branch_channels,
            kernel_size=(k3, k3),
            stride=stride,
            padding=(k3 // 2, k3 // 2),
            bias=False
        )

        self.se_gate = SEGroupGate(channels, num_groups=num_scales, reduction=16)

        self.last_desc = None
        self.last_similarity = None
        self.last_attention = None
        self.last_group_weights = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, StdConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f0 = self.branch0(x)
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        feat_stack = torch.stack([f0, f1, f2, f3], dim=1)  # [B, 4, C/4, H, W]
        desc = feat_stack.mean(dim=(-1, -2))
        sim = torch.matmul(desc, desc.transpose(1, 2))
        attn = torch.softmax(sim, dim=-1)
        enhanced = torch.einsum('bij,bjchw->bichw', attn, feat_stack)
        b, n, c_each, h, w = enhanced.shape
        fused = enhanced.reshape(b, n * c_each, h, w)
        out, group_weights = self.se_gate(fused)

        self.last_desc = desc.detach()
        self.last_similarity = sim.detach()
        self.last_attention = attn.detach()
        self.last_group_weights = group_weights.detach()
        return out


class PreActBottleneck(nn.Module):

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)

        self.conv2 = CrossScalePyramidFusion(cmid, stride=stride, k1=1, k2=3, k3=5)

        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        # conv2 已由 CFFM 替换，无法直接加载原始单个 3x3 卷积参数
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
