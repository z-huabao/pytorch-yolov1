# -*- coding: utf-8 -*-
#python3

import math
import torch
import torchviz
from torch import nn
from torchvision import models

class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = torch.relu(out)
        return out

class Backbone(nn.Module):
    """
    get backbone model from torchvision.models, such as 'resnet50',
    and use inner layer activation as output, such as 'layer4'
    """
    def __init__(self, net, layer, pretrained=False):
        """
        net: string, 'resnet50', 'vgg16' ...
        layer: string, 'layer4', 'features' ...
        """
        def set_hook(net, name):
            def hook(model, input, output):
                net.notop_out = output
            getattr(net, name).register_forward_hook(hook)

        super(Backbone, self).__init__()
        self.name = net
        self.net = getattr(models, net)(pretrained=pretrained)
        set_hook(self.net, layer)

    def forward(self, x):
        self.net(x)
        return self.net.notop_out

class YoloV1(nn.Module):
    """build yolo-v1 network"""
    def __init__(self, backbone):
        """backbone: nn.Module, a model has input [N,3,H,W], output [N,C,h,w]"""
        super(YoloV1, self).__init__()
        self.backbone = backbone
        self.name = backbone.name + '-yolov1'
        self.layer5 = self._make_detnet_layer(in_channels=2048)
        self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_detnet_layer(self,in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)

        x = self.layer5(x)
        x = self.conv_end(x)
        x = self.bn_end(x)

        x = torch.sigmoid(x) #归一化到0-1
        # x = x.view(-1,7,7,30)
        x = x.permute(0,2,3,1) #(-1,7,7,30)
        return x

if __name__ == '__main__':
    backbone = Backbone('resnet50', 'layer4', pretrained=True)
    net = YoloV1(backbone)

    g = torchviz.make_dot(net(torch.rand(1,3,448,448)), params=dict(net.named_parameters()))
    g.view()

    print('loading weights')
    net.load_state_dict(torch.load('output/%s-best.pth' % net.name))
    print('finish')
