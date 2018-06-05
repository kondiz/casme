import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNetShared(ResNet):
    def forward(self, x):
        l = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l.append(x)

        x = self.layer1(x)
        l.append(x)
        x = self.layer2(x)
        l.append(x)
        x = self.layer3(x)
        l.append(x)
        x = self.layer4(x)
        l.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, l

def resnet50shared(pretrained=False, **kwargs):
    model = ResNetShared(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model

class Decoder(nn.Module):

    def __init__(self, in_planes, final_upsample_mode = 'nearest'):
        super(Decoder, self).__init__()

        self.conv1x1_1 = self._make_conv1x1_upsampled(in_planes[0], 64)
        self.conv1x1_2 = self._make_conv1x1_upsampled(in_planes[1], 64, 2)
        self.conv1x1_3 = self._make_conv1x1_upsampled(in_planes[2], 64, 4)
        self.conv1x1_4 = self._make_conv1x1_upsampled(in_planes[3], 64, 8)
        self.final = nn.Sequential(
            nn.Conv2d(64 + 4*64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode=final_upsample_mode)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv1x1_upsampled(self, inplanes, outplanes, scale_factor=None):
        if scale_factor:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, l):
        k = []
        k.append(l[0])
        k.append(self.conv1x1_1(l[1]))
        k.append(self.conv1x1_2(l[2]))
        k.append(self.conv1x1_3(l[3]))
        k.append(self.conv1x1_4(l[4]))

        return self.final(torch.cat(k, 1))

def decoder(**kwargs):
    return Decoder([256, 512, 1024, 2048], **kwargs)