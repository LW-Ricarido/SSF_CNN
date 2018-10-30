import torch
import torch.nn as nn
import math
from .Strength import Strength_Conv2d
from copy import deepcopy

__all__ = ['Bottleneck','SSF_ResNet','ssf_resnet18','ssf_resnet34','ssf_resnet50','ssf_resnet101','ssf_resnet152']

def Sconv3x3(in_channels,out_channels,stride = 1):
    "3x3 Strenght_Conv2d with padding"
    return Strength_Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Sconv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Sconv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        #TODO:wait for check
        self.conv1 = Strength_Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Strength_Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Strength_Conv2d(planes,planes * 4,kernel_size=1,bias=False)
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


class SSF_ResNet(nn.Module):

    def __init__(self,block,layers,args):
        self.inplanes = 64
        super(SSF_ResNet, self).__init__()
        self.conv1 = Strength_Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, args.output_classes)

        for m in self.modules():
            if isinstance(m,Strength_Conv2d):
                n = m.kernel_size * m.kernel_size * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Strength_Conv2d(self.inplanes,planes * block.expansion,
                                kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))

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
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        return x


def ssf_resnet18(args):
    """Constructs a ResNet-18 model.

    Args:
        pretrained : If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_ResNet(BasicBlock,[2,2,2,2],args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_ResNet(BasicBlock, [2,2,2,2],args)


def ssf_resnet34(args):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_ResNet(BasicBlock, [3, 4, 6, 3], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_ResNet(BasicBlock, [3, 4, 6, 3], args)


def ssf_resnet50(args):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_ResNet(Bottleneck, [3, 4, 6, 3], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_ResNet(Bottleneck, [3, 4, 6, 3], args)


def ssf_resnet101(args):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_ResNet(Bottleneck, [3, 4, 23, 3], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_ResNet(Bottleneck, [3, 4, 23, 3], args)


def ssf_resnet152(args):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_ResNet(Bottleneck, [3, 8, 36, 3], args)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_ResNet(Bottleneck, [3, 8, 36, 3], args)
