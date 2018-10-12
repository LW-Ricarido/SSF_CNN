import torch
import torch.nn as nn
import math
from .Strength import Strength_Conv2d
from copy import deepcopy

__all__ = [
   'ssf_vgg11', 'ssf_vgg11_bn', 'ssf_vgg13', 'ssf_vgg13_bn', 'ssf_vgg16', 'ssf_vgg16_bn','ssf_vgg19', 'ssf_vgg19_bn',
]

def Sconv3x3(in_channels,out_channels,stride = 1):
    "3x3 Strenght_Conv2d with padding"
    return Strength_Conv2d(in_channels,out_channels,kerner_size=3,stride=1,padding=1)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg,batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = Sconv3x3(in_channels,v)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = v
    return  nn.Sequential(*layers)

class SSF_VGG(nn.Module):
    def __init__(self,features,args,init_weights = True):
        super(SSF_VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,self.args.output_classes)
        )
        self.args = args
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        TODO:wait for implementing weights of the whole SSF_VGG
        :return:
        """
        print("test")


def ssf_vgg11(args):
    """VGG 11-layer model (configuration "A")

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['A']),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_VGG(make_layers(cfg['A']),args)



def ssf_vgg11_bn(args):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['A'],batch_norm=True),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_VGG(make_layers(cfg['A'],batch_norm=True),args)



def ssf_vgg13(args):
    """VGG 13-layer model (configuration "B")

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['B']),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_VGG(make_layers(cfg['B']),args)



def ssf_vgg13_bn(args):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['B'],batch_norm=True),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_VGG(make_layers(cfg['B'],batch_norm=True),args)



def ssf_vgg16(args):
    """VGG 16-layer model (configuration "D")

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['D']),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_VGG(make_layers(cfg['D']),args)



def ssf_vgg16_bn(args):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['D'],batch_norm=True),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_VGG(make_layers(cfg['D'],batch_norm=True),args)



def ssf_vgg19(args):
    """VGG 19-layer model (configuration "E")

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['E']),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    return SSF_VGG(make_layers(cfg['E']),args)



def ssf_vgg19_bn(args):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        args.pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.pretrained:
        model = SSF_VGG(make_layers(cfg['E'],batch_norm=True),args,False)
        pretrained_dict = torch.load(args.pretrained)
        model_dict = model.state_dict()
        keys = deepcopy(pretrained_dict).keys()
        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]


        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
    return SSF_VGG(make_layers(cfg['E'],batch_norm=True),args)

    




