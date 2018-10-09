import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.parameter as param
import torch.optim as optim

class Strength_Conv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kerner_size, stride=1,
                 padding = 0,dilation=1,groups=1,bias=True):
        super(Strength_Conv2d, self).__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.weight = param.Parameter(torch.ones((out_channels,in_channels/groups,kerner_size,kerner_size)))
        self.weight.requires_grad = False
        # self.register_parameter("mweight", self.weight)
        self.t = param.Parameter(torch.randn((out_channels)))
        self.t.requires_grad = True
        # self.register_parameter("strength",self.t)


    def forward(self, input):
        self.weight = param.Parameter(self.weight.transpose(0,3))
        self.weight = self.weight * self.t
        self.weight = self.weight.transpose(0,3)
        return func.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)

if __name__ == '__main__':
    input = torch.randn(1,3,5,5)
    SConv2d = Strength_Conv2d(3,64,kerner_size=3)
    out = SConv2d.forward(input)
    print(out)
    # optimzier = optim.SGD(filter(lambda p:p.requires_grad,SConv2d.named_parameters()))