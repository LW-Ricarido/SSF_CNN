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
        self.weight = param.Parameter(torch.ones((out_channels,in_channels/groups,kerner_size,kerner_size)))
        self.weight.requires_grad = False
        self.t = param.Parameter(torch.randn((out_channels)))
        self.t.requires_grad = True
        if bias:
            self.bias = param.Parameter(torch.ones(out_channels))
        else:
            self.bias = None


    def forward(self, input):
        return func.conv2d(input,(self.weight.transpose(0,3) * self.t).transpose(0,3),self.bias,self.stride,self.padding,self.dilation,self.groups)

if __name__ == '__main__':
    #test_strength_conv2d
    input = torch.randn(1,3,5,5)
    target = torch.randn(1,1,3,3)
    SConv2d = Strength_Conv2d(3,1,kerner_size=3)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, SConv2d.parameters()), lr=1e-3)
    loss = nn.L1Loss()
    for i in range(500):
        out = SConv2d(input)
        myloss = loss(out,target)
        print(SConv2d.t)
        optimizer.zero_grad()
        myloss.backward()
        optimizer.step()
