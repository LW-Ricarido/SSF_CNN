import torch
import torch.nn as nn
import math
import torch.nn.functional as func


class elastic_net(nn.Module):
    def __init__(self,lambda1,lambda2=1e-4):
        super(elastic_net, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, filter):
        for x in filter:
            return self.lambda1*torch.norm(x,1)+self.lambda2*torch.norm(x,2)

def get_elastic_net(lambda1,lambda2=1e-4):
    return elastic_net(lambda1=lambda1,lambda2=lambda2)