import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class L1loss(nn.Module):
    def __init__(self):
        super(L1loss,self).__init__()
        self.loss = nn.L1Loss() # L1Loss
    def forward(self,x,y):
        loss = self.loss(x,y)
        return loss