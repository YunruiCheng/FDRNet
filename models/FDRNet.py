import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx


class convd(nn.Module): 
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size//2) 
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
    def forward(self, x):
        x = self.conv(self.padding(x))
        x = self.relu(x)
        return x

def get_residue(tensor , r_dim = 1):
    """
    return residue_channle (RGB)
    """
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

class Get_gradient(nn.Module): 
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Prior_Sp(nn.Module): 
    """ Channel attention module"""
    def __init__(self, in_dim=64):
        super(Prior_Sp, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()
    def forward(self,x, prior):
        
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)        
        attention_x = x * attention
        attention_p = prior * attention

        x_out = x+attention_x
        prior_out = prior+attention_p


        return x_out, prior_out


class PSA(nn.Module):	

    def __init__(self, channel=512,reduction=4,S=4):
        super().__init__()
        self.S=S

        self.convs = nn.ModuleList([])

        for i in range(S):
            self.convs.append(nn.Conv2d(channel//S,channel//S,kernel_size=3,padding=i+1, dilation=i+1))
            
        self.se_blocks = nn.ModuleList([])
        for i in range(S):
            self.se_blocks.append(nn.Sequential(	
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel//S, channel // (S*reduction),kernel_size=1, bias=False),	
                nn.ReLU(inplace=False),
                nn.Conv2d(channel // (S*reduction), channel//S,kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        
        self.softmax=nn.Softmax(dim=1)


    def forward(self, x):
        b, c, h, w = x.size()

        SPC_out=x.view(b,self.S,c//self.S,h,w) 
        
        for idx,conv in enumerate(self.convs):
            spc_conv = SPC_out.clone()
            SPC_out[:,idx,:,:,:]=conv(spc_conv[:,idx,:,:,:])	

        SE_out=torch.zeros_like(SPC_out)
        for idx,se in enumerate(self.se_blocks):
            SE_out[:,idx,:,:,:]=se(SPC_out[:,idx,:,:,:])	
        
        softmax_out=self.softmax(SE_out)

        PSA_out=SPC_out*softmax_out	

        PSA_out=PSA_out.view(b,-1,h,w)

        return PSA_out		


class DIPSABlock(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel=3, dilation=1):
        super().__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv1 = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad, dilation=dilation) 

        self.dipsa = PSA(oup_dim) 
        self.conv2 = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad, dilation=dilation) 
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(self.dipsa(x)))
        if x.shape[1] == res.shape[1]: 
          x = self.relu(res + x) 
        return x


class FDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 64 
        self.depth = 6          
        self.stage_num = 3      
        self.grad_p = Get_gradient() 
        self.grad_p2 = Get_gradient()
        self.grad_p3 = Get_gradient() 
        self.fuse_RG1 = nn.Conv2d(6, 3, 1, 1, dilation=1) 
        self.fuse_RG2 = nn.Conv2d(6, 3, 1, 1, dilation=1) 
        self.fuse_RG3 = nn.Conv2d(6, 3, 1, 1, dilation=1) 
        self.conv1 = nn.Conv2d(3, self.channel, 3, padding=1, dilation=1) 
        self.conv2 = nn.Conv2d(3, self.channel, 3, padding=1, dilation=1) 
        self.conv3 = nn.Conv2d(3, self.channel, 3, padding=1, dilation=1) 
        self.grad_rcp_conv1 = nn.Conv2d(3, self.channel, 3, padding=1, dilation=1) 
        self.grad_rcp_conv2 = nn.Conv2d(3, self.channel, 3, padding=1, dilation=1) 
        self.grad_rcp_conv3 = nn.Conv2d(3, self.channel, 3, padding=1, dilation=1) 

        # fuse res
        self.prior = Prior_Sp() 
        self.prior2 = Prior_Sp() 
        self.prior3 = Prior_Sp() 
        self.fuse_res = nn.Conv2d(self.channel*2, self.channel, 1, 1, dilation=1) 
        self.fuse_res2 = nn.Conv2d(self.channel*2, self.channel, 1, 1, dilation=1) 
        self.fuse_res3 = nn.Conv2d(self.channel*2, self.channel, 1, 1, dilation=1) 

        self.ag1 = convd(self.channel*2,self.channel,3,1)
        self.ag2 = convd(self.channel*3,self.channel,3,1)
        self.ag2_en = convd(self.channel*2, self.channel, 3, 1)
        self.ag_en = convd(self.channel*3, self.channel, 3, 1)
        self.relu = nn.ReLU()

        self.dipsas1 = nn.ModuleList(
            [DIPSABlock(self.channel, self.channel) for i in range(self.depth)]
        )  
        self.dec1 = nn.Sequential( 
            nn.Conv2d(self.channel, self.channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channel, 3, 1),
        )

        self.dipsas2 = nn.ModuleList(
            [DIPSABlock(self.channel, self.channel) for i in range(self.depth)]
        )  
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channel, 3, 1),
        )

        self.dipsas3 = nn.ModuleList(
            [DIPSABlock(self.channel, self.channel) for i in range(self.depth)]
        )  
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channel, 3, 1), 
            # nn.ReLU(),
        )

    
    def forward(self, x):
        derains = []
        outputs = []

        ## Stage 1
        res_x = get_residue(x) 
        res_x = torch.cat((res_x, res_x, res_x), dim=1) 
        grad_x = self.grad_p(x) # 3c,grad
        MixPrior_x = self.fuse_RG1(torch.cat((res_x,grad_x), dim=1)) 
        MixPrior_x = self.relu(self.grad_rcp_conv1(MixPrior_x)) 
        

        x = self.conv1(x) 
        init_feat1 = x

        # IFM
        x_p, MixPrior_x_p = self.prior(x, MixPrior_x) 
        x_s = torch.cat((x_p, MixPrior_x_p),dim=1) 
        x = self.relu(self.fuse_res(x_s))  
        for dipsa in self.dipsas1: 
            x = dipsa(x) 
        
        out_feat1 = x
        derain = self.dec1(x) 
        derains.append(derain)
        x = derain 


        ## Stage 2
        res_x = get_residue(x) 
        res_x = torch.cat((res_x, res_x, res_x), dim=1) 
        grad_x = self.grad_p(x) 
        MixPrior_x = self.fuse_RG2(torch.cat((res_x,grad_x), dim=1)) 
        MixPrior_x = self.relu(self.grad_rcp_conv2(MixPrior_x)) 
        

        x = self.conv2(x) 
        init_feat2 = x
        
        x = self.ag1(torch.cat((init_feat1,init_feat2),dim=1))
        
        # IFM
        x_p, MixPrior_x_p = self.prior2(x, MixPrior_x) 
        x_s = torch.cat((x_p, MixPrior_x_p),dim=1) 
        x = self.relu(self.fuse_res2(x_s))  

        for dipsa in self.dipsas2: 
            x = dipsa(x) 
        
        out_feat2 = x
        x = self.ag2_en(torch.cat((out_feat1,out_feat2), dim=1)) 
        derain = self.dec2(x) 
        derains.append(derain)
        x = derain  

        ## Stage 3
        res_x = get_residue(x) 
        res_x = torch.cat((res_x, res_x, res_x), dim=1) 
        grad_x = self.grad_p(x) 
        MixPrior_x = self.fuse_RG3(torch.cat((res_x,grad_x), dim=1)) 
        MixPrior_x = self.relu(self.grad_rcp_conv3(MixPrior_x)) 

        x = self.conv3(x) 
        init_feat3 = x
        x = self.ag2(torch.cat((torch.cat((init_feat1,init_feat2),dim=1),init_feat3),dim=1))

        # IFM
        x_p, MixPrior_x_p = self.prior3(x, MixPrior_x) 
        x_s = torch.cat((x_p, MixPrior_x_p),dim=1) 
        x = self.relu(self.fuse_res3(x_s))  

        for dipsa in self.dipsas3: 
            x = dipsa(x) 

        out_feat3 = x
        x = self.ag_en(torch.cat((torch.cat((out_feat1,out_feat2), dim=1),out_feat3),dim=1))
        derain = self.dec3(x) 
        derains.append(derain)
        x = derain        

        for i in range(self.stage_num): 
            outputs.append(derains[self.stage_num-i-1])

        return outputs


if __name__ == '__main__':
    input=torch.randn(50,3,256,256)
    net = DPSANet_GradPrior()

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

    output=net(input)
    print(output[0].shape) 


    

