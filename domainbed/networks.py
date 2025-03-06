# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.utils.model_zoo as model_zoo
import numpy as np
from domainbed.lib import wide_resnet
import random


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.firstCall = True
        self.mean_style = []
        self.std_style = []
        self.count = 0
        self.state = 'train'
        self.mean_1 = None
        self.std_1 = None
       
    def forward(self, x):
        #if (not self.training): 
        if self.state =='tent':   
            return x
            mean = x.mean(dim=[0, 2, 3], keepdim=False)
            std = (x.var(dim=[0, 2, 3], keepdim=False) + self.eps).sqrt()
            
            dists_mean = []
            dists_std = []
            for t in range(3):
                dists_mean.append(torch.cdist(mean.reshape(1,x.shape[1]), self.mean_style[t].reshape(1,x.shape[1])))
                dists_std.append(torch.cdist(std.reshape(1,x.shape[1]),self.std_style[t].reshape(1,x.shape[1])))
            #print(dists_mean)
            dists_mean = torch.cat(dists_mean)
            dists_std = torch.cat(dists_std)
            _,mean_idx  = torch.min(dists_mean,0)
            _,std_idx  = torch.min(dists_std,0)
            
            #alpha = 1-(self.count % 10)/10
            
            #alpha = 0
            #new_mean = alpha * mean + (1 - alpha) * self.mean_style[mean_idx]
            #new_std = alpha * std + (1 - alpha) * self.std_style[std_idx]
            if random.random()<.5:
                pass
            else:      
                x = (x - mean.reshape(1, x.shape[1], 1, 1)) / std.reshape(1, x.shape[1], 1, 1)
                x = x * self.std_style[std_idx].reshape(1, x.shape[1], 1, 1) + self.mean_style[mean_idx].reshape(1, x.shape[1], 1, 1)
                #x = x * new_std.reshape(1, x.shape[1], 1, 1) + new_mean.reshape(1, x.shape[1], 1, 1)
                pass
                
            self.count = self.count + 1
            return x
            
                
        if self.state =='test':
            return x
        
        if self.state =='train':
            return x
            if x.size()[0] != 48:
                return x
            current_mean = []
            current_std = []
            for t in range(3):
                mean = x[t*16:(t+1)*16].mean(dim=[0, 2, 3], keepdim=False)
                std = (x[t*16:(t+1)*16].var(dim=[0, 2, 3], keepdim=False) + self.eps).sqrt()
                current_mean.append(mean.reshape(1,x.shape[1]))
                current_std.append(std.reshape(1,x.shape[1]))
                current_mean[t] = current_mean[t].repeat(16,1)
                current_std[t] = current_std[t].repeat(16,1)
                
                #x[t*16:(t+1)*16] = (x[t*16:(t+1)*16] - mean.reshape(1, x.shape[1], 1, 1)) / std.reshape(1, x.shape[1], 1, 1)
                
                if self.firstCall == True:
                    self.mean_style.append(mean.detach())
                    self.std_style.append(std.detach())
                    
                    self.mean_style[t] = self.mean_style[t].reshape(1,x.shape[1])
                    self.std_style[t] = self.std_style[t].reshape(1,x.shape[1])
                else:
                    #print(mean.size())
                    #print(self.meanStyle.size())
                    self.mean_style[t] = .01*mean.detach() + .99*self.mean_style[t]
                    self.std_style[t] = .01*std.detach() + .99*self.std_style[t]
                    
            if self.firstCall == True:
                self.mean_style = torch.cat(self.mean_style)
                self.std_style = torch.cat(self.std_style)
                
            current_mean = torch.cat(current_mean) 
            current_std = torch.cat(current_std) 
            #print(current_mean.size())
            x = (x - current_mean.reshape(x.shape[0], x.shape[1], 1, 1)) / current_std.reshape(x.shape[0], x.shape[1], 1, 1)
            adapt_mean = []
            adapt_std = []
            for t in range(3):
                adapt_mean.append(self.mean_style[t].repeat(16,1))
                adapt_std.append(self.std_style[t].repeat(16,1))
            adapt_mean = torch.cat((adapt_mean[random.randint(0, 2)],adapt_mean[random.randint(0, 2)],adapt_mean[random.randint(0, 2)]), 0) 
            adapt_std = torch.cat((adapt_std[random.randint(0, 2)], adapt_std[random.randint(0, 2)], adapt_std[random.randint(0, 2)]), 0)   
            x = x * adapt_std.reshape(x.shape[0], x.shape[1], 1, 1) + adapt_mean.reshape(x.shape[0], x.shape[1], 1, 1)
            
            self.firstCall = False
        
        
        
        return x
    
    
    '''
    def forward(self,x):
        if self.state == 'train':
            self.mean_1 = x[0:16].mean(dim=[0, 2, 3], keepdim=False)
            self.std_1 = (x[0:16].var(dim=[0, 2, 3], keepdim=False) + self.eps).sqrt()
          
        mean_in = x.mean(dim=[ 2, 3], keepdim=False)
        std_in = (x.var(dim=[ 2, 3], keepdim=False) + self.eps).sqrt()
        if self.state == 'train' or self.state == 'test' or self.state == 'tent':
            x = (x - mean_in.reshape(x.shape[0], x.shape[1], 1, 1)) / std_in.reshape(x.shape[0], x.shape[1], 1, 1)
            x = x * self.std_1.reshape(1, x.shape[1], 1, 1) + self.mean_1.reshape(1, x.shape[1], 1, 1)
        return x
    '''
        
            



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class UResNet(nn.Module):

    def __init__(
            self, block, layers, pertubration=None, uncertainty=0.0, **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pertubration0 = pertubration( p=uncertainty) if pertubration else nn.Identity()
        self.pertubration1 = pertubration( p=uncertainty) if pertubration else nn.Identity()
        self.pertubration2 = pertubration( p=uncertainty) if pertubration else nn.Identity()
        self.pertubration3 = pertubration( p=uncertainty) if pertubration else nn.Identity()
        self.pertubration4 = pertubration( p=uncertainty) if pertubration else nn.Identity()
        self.pertubration5 = pertubration( p=uncertainty) if pertubration else nn.Identity()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.pertubration0(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.pertubration1(x)
        x = self.layer1(x)
        x = self.pertubration2(x)
        x = self.layer2(x)
        x = self.pertubration3(x)
        
        x = self.layer3(x)
        #x = self.pertubration4(x)
        x = self.layer4(x)
        #x = self.pertubration5(x)

        return x

    def forward(self, x):
        
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
"""
Standard residual networks
"""

def uresnet50(pretrained=True, uncertainty=0.5, **kwargs):
    model = UResNet(block=Bottleneck, layers=[3, 4, 6, 3],
                    pertubration=DistributionUncertainty, uncertainty=uncertainty)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            if network is None:
                network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
            self.network = network
            self.n_outputs = 512
        else:
            if network is None:
                #network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
                network = uresnet50()
            self.network = network
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        
        #del self.network.fc
        #self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams["freeze_bn"] is False:
            for m in self.network.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(False)
                    #m.register_parameter("weight", None)
                    #m.register_parameter("bias", None)
                    #m.affine = False
                    #m.track_running_stats = False
            return
        
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                #m.affine = False
                #m.track_running_stats = False
               


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")
