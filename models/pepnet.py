from os.path import join, exists, dirname, abspath
import subprocess, h5py, numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#------------------------------------------------------------------------------------------------------

sys.path.insert(0, dirname(dirname(abspath("__file__"))))
num_do_mcmc = 50

#------------------------------Define a 1D convolution layer to be used--------------------------------
def conv1x3(in_planes, out_planes, stride=1):
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#---------------------------------Define BasicBlock Module for Resnet----------------------------------
# Contains 2 convolutional Layers
# Used as the block in our model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)   
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
		
		#not used
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#==============================================not used===============================================
#-----------------------------Define BottleNeck Module for Resnet-------------------------------------
# Contains 3 Convolutional Layers
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
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
#========================================================================================================

#---------------------------------------Define Resnet--------------------------------------------------
# Number of columns i.e. 40 does not change throughout resnet, except the avgpool layer.
# Ignoring the batch size for easy understanding
class ResNet(nn.Module):

    def __init__(self, block, layers, seq_len=40, conv_fn=256, embed_size=1400):
        self.inplanes = conv_fn   #256
        super(ResNet, self).__init__()
        # 1400 X 40 --> 256 X 40
        self.conv1 = nn.Conv1d(embed_size, conv_fn, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(conv_fn)
        self.relu = nn.ReLU(inplace=True)

        stride = [1 if i==0 else 2 for i in range(len(layers))] #stride = [1], layers = [5]

        #_make_layer parameters passed: (BasicBlock, 256, 5, 1)
        self.layers = nn.ModuleList([self._make_layer(block, conv_fn*(2**idx), layer, stride=stride[idx]) for idx, layer in enumerate(layers)])
        
        to_divide = 2**(len(layers)-1)      #1  

        # 256 x 40 --> 256 x 1       
        self.avgpool = nn.AvgPool1d((seq_len+to_divide-1)//to_divide, stride=1)
        # Thus, each 1400 x 40 input is converted to 256 x 1 output 

        # initializing weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        #----------------------not used-----------------------
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        #------------------------------------------------------

        layers = []   #contains the number of blocks
        #256 x 40 --> 256 x 40 (5 times)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x returned is of shape (batch_size, 256)
        # Each row is an example represented as 256 element vector
        return x

#-----------------Define the binding affinity module using resnet------------------------------------
class PEPnet(nn.Module):

    def __init__(self, config):
        super(PEPnet, self).__init__()

        block_type = BasicBlock if config['pep_block'] == 'basic' else Bottleneck
        self.resnet = ResNet(
                block_type, #Basic Block
                config['pep_layers'],  #5
                seq_len = config['pep_len'],  #40
                conv_fn = config['pep_conv_fn'],  #256
                embed_size = config['pep_embed_size'] + config['mhc_embed_size']*config['mhc_len'],  #1400
                )
        self.outlen =  config['pep_conv_fn']*(2**(len(config['pep_layers'])-1)) * block_type(1, 1).expansion #256
        self.config = config

    def forward(self, x):
        return self.resnet(x)

#------------------Combine the binding affinity module and peptide embedding module-------------------
class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.PEPnet = PEPnet(config)
		    # 2 is added because of sigmoid transformed L and 1-L concatenation
        # 258(256 + 2) --> 64
        self.fc1_alpha = nn.Linear(self.PEPnet.outlen+2, config['dense_size'])
        # 258(256 + 2) --> 64
        self.fc1_beta = nn.Linear(self.PEPnet.outlen+2, config['dense_size'])
        # 64 --> 1 (gives mean)
        self.fc2_alpha = nn.Linear(config['dense_size'], config['class_num'])
        # 64 --> 1 (gives variance)
        self.fc2_beta = nn.Linear(config['dense_size'], config['class_num'])
        #2 is added for mean and variance
        #2562(2560 + 2) --> 1 (predicts ligand/non-ligand)
        self.fc_mass = nn.Linear(config['mass_embed_size']+2, 1)
        self.nl = nn.Tanh()
        self.config = config
        
    def embed(self, mhc, pep, lenpep):
    	#flatten the mhc from 40 x 34 to 1360 x 40(This 40 is basically 1st column repeated 40 times)
        mhc_flat = mhc.view(-1, self.config['mhc_embed_size']*self.config['mhc_len'], 1).repeat(1, 1, self.config['pep_len'])
        # concatenate 40 x 40 peptide embedding and 1360 x 40 mhc layer to form 1400 x 40 input
        pep_in = torch.cat((pep, mhc_flat), dim=1)
        # send it in the PEP net
        pep_out = self.PEPnet(pep_in)
        # pep_out shape is (batch_size, 256)
        # concatenate output with sigmoid(L) and 1 - sigmoid(L) entries
        return torch.cat([pep_out, lenpep], dim=1)

    def forward(self, mhc, pep, lenpep, elmo):
        x = self.embed(mhc, pep, lenpep)
        # predict mean
        # x is now of shape (batch, 258)
        m = F.sigmoid(self.fc2_alpha(self.nl(self.fc1_alpha(x))))
        # predict variance
        v = F.softplus(self.fc2_beta(self.nl(self.fc1_beta(x))))
        # predict label
        input2mass = torch.cat([elmo.view(len(x), -1), m, v], dim=1)
        mass_pred = F.sigmoid(self.fc_mass(input2mass))
        return m, v, mass_pred