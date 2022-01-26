import torch
import torch.nn as nn
import numpy as np
import nics_fix_pt.nn_fix as nnf
from utils import *
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}
BITWIDTH = 16
class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    
    # initialize some fix configurations
    self.conv1_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    activation_num = 4
    self.fix_params = [
      generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
      for _ in range(activation_num)
    ]

    # initialize activation fix modules
    for i in range(len(self.fix_params)):
      setattr(self, "fix"+str(i), nnf.Activation_fix(nf_fix_params=self.fix_params[i]))

    self.relu = nn.ReLU(inplace=False)
    self.conv = nnf.Conv2d_fix(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, nf_fix_params=self.conv1_fix_params)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
	
    #self.op = nn.Sequential(
#      nn.ReLU(inplace=False),
#      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
#      nn.BatchNorm2d(C_out, affine=affine)
#    )

  def forward(self, x):
    x =self.fix0(x)
    x = self.fix1(self.relu(x))
    x = self.fix2(self.conv(x))
    x = self.fix3(self.bn(x))

    return x
    #return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.conv1_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    self.conv2_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    activation_num = 5
    self.fix_params = [
      generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
      for _ in range(activation_num)
    ]

    # initialize activation fix modules
    for i in range(len(self.fix_params)):
      setattr(self, "fix"+str(i), nnf.Activation_fix(nf_fix_params=self.fix_params[i]))
    #self.op = nn.Sequential(
     # nn.ReLU(inplace=False),
     # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
     # nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
     # nn.BatchNorm2d(C_out, affine=affine),
     # )
	  
    self.R1 = nn.ReLU(inplace=False)
    self.conv1 = nnf.Conv2d_fix(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False, nf_fix_params=self.conv1_fix_params)
    self.conv2 = nnf.Conv2d_fix(C_in, C_out, kernel_size=1, padding=0, bias=False, nf_fix_params=self.conv2_fix_params)
    self.bn1 = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.fix0(x)
    x = self.fix1(self.R1(x))
    x = self.fix2(self.conv1(x))
    x = self.fix3(self.conv2(x))
    x = self.fix4(self.bn1(x))
    return x
    # return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    # initialize some fix configurations
    self.conv1_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    self.conv2_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    self.conv3_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    self.conv4_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    activation_num = 9
    self.fix_params = [
      generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
      for _ in range(activation_num)
    ]

    # initialize activation fix modules
    for i in range(len(self.fix_params)):
      setattr(self, "fix"+str(i), nnf.Activation_fix(nf_fix_params=self.fix_params[i]))

    # self.op = nn.Sequential(
    self.R1 = nn.ReLU(inplace=False)
    self.conv1 = nnf.Conv2d_fix(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False, nf_fix_params=self.conv1_fix_params)
    self.conv2 = nnf.Conv2d_fix(C_in, C_in, kernel_size=1, padding=0, bias=False, nf_fix_params=self.conv2_fix_params)
    self.bn1= nn.BatchNorm2d(C_in, affine=affine)
    self.R2 = nn.ReLU(inplace=False)
    self.conv3 = nnf.Conv2d_fix(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False, nf_fix_params=self.conv3_fix_params)
    self.conv4 = nnf.Conv2d_fix(C_in, C_out, kernel_size=1, padding=0, bias=False, nf_fix_params=self.conv4_fix_params)
    self.bn2 = nn.BatchNorm2d(C_out, affine=affine)
      # )
  #
  def forward(self, x):
    x = self.fix0(x)
    x = self.fix1(self.R1(x))
    x = self.fix2(self.conv1(x))
    x = self.fix3(self.conv2(x))
    x = self.fix4(self.bn1(x))
    x = self.fix5(self.R2(x))
    x = self.fix6(self.conv3(x))
    x = self.fix7(self.conv4(x))
    x = self.fix8(self.bn2(x))
    return x
    # return self.op(x)

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class FactorizedReduce(nnf.FixTopModule):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0

    self.conv1_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    self.conv2_fix_params = generate_default_fix_cfg(
      ["weight"], method=1, bitwidth=BITWIDTH)

    activation_num = 5
    self.fix_params = [
      generate_default_fix_cfg(["activation"], method=1, bitwidth=BITWIDTH)
      for _ in range(activation_num)
    ]

    # initialize activation fix modules
    for i in range(len(self.fix_params)):
      setattr(self, "fix"+str(i), nnf.Activation_fix(nf_fix_params=self.fix_params[i]))

    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nnf.Conv2d_fix(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, nf_fix_params=self.conv1_fix_params)
    self.conv_2 = nnf.Conv2d_fix(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, nf_fix_params=self.conv2_fix_params)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.fix0(x)

    x = self.fix1(self.relu(x))
    out = torch.cat([self.fix2(self.conv_1(x)), self.fix3(self.conv_2(x[:,:,1:,1:]))], dim=1)
    out = self.fix4(self.bn(out))
    return out

