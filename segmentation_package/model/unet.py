###################################
# Author smile                    #
# Date 2018-12-11                 #
# Base models: FCN, UNet, Deeplab #
###################################
from torch import nn
import torch
import torch.nn.functional as F
from ..utils.model_utils import initialize_weights


########
# UNet #
########
class _EncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, dropout=False):
    super(_EncoderBlock, self).__init__()
    layers = [
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    ]
    if dropout:
      layers.append(nn.Dropout())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    self.encode = nn.Sequential(*layers)

  def forward(self, x):
    return self.encode(x)

class _DecoderBlock(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(_DecoderBlock, self).__init__()
    self.decode = nn.Sequential(
      nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(middle_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(middle_channels),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
    )

  def forward(self, x):
    return self.decode(x)

class UNet(nn.Module):
  def __init__(self, num_classes, base_channel=4):
    super(UNet, self).__init__()
    self.enc1 = _EncoderBlock(3, base_channel)  # _EncoderBlock(3, 64)
    self.enc2 = _EncoderBlock(base_channel, base_channel * 2)
    self.enc3 = _EncoderBlock(base_channel * 2, base_channel * 4)
    self.enc4 = _EncoderBlock(base_channel * 4, base_channel * 8)
    self.center = _DecoderBlock(base_channel * 8, base_channel * 16, base_channel * 8)
    self.dec4 = _DecoderBlock(base_channel * 16, base_channel * 8, base_channel * 4)
    self.dec3 = _DecoderBlock(base_channel * 8, base_channel * 4, base_channel * 2)
    self.dec2 = _DecoderBlock(base_channel * 4, base_channel * 2, base_channel)
    self.dec1 = nn.Sequential(
      nn.Conv2d(base_channel * 2, base_channel, kernel_size=3, padding=1),
      nn.BatchNorm2d(base_channel),
      nn.ReLU(inplace=True),
      nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
      nn.BatchNorm2d(base_channel),
      nn.ReLU(inplace=True),
    )
    self.final = nn.Conv2d(base_channel, num_classes, kernel_size=1)
    initialize_weights(self)

  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(enc1)
    enc3 = self.enc3(enc2)
    enc4 = self.enc4(enc3)
    center = self.center(enc4)
    dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
    dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
    dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
    dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
    final = self.final(dec1)
    return F.interpolate(final, x.size()[2:], mode='bilinear', align_corners=True)
    # return F.interpolate(final, torch.Size([400, 400]), mode='bilinear', align_corners=True)
