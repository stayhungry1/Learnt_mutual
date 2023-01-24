###这个文件，可以跑coco的测试，速度没问题！ (原版为rcnn_P2down2345outMSE_zeropadnew)
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers.batch_norm import FrozenBatchNorm2d, get_norm

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

# # --add compressai framework
import sys
sys.path.append("/media/data/liutie/VCM/rcnn/VCMbelle_0622")
sys.path.append("/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM")
# sys.path.append("/media/data/ccr/liutieCompressAI/")
# sys.path.append("/media/data/ccr/liutieCompressAI/VCM/")
from compressai.datasets.parse_p2_feature import _joint_split_features
# 只用到了IRN_inference和main_cai
# from examples.train_in_this import * ###这句话是拖慢coco测试的大问题！！！
import train_in_this_copyuseful

import functools
from torch.autograd import Variable
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from compressai.models import *
import random
import time
import json

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################################################################
# Functions
#############################################################################################
def Pfeature_replicatepad(feat, factor=16): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    h_new_left = (h_new - h) // 2
    h_new_right = (h_new - h) - h_new_left
    w_new_left = (w_new - w) // 2
    w_new_right = (w_new - w) - w_new_left
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ReplicationPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_replicatepad_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def Pfeature_replicatepad_youxiajiao(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ReplicationPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao128(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    #加了下面4行让hw_new最小为128
    if h_new < 128:
        h_new = 128
    if w_new < 128:
        w_new = 128
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao128_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def Pfeature_zeropad_youxiajiao(feat, factor=16):
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def mkdirs(path):
    # if not os.path.exists(path):
    #     os.makedirs(path)
    os.makedirs(path, exist_ok=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_compG(input_nc, output_nc, ngf, n_downsample_global=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netcompG = CompGenerator(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #netcompG.cuda(gpu_ids[1])
        #netcompG.cuda('cuda:1')
        #netcompG.cuda(1)
        netcompG.to(device)

    netcompG.apply(weights_init)
    return netcompG

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    else:
        raise('generator not implemented!')
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #netG.cuda(gpu_ids[0])
        #netG.cuda(1)
        netG.to(device)
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #netD.cuda(gpu_ids[0])
        netD.to(device)
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class CompGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=3, norm_layer=nn.BatchNorm2d):
        super(CompGenerator, self).__init__()
        self.output_nc = output_nc

        #model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True)]
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                 nn.ReLU(True)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf * (2 ** n_downsampling), output_nc, kernel_size=7, padding=0)] #nn.Tanh()
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        # n_downsampling=0
        # input: 1x3xwxh
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # output: 1x64xwxh
        # model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample / NIMA: instead of DS, we feed the downsampled_bic image (1/4)
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

            # after 4 downsampling
        # output: 1x128x240x248
        # output: 1x256x120x124
        # output: 1x256x60x62
        # output: 1x1024x30x31

        ### resnet blocks
        mult = 2 ** n_downsampling #n_downsampling=0则为1
        for i in range(n_blocks): #9个residual block
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # n_downsampling=1

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)] #nn.Tanh()
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # print(x.size())
        out = x + self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        # tt
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
                #############################################################################################

class RateDistortionLoss(nn.Module):  # 只注释掉了109行的bpp_loss, 08021808又加上了
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    # def forward(self, output, target, lq, x_l, x_enh): #0.001
    def forward(self, output, target, height, width):  # 0.001 #, lq, x_l, x_enh
        # N, _, _, _ = target.size()
        N, _, H, W = target.size()
        out = {}
        num_pixels_feature = N * H * W
        num_pixels = N * height * width
        print('ratedistortion functions: image hxw: %dx%d, num_pixel: %d' % (height, width, num_pixels))

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        bpp_temp = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels_feature))
            for likelihoods in output["likelihoods"].values()
        )
        print('ratedistortion functions: bpp_img/bpp_feat: %8.4f/%8.4f' % (out["bpp_loss"].item(), bpp_temp))
        # # out["mse_loss"] = self.mse(lq, target)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]  # lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out

def Pfeature_zeropad_youxiajiao256(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    if h % factor == 0:
        h_new = h
    else:
        h_new = ((h // factor) + 1) * factor
    if w % factor == 0:
        w_new = w
    else:
        w_new = ((w // factor) + 1) * factor
    #加了下面4行让hw_new最小为128
    if h_new < 256:
        h_new = 256
    if w_new < 256:
        w_new = 256
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao256_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """

        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        ############################################################################20220519
        self.gpu_ids = [0,1,2]
        # self.compG = define_compG(256, 256, 64, 3, norm='instance', gpu_ids=self.gpu_ids)
        # self.netG = define_G(256, 256, 64, 'global', 3, 9, 1, 3, 'instance', gpu_ids=self.gpu_ids)
        # #self.netD = define_D(3, 64, 3, 'instance', 'store_true', 2, getIntermFeat=False, gpu_ids=self.gpu_ids)
        # #self.compG = define_compG(netG_input_nc, opt.output_nc, opt.ncf, opt.n_downsample_comp, norm=opt.norm, gpu_ids=self.gpu_ids)
        # #self.netG = define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        # #self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        # ############################################################################20220519
        # #####ccr added
        
        # ## ywz add for CAI
        # self.CAI = main_cai()

        for p in self.backbone.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.backbone)

        for p in self.proposal_generator.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.proposal_generator)

        for p in self.roi_heads.parameters():
            p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.roi_heads)
        ############################
        ############################belle
        compressaiargs_experiment = 'rcnn_belle_0730'
        if not os.path.exists(os.path.join('experiments', compressaiargs_experiment)):
            os.makedirs(os.path.join('experiments', compressaiargs_experiment))
        # compressaiargs_model = "bmshj2018-hyperprior"
        # compressaiargs_quality = 4 #default=1 #指令里
        compressaiargs_learning_rate = 0.0001 #指令里
        compressaiargs_aux_learning_rate = 0.001 #new_train.py的parse_args
        ######lambda设置的取值
        # compressaiargs_lambda = 8.0
        compressaiargs_lambda = 4.0
        # compressaiargs_lambda = 2.0
        # compressaiargs_lambda = 1.0
        # compressaiargs_lambda = 0.512
        # compressaiargs_lambda = 0.256
        # compressaiargs_lambda = 0.128
        # compressaiargs_lambda = 0.064
        #####################
        self.belle_clip_max_norm = 1.0
        # # self.net_belle = image_models[compressaiargs_model](quality=compressaiargs_quality)
        # self.net_belle = ScaleHyperpriorMulti(M=192, N=128)  # M=192, N=128,  320,192
        # # self.net_belle = ScaleHyperpriorMulti(M=512, N=320)
        # self.net_belle = Cheng2020AttentionMulti(N=192)
        self.net_belle = Cheng2020Anchor(N=192)
        self.net_belle = self.net_belle.to(device)
        # print('#####################################PRINT NET_BELLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(self.net_belle)
        self.optimizer_belle, self.belle_aux_optimizer = train_in_this_copyuseful.configure_optimizers(self.net_belle, compressaiargs_learning_rate, compressaiargs_aux_learning_rate)
        # self.belle_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_belle, milestones=[2000, 4250], gamma=0.5)
        self.belle_criterion = train_in_this_copyuseful.RateDistortionLoss(compressaiargs_lambda)
        self.i_step_count = 0
        # compressai_logdir = '/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM/tensorboard_belle/EXP_cheng2020attn_256chinput_P2345MSE_lambda1_N192_smalltrain5W_eachdnorm_95kcontinue_08191910'
        # compressai_logdir = '/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM/tensorboard_belle/EXP_cheng2020attn_256chinput_P2345MSE_lambda1_N192_7imgtrain_eachdnorm_08192010'
        # compressai_logdir = '/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_lambda1_N192_7imgtrainft12999_small5Wtrain_eachdnorm_08211030'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiao128_lambda1_N192_small5Wtrain_eachdnorm_08271100/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiao128_lambda1_N192_small5Wtrain_eachdnorm_08271100_29kcontinue/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1_N192_small5Wtrain_eachdnorm_09011100/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1_N192_7imgtrainft14999_small5Wtrain_eachdnorm_09011500/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1_N192_7imgtrainft14999_small5Wtrain_eachdnorm_12kcontinue_09021000/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1_N192_7imgtrainft14999_small5Wtrain_eachdnorm_12k64kcontinue_09051530/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda2_N192_7imgtrain_eachdnorm_09092000/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda2_N192_7imgtrainft14999_small5Wtrain_eachdnorm_09092200/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1chu2_N192_7imgtrain_eachdnorm_09191600/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1chu2_N192_7imgtrainft14999_small5Wtrain_eachdnorm_09191900/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1chu2_N192_lambda1ft135999_small5Wtrain_eachdnorm_09211430/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1chu4_N192_lambda1chu2ft3999_small5Wtrain_eachdnorm_09221600/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1chu4_N192_small5Wtrain_eachdnorm_10011030/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1chu8_N192_lambda1chu4ft3999_small5Wtrain_eachdnorm_09231300/'
        compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda1chu16_N192_lambda1chu8ft3999_small5Wtrain_eachdnorm_10142030/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda4_N192_lambda2ft125999_small5Wtrain_eachdnorm_09261700/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P2down2P345MSE_zeroyouxiajiaonew_lambda8_N192_lambda4ft3999_small5Wtrain_eachdnorm_10051600/'
        mkdirs(compressai_logdir)
        self.belle_writer = SummaryWriter(log_dir=compressai_logdir)
        self.belle_savetensorboardfreq = 200

        # 读取cocotest5000的numpixel文件用于算inference的bpp
        path_save = 'cocotest5000_numpixel.json' #new_dict[fname_simple][0] [1] [2] 分别为height, width, num_pixel fname_simple为 '000a1249af2bc5f0'
        tf = open(path_save, "r")
        self.numpixel_test5000 = json.load(tf)

        self.path_bppsave = '/media/data/liutie/VCM/rcnn/liutie_save/output/chenganchor_bpp_lambda1_cocofinetune_cocotest5000_hw576.json' ####记得改这里的lambda
        self.path_bppsave_p3 = '/media/data/liutie/VCM/rcnn/liutie_save/output/chenganchor_bpp_lambda1_cocofinetune_cocotest5000_hw576_P3.json'
        self.path_bppsave_p4 = '/media/data/liutie/VCM/rcnn/liutie_save/output/chenganchor_bpp_lambda1_cocofinetune_cocotest5000_hw576_P4.json'
        self.path_bppsave_p5 = '/media/data/liutie/VCM/rcnn/liutie_save/output/chenganchor_bpp_lambda1_cocofinetune_cocotest5000_hw576_P5.json'
        self.bpp_test5000 = {}
        self.bpp_test5000_p3 = {}
        self.bpp_test5000_p4 = {}
        self.bpp_test5000_p5 = {}
        # compressai_lmbda = 1.0
        self.criterion = RateDistortionLoss(lmbda=compressaiargs_lambda)

        ###加载训练好的CPmodule和PRmodule的lambda1模型
        # path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda1_models/model_0037999.pth' #lambda=1
        # path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda2_models/model_0037999.pth' #lambda=2
        # path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda4_models/model_0039999.pth' #lambda=4
        # path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda1chu2_models/model_0003999_CPPRmodule.pth' #lambda=1chu2
        # path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda1chu4_models/model_0037999.pth' #lambda=1chu4
        # path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda1chu8_models/model_0037999.pth' #lambda=1chu8
        # path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda4_models_cocofinetune/model_0019999.pth' #lambda=4
        path_savepth = '/media/data/liutie/VCM/rcnn/liutie_save/lambda1_models_cocofinetune/model_0019999.pth' #lambda=1
        self.CPmodule = Cheng2020Anchor_CPmodule(N=192)
        self.CPmodule = self.CPmodule.to(device)
        pretrained_dict = torch.load(path_savepth)
        model_dict = self.CPmodule.state_dict()
        pretrained_dict = {key[10:]: value for key, value in pretrained_dict['model'].items() if ('net_belle' in key)}  # 去掉前缀net_belle.
        model_dict.update(pretrained_dict)
        self.CPmodule.load_state_dict(model_dict)

        gpu_ids = [0, 1, 2]
        self.PRmodule = define_G(256, 256, 128, 'global', 0, 9, 1, 3, 'instance', gpu_ids=gpu_ids)  # 3->0 64->128
        self.PRmodule = self.PRmodule.to(device)
        pretrained_dict = torch.load(path_savepth)
        model_dict = self.PRmodule.state_dict()
        pretrained_dict = {key[5:]: value for key, value in pretrained_dict['model'].items() if ('netG' in key)}  # 去掉前缀netG.
        model_dict.update(pretrained_dict)
        self.PRmodule.load_state_dict(model_dict)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        # print(batched_inputs[0]['instances'],'---------------------------------batched_inputs')
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # print(images.tensor.size(),'---------------------------------images.tensor after preprocess size')
        features = self.backbone(images.tensor)
        # print(features["p2"].size(),'---------------------------------feature P2 output from backbone size')

        # #######################################################################################
        # #20220516 ccr added start
        # import torch.nn.functional as F
        # _scale = 23.4838
        # _min = -23.1728
        # def feature_slice(image, shape):
        #     height = image.shape[0]
        #     width = image.shape[1]
        #
        #     blk_height = shape[0]
        #     blk_width = shape[1]
        #     print(blk_height,blk_width,'--------------------h and w')
        #     blk = []
        #
        #     for y in range(height // blk_height):
        #         for x in range(width // blk_width):
        #             y_lower = y * blk_height
        #             y_upper = (y + 1) * blk_height
        #             x_lower = x * blk_width
        #             x_upper = (x + 1) * blk_width
        #             blk.append(image[y_lower:y_upper, x_lower:x_upper])
        #             print(type(blk),'-------------------------blk')
        #     #feature = torch.from_numpy(np.array(blk))
        #     #feature = torch.tensor(blk)
        #     feature = torch.tensor([item.cpu().detach().numpy() for item in blk]).cuda()
        #     return feature
        #
        # def quant_fix(features):
        #     for name, pyramid in features.items():
        #         pyramid_q = (pyramid - _min) * _scale
        #         features[name] = pyramid_q
        #     return features
        #
        # def dequant_fix(x):
        #     return x.type(torch.float32) / _scale + _min
        #
        # import cv2
        # #print(features["p2"].shape, '---------------------------------------features["p2"] size')
        # features_copy = features.copy()
        # features_draw = quant_fix(features_copy)
        # del features_draw["p6"]
        # feat = [features_draw["p2"].squeeze(), features_draw["p3"].squeeze(), features_draw["p4"].squeeze(),features_draw["p5"].squeeze()]
        # #print(features_draw["p2"].shape,'----------------------------------------------p2shape')
        # squeezedP2 = features_draw["p2"].squeeze()
        # #print(squeezedP2.shape, '----------------------------------------------squeezed p2shape')
        # splitp2 = torch.split(features_draw["p2"], 1, dim=0)
        # splitp3 = torch.split(features_draw["p3"], 1, dim=0)
        # splitp4 = torch.split(features_draw["p4"], 1, dim=0)
        # splitp5 = torch.split(features_draw["p5"], 1, dim=0)
        # listsplit = []
        # for p in features:
        #     if p=='p2' or p=='p3' or p=='p4':
        #         compG_input = features[p]
        #         comp_image = self.compG.forward(compG_input)
        #         upsample = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        #         up_image = upsample(comp_image)
        #         input_flabel = None
        #         input_fconcat = up_image
        #         res = self.netG.forward(input_fconcat)
        #         fake_image_f = res + up_image
        #         features[p] = fake_image_f

        # --------------------------------------------------
        # time1_start = time.time()
        device = self.device
        i_select_whichP = random.randint(2, 5)
        if i_select_whichP == 2:
            cai_input_tensor = features["p2"]  # float32
            cai_input_tensor = F.interpolate(cai_input_tensor, scale_factor=0.5, mode="bilinear", align_corners=False)  # [1, 256, h/4, w/4]->[1, 256, h/8, w/8]
            d_originalsize = cai_input_tensor
            print(cai_input_tensor.size(), '-------------------Select P2 (original size) (before padding)')
            cai_input_tensor_new, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_zeropad_youxiajiao128(cai_input_tensor, 64) #P2 zeroyouxiajiao128
        elif i_select_whichP == 3:
            cai_input_tensor = features["p3"]  # float32
            d_originalsize = cai_input_tensor
            print(cai_input_tensor.size(), '-------------------Select P3 (original size) (before padding)')
            cai_input_tensor_new, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_zeropad_youxiajiao128(cai_input_tensor, 64) #P3 zeroyouxiajiao128
        elif i_select_whichP == 4:
            cai_input_tensor = features["p4"]  # float32
            d_originalsize = cai_input_tensor
            print(cai_input_tensor.size(), '-------------------Select P4 (original size) (before padding)')
            cai_input_tensor_new, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_zeropad_youxiajiao(cai_input_tensor, 16) #P4 zeroyouxiajiao16
        elif i_select_whichP == 5:
            cai_input_tensor = features["p5"]  # float32
            d_originalsize = cai_input_tensor
            print(cai_input_tensor.size(), '-------------------Select P5 (original size) (before padding)')
            cai_input_tensor_new, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_zeropad_youxiajiao(cai_input_tensor, 16) #P5 zeroyouxiajiao16
        # # cai_input_tensor_p4 = features["p4"]  # float32
        # ###将P2crop成64的倍数
        # # h_p2 = cai_input_tensor.size()[2]
        # # w_p2 = cai_input_tensor.size()[3]
        # # h_p2_new = h_p2 // 64 * 64
        # # w_p2_new = w_p2 // 64 * 64
        # # target_size = [cai_input_tensor.size()[0], cai_input_tensor.size()[1], h_p2_new, w_p2_new]  # [b, 256, 128, 192]
        # # cai_input_tensor_new = torch.zeros(target_size)
        # # cai_input_tensor_new = cai_input_tensor[:, 0:cai_input_tensor_new.size()[1], 0:cai_input_tensor_new.size()[2], 0:cai_input_tensor_new.size()[3]]
        # # # print(cai_input_tensor_new.size(), '-------------------CAI P2 input new size')
        # # h_p4_new = int(h_p2_new / 4)
        # # w_p4_new = int(w_p2_new / 4)
        # # target_size_p4 = [cai_input_tensor_p4.size()[0], cai_input_tensor_p4.size()[1], h_p4_new, w_p4_new]  # [b, 256, 128 / 4, 192 / 4]
        # # cai_input_tensor_p4_new = torch.zeros(target_size_p4)
        # # cai_input_tensor_p4_new = cai_input_tensor_p4[:, 0:cai_input_tensor_p4_new.size()[1], 0:cai_input_tensor_p4_new.size()[2], 0:cai_input_tensor_p4_new.size()[3]]
        # # # print(cai_input_tensor_p4_new.size(), '-------------------CAI P4 input new size')
        # ###将P2pad成16的倍数
        # cai_input_tensor_new, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_zeropad_youxiajiao(cai_input_tensor, 16)
        ##cai_input_tensor_new, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_replicatepad(cai_input_tensor, 16)
        self.net_belle.train()
        d = cai_input_tensor_new #[:, channel_idx: (channel_idx + 1), :, :]
        d = d.to(device)
        d_originalsize = d_originalsize.to(device)
        guiyihua_min = torch.min(d)
        guiyihua_scale = torch.max(d) - torch.min(d)
        d = (d - guiyihua_min) / guiyihua_scale
        d_originalsize = (d_originalsize - guiyihua_min) / guiyihua_scale
        print(d.size(), '-------------------cheng input size')
        # d_p4 = cai_input_tensor_p4_new #[:, channel_idx: (channel_idx + 1), :, :]
        # d_p4 = d_p4.to(device)
        # guiyihua_min_p4 = torch.min(d_p4)
        # guiyihua_scale_p4 = torch.max(d_p4) - torch.min(d_p4)
        # d_p4 = (d_p4 - guiyihua_min_p4) / guiyihua_scale_p4
        # # d_p4 = (d_p4 + 12.5) / 25.0  # 2张图是-12.3-12.3
        # # d_p4 = (d_p4 - self.guiyihua_min) / self.guiyihua_scale
        # # print(d_p4.size(), '-------------------belle input (P4) size')
        # # time1_end = time.time()
        # # time1 = time1_end - time1_start
        # # time2_start = time.time()
        self.optimizer_belle.zero_grad()  # optimizer.zero_grad()
        self.belle_aux_optimizer.zero_grad()
        net_belle_output = self.net_belle(d)
        out_criterion = self.belle_criterion(net_belle_output, d)
        out_criterion["loss"].backward()
        if self.belle_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net_belle.parameters(), self.belle_clip_max_norm)
        self.optimizer_belle.step()

        aux_loss = self.net_belle.aux_loss()
        aux_loss.backward()
        self.belle_aux_optimizer.step()
        psnr_temp = 10 * math.log10(1 / out_criterion["mse_loss"].item())

        # d_output = Pfeature_zeropad_youxiajiao_reverse(net_belle_output["x_hat"], h_new_p4_left, h_new_p4_right, w_new_p4_left, w_new_p4_right)
        d_output = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output["x_hat"], h_new_left, h_new_right, w_new_left, w_new_right)
        define_mse = nn.MSELoss()
        mse_temp = define_mse(d_output, d_originalsize)
        psnr_temp_originalsize = 10 * math.log10(1 / mse_temp)
        print(
            f'Loss: {out_criterion["loss"].item():.3f} |'
            f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
            f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
            f"\tAux loss: {aux_loss.item():.2f}"
            f'\tPSNR: {psnr_temp:.3f} |'
            f'\tPSNR_orisize: {psnr_temp_originalsize:.3f} |'
        )
        print("i_step: %d, max/min(cheng input): %8.4f/%8.4f, max/min(cheng output): %8.4f/%8.4f"
              % (self.i_step_count, torch.max(d), torch.min(d), torch.max(net_belle_output["x_hat"]), torch.min(net_belle_output["x_hat"])))

        #net_belle_output["x_hat"]小于0置为0，大于1置为1

        if (self.i_step_count % self.belle_savetensorboardfreq == 0): # and (channel_idx == 0):
            i_select_channel = random.randint(0, 255)
            self.belle_writer.add_scalar("training/Loss", out_criterion["loss"], global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/MSE loss", out_criterion["mse_loss"], global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/Bpp loss", out_criterion["bpp_loss"], global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/Aux loss", aux_loss.item(), global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/PSNR", psnr_temp, global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/PSNR_orisize", psnr_temp_originalsize, global_step=self.i_step_count)
            self.belle_writer.add_image('images.tensor', images.tensor[0, :, :, :], global_step=self.i_step_count, dataformats='CHW')  # dataformats='HWC')
            self.belle_writer.add_image('Pfeature_GT', d[0, i_select_channel:(i_select_channel+1), :, :], global_step=self.i_step_count, dataformats='CHW')  # dataformats='HWC')
            # self.belle_writer.add_image('P4_GT', d_p4[0, i_select_channel:(i_select_channel+1), :, :], global_step=self.i_step_count, dataformats='CHW')  # dataformats='HWC')
            self.belle_writer.add_image('netcheng_output', net_belle_output["x_hat"][0, i_select_channel:(i_select_channel+1), :, :], global_step=self.i_step_count, dataformats='CHW')  # dataformats='HWC')

        self.i_step_count = self.i_step_count + 1

        # time2_end = time.time()
        # time2 = time2_end - time2_start
        #
        # time3_start = time.time()
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # time3_end = time.time()
        # time3 = time3_end - time3_start
        # print('time: p2preprocess/netbelle/RCNNhouduan: %8.4f/%8.4f/%8.4f' %(time1, time2, time3))
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)


        # for i in batched_inputs:
        #     print(i)
        # print(batched_inputs[0]["file_name"]) #datasets/coco/val2017/000000000139.jpg #inference时batchsize为1
        fname_temp = os.path.basename(batched_inputs[0]["file_name"])[0:-4] #000000000139
        # print(fname_temp)

        d_p2 = features['p2']  # [1, 256, 200, 304]
        d_p2_down2 = F.interpolate(d_p2, scale_factor=0.5, mode="bilinear", align_corners=False)
        d_p3 = features['p3']  # [1, 256, 200, 304]
        d_p4 = features['p4']  # [1, 256, 200, 304]
        d_p5 = features['p5']  # [1, 256, 200, 304]
        # normlize p2345
        max_temp = [torch.max(d_p2), torch.max(d_p3), torch.max(d_p4), torch.max(d_p5)]
        max_temp = torch.as_tensor(max_temp)
        min_temp = [torch.min(d_p2), torch.min(d_p3), torch.min(d_p4), torch.min(d_p5)]
        min_temp = torch.as_tensor(min_temp)
        guiyihua_max = torch.max(max_temp)
        guiyihua_min = torch.min(min_temp)
        guiyihua_scale = guiyihua_max - guiyihua_min
        ###pad
        # d_p2_new, h_p2_new_left, h_p2_new_right, w_p2_new_left, w_p2_new_right = Pfeature_zeropad_youxiajiao128(d_p2_down2, 64)  # P2 zeroyouxiajiao128
        # d_p3_new, h_p3_new_left, h_p3_new_right, w_p3_new_left, w_p3_new_right = Pfeature_zeropad_youxiajiao128(d_p3, 64)  # P3 zeroyouxiajiao128
        d_p2_new, h_p2_new_left, h_p2_new_right, w_p2_new_left, w_p2_new_right = Pfeature_zeropad_youxiajiao256(d_p2, 32)
        d_p3_new, h_p3_new_left, h_p3_new_right, w_p3_new_left, w_p3_new_right = Pfeature_zeropad_youxiajiao128(d_p3, 16)
        d_p4_new, h_p4_new_left, h_p4_new_right, w_p4_new_left, w_p4_new_right = Pfeature_zeropad_youxiajiao(d_p4, 16)  # P4 zeroyouxiajiao16
        d_p5_new, h_p5_new_left, h_p5_new_right, w_p5_new_left, w_p5_new_right = Pfeature_zeropad_youxiajiao(d_p5, 16)  # P5 zeroyouxiajiao16
        d_p2 = (d_p2 - guiyihua_min) / guiyihua_scale
        d_p2_down2 = (d_p2_down2 - guiyihua_min) / guiyihua_scale
        d_p3 = (d_p3 - guiyihua_min) / guiyihua_scale
        d_p4 = (d_p4 - guiyihua_min) / guiyihua_scale
        d_p5 = (d_p5 - guiyihua_min) / guiyihua_scale
        d_p2_new = (d_p2_new - guiyihua_min) / guiyihua_scale
        d_p3_new = (d_p3_new - guiyihua_min) / guiyihua_scale
        d_p4_new = (d_p4_new - guiyihua_min) / guiyihua_scale
        d_p5_new = (d_p5_new - guiyihua_min) / guiyihua_scale
        # net_belle_output_p2 = self.net_belle(d_p2_new)
        net_belle_output_p3 = self.CPmodule(d_p2_new)
        d_output_p3 = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output_p3["x_hat"], h_p3_new_left, h_p3_new_right, w_p3_new_left, w_p3_new_right)

        fake_image_f_GT = d_p2
        upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        up_image = upsample(net_belle_output_p3["x_hat"])
        res = self.PRmodule.forward(up_image)
        fake_image_f = res + up_image
        d_output_p2 = Pfeature_zeropad_youxiajiao256_reverse(fake_image_f, h_p2_new_left, h_p2_new_right, w_p2_new_left, w_p2_new_right)
        print(d_output_p2.size(), '-------------------Finenet output P2 size')
        print('max/min_P3up(Finenet input): %8.4f/%8.4f, max/min_P2(GT): %8.4f/%8.4f, max/min_P2(Finenet output): %8.4f/%8.4f' % (
            torch.max(up_image), torch.min(up_image), torch.max(d_p2), torch.min(d_p2), torch.max(d_output_p2), torch.min(d_output_p2)))

        net_belle_output_p4 = self.net_belle(d_p4_new)
        d_output_p4 = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output_p4["x_hat"], h_p4_new_left, h_p4_new_right, w_p4_new_left, w_p4_new_right)
        net_belle_output_p5 = self.net_belle(d_p5_new)
        d_output_p5 = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output_p5["x_hat"], h_p5_new_left, h_p5_new_right, w_p5_new_left, w_p5_new_right)

        print('max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P2(Cheng output): %8.4f/%8.4f' % (torch.max(d_p2_new), torch.min(d_p2_new), torch.max(d_output_p2), torch.min(d_output_p2)))
        print('max/min_P3(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f' % (torch.max(d_p3_new), torch.min(d_p3_new), torch.max(d_output_p3), torch.min(d_output_p3)))
        print('max/min_P4(GT)(Cheng input): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f' % (torch.max(d_p4_new), torch.min(d_p4_new), torch.max(d_output_p4), torch.min(d_output_p4)))
        print('max/min_P5(GT)(Cheng input): %8.4f/%8.4f, max/min_P5(Cheng output): %8.4f/%8.4f' % (torch.max(d_p5_new), torch.min(d_p5_new), torch.max(d_output_p5), torch.min(d_output_p5)))

        features_cheng = features.copy()
        features_p345 = features.copy()
        features_cheng["p2"] = d_output_p2 * guiyihua_scale + guiyihua_min
        features_p345["p3"] = d_output_p3 * guiyihua_scale + guiyihua_min
        features_p345["p4"] = d_output_p4 * guiyihua_scale + guiyihua_min
        features_p345["p5"] = d_output_p5 * guiyihua_scale + guiyihua_min
        print('After denormlize: max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P2(Cheng output): %8.4f/%8.4f' % (
        torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_cheng["p2"]), torch.min(features_cheng["p2"])))
        print('After denormlize: max/min_P3(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f' % (
        torch.max(features["p3"]), torch.min(features["p3"]), torch.max(features_p345["p3"]), torch.min(features_p345["p3"])))
        print('After denormlize: max/min_P4(GT)(Cheng input): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f' % (
        torch.max(features["p4"]), torch.min(features["p4"]), torch.max(features_p345["p4"]), torch.min(features_p345["p4"])))
        print('After denormlize: max/min_P5(GT)(Cheng input): %8.4f/%8.4f, max/min_P5(Cheng output): %8.4f/%8.4f' % (
        torch.max(features["p5"]), torch.min(features["p5"]), torch.max(features_p345["p5"]), torch.min(features_p345["p5"])))
        # cheng_feat = quant_fix(features_cheng.copy())

        # fname_temp = utils.simple_filename(inputs[0]["file_name"])
        # heigh_temp = self.height_temp
        # width_temp = self.width_temp
        # numpixel_temp = self.numpixel_temp
        heigh_temp = self.numpixel_test5000[fname_temp][0]
        width_temp = self.numpixel_test5000[fname_temp][1]
        numpixel_temp = self.numpixel_test5000[fname_temp][2]
        out_criterion_p3 = self.criterion(net_belle_output_p3, d_p3_new, heigh_temp, width_temp)
        out_criterion_p4 = self.criterion(net_belle_output_p4, d_p4_new, heigh_temp, width_temp)
        out_criterion_p5 = self.criterion(net_belle_output_p5, d_p5_new, heigh_temp, width_temp)
        print('image hxw: %dx%d, num_pixel: %d' % (heigh_temp, width_temp, numpixel_temp))
        define_mse = nn.MSELoss()
        out_criterion_p3["mse_loss"] = define_mse(d_output_p3, d_p3)
        psnr_temp = train_in_this_copyuseful.mse2psnr(out_criterion_p3["mse_loss"])
        print('[P3] bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion_p3["bpp_loss"].item(), out_criterion_p3["mse_loss"].item(), psnr_temp))
        define_mse = nn.MSELoss()
        out_criterion_p4["mse_loss"] = define_mse(d_output_p4, d_p4)
        psnr_temp = train_in_this_copyuseful.mse2psnr(out_criterion_p4["mse_loss"])
        print('[P4] bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion_p4["bpp_loss"].item(), out_criterion_p4["mse_loss"].item(), psnr_temp))
        define_mse = nn.MSELoss()
        out_criterion_p5["mse_loss"] = define_mse(d_output_p5, d_p5)
        psnr_temp = train_in_this_copyuseful.mse2psnr(out_criterion_p5["mse_loss"])
        print('[P5] bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion_p5["bpp_loss"].item(), out_criterion_p5["mse_loss"].item(), psnr_temp))
        bpp_p2345_temp = out_criterion_p3["bpp_loss"].item() + out_criterion_p4["bpp_loss"].item() + out_criterion_p5["bpp_loss"].item()
        print('[P2345] bpp: %8.4f' % (bpp_p2345_temp))
        self.bpp_test5000[fname_temp] = [bpp_p2345_temp]
        self.bpp_test5000_p3[fname_temp] = [out_criterion_p3["bpp_loss"].item()]
        self.bpp_test5000_p4[fname_temp] = [out_criterion_p4["bpp_loss"].item()]
        self.bpp_test5000_p5[fname_temp] = [out_criterion_p5["bpp_loss"].item()]
        ###bpp_all
        tf = open(self.path_bppsave, "w")
        json.dump(self.bpp_test5000, tf)
        tf.close()
        ###bpp_p3
        tf_p3 = open(self.path_bppsave_p3, "w")
        json.dump(self.bpp_test5000_p3, tf_p3)
        tf_p3.close()
        ###bpp_p4
        tf_p4 = open(self.path_bppsave_p4, "w")
        json.dump(self.bpp_test5000_p4, tf_p4)
        tf_p4.close()
        ###bpp_p5
        tf_p5 = open(self.path_bppsave_p5, "w")
        json.dump(self.bpp_test5000_p5, tf_p5)
        tf_p5.close()
        #################################
        #################################

        # image_feat = quant_fix(features_p345.copy())
        #
        # fname = utils.simple_filename(inputs[0]["file_name"])
        # fname_feat = f"../../liutie_save/feature/{self.set_idx}_ori/{fname}.png"  # 用于存P345
        # fname_ds = f"../../liutie_save/feature/{self.set_idx}_ds/{fname}.png"  # 用于存P2
        #
        # with open(f"../../liutie_save/info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
        #     torch.save(inputs, inputs_f)

        # ####################################ccr added 3 parts
        # utils.save_feature_map_onlyp2(fname_ds, cheng_feat)  # 用于存P2
        # utils.save_feature_map_p345(fname_feat, image_feat)  # 用于存P345
        # return fname_feat

        ##进入RCNN后端前给feature赋为压缩后的特征
        features["p2"] = d_output_p2 * guiyihua_scale + guiyihua_min
        features["p3"] = d_output_p3 * guiyihua_scale + guiyihua_min
        features["p4"] = d_output_p4 * guiyihua_scale + guiyihua_min
        features["p5"] = d_output_p5 * guiyihua_scale + guiyihua_min

        if detected_instances is None:
            if self.proposal_generator is not None: #走这里
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess: #走这里
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        print(features['p5'].size(), '----------------------------------------------------------------------ProposalNetwork features')

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
