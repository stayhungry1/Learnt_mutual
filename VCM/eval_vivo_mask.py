###P2inP3out
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
import subprocess
# import utils
import utils_copy
from quantizer import quant_fix, dequant_fix#
# from VTM_encoder import run_vtm
from VTM_encoder_ccr import run_vtm
from cvt_detectron_coco_oid_vivo import conversion
import scipy.io as sio
from typing import Tuple, Union
import PIL.Image as Image
import math
import json
import scipy.io as sio
import Anchors.oid_mask_encoding as oid_mask_encoding
import sys
# sys.path.append("/media/data/liutie/VCM/rcnn/VCMbelle_0622")
# sys.path.append("/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM")
sys.path.append("/media/data/ccr/liutieCompressAI/")
sys.path.append("/media/data/ccr/liutieCompressAI/VCM/")
from compressai.datasets.parse_p2_feature import _joint_split_features
from examples.train_in_this import * #只用到了IRN_inference和main_cai
#
# main_cai()
# # --add compressai framework done

import functools
from torch.autograd import Variable
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from compressai.models import *

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
        # netG.to(device)
        netG.cuda()
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

def Pfeature_zeropad_youxiajiao64(feat, factor=16): #相比于Pfeature_replicatepad的区别为pad从上下左右变为右下角 输入feat为[b, 256, h, w]
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
    #加了下面4行让hw_new最小为64
    if h_new < 64:
        h_new = 64
    if w_new < 64:
        w_new = 64
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)

def Pfeature_zeropad_youxiajiao64_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def Pfeature_zeropad_youxiajiao32(feat, factor=16):
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
    if h_new < 32:
        h_new = 32
    if w_new < 32:
        w_new = 32
    h_new_left = 0 #(h_new - h) // 2
    h_new_right = h_new - h
    w_new_left = 0
    w_new_right = w_new - w
    # nn.ReplicationPad2d((1, 2, 3, 2))  #左侧填充1行，右侧填充2行，上方填充3行，下方填充2行
    pad_func = nn.ZeroPad2d((w_new_left, w_new_right, h_new_left, h_new_right))
    feat_pad = pad_func(feat)
    return feat_pad, h_new_left, h_new_right, w_new_left, w_new_right #恢复时h_new_left:(h_now-h_right)


def Pfeature_zeropad_youxiajiao32_reverse(feat, h_new_left, h_new_right, w_new_left, w_new_right): #输入feat为[b, 256, h, w]
    h = feat.size()[2]
    w = feat.size()[3]
    feat_new = feat[:, :, h_new_left:(h-h_new_right), w_new_left:(w-w_new_right)]
    return feat_new

def padding_size(ori_size, factor_size):
    if ori_size % factor_size == 0:
        return ori_size
    else:
        return factor_size * (ori_size // factor_size + 1)


def mse2psnr(mse):
    # 根据Hyper论文中的内容，将MSE->psnr(db)
    # return 10*math.log10(255*255/mse)
    return 10 * math.log10(1 / mse)


def compute_metrics(
        a: Union[np.array, Image.Image],
        b: Union[np.array, Image.Image],
        max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    # m = ms_ssim(a, b, data_range=max_val).item()
    m = 0
    return p, m


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


class RateDistortionLoss_P2(nn.Module):  #
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
        out["mse_loss"] = self.mse(output["x_hat_p2"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]  # lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out

class RateDistortionLoss_P4(nn.Module):  #
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
        out["mse_loss"] = self.mse(output["x_hat_p4"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]  # lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out

class RateDistortionLoss_P5(nn.Module):  #
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
        out["mse_loss"] = self.mse(output["x_hat_p5"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]  # lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out


class Eval:
    def __init__(self, settings, index) -> None:
        self.settings = settings
        self.set_idx = index
        self.VTM_param = settings["VTM"]
        print('load model path: %s' % (settings["pkl_path"]))
        self.model, self.cfg = utils_copy.model_loader(settings)  # load模型进来
        self.prepare_dir()
        utils_copy.print_settings(settings, index)

        self.pixel_num = settings["pixel_num"]

        compressai_lmbda = 1.0
        # ###P2inP2345out
        # self.criterion_p3 = RateDistortionLoss(lmbda=compressai_lmbda)
        # self.criterion_p2 = RateDistortionLoss_P2(lmbda=compressai_lmbda)
        # self.criterion_p4 = RateDistortionLoss_P4(lmbda=compressai_lmbda)
        # self.criterion_p5 = RateDistortionLoss_P5(lmbda=compressai_lmbda)
        ##oldversion: P2inP3out
        self.criterion = RateDistortionLoss(lmbda=compressai_lmbda)

        # 读取文件
        # path_save = './dataset/annotations_5k/newOpenimage_numpixel_text5000.json'  # new_dict[fname_simple][0] [1] [2] 分别为height, width, num_pixel fname_simple为 '000a1249af2bc5f0'
        path_save = '/media/data/ccr/zzf/vcm/VCM/dataset/annotations_5k/newOpenimage_numpixel_text5000.json'  # new_dict[fname_simple][0] [1] [2] 分别为height, width, num_pixel fname_simple为 '000a1249af2bc5f0'
        tf = open(path_save, "r")
        self.numpixel_test5000 = json.load(tf)
        # 第一版结果
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda16_iter0011999.json' # 15
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda4_iter0065999.json' # 12
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda2_iter0034999.json' # 11
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter0054999.json' # 10
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda0512_iter0050999.json' # 13
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda0256_iter0038999.json' # 14
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda0256_iter0038999.json' # 14

        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda16_iter0008999.json' # 18
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda4_iter0059999.json' # 19
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda16_new_iter0014999.json' # 20
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter0049999.json' # 41
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter0044999.json' # 42
        # self.path_bppsave = '../../zzf_save/output/new_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter0039999.json' # 43



        # 重新第二版训练结果
        # self.path_bppsave = '../../zzf_save/output/new2_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter0044999.json' # 21
        # self.path_bppsave = '../../zzf_save/output/new2_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter_plus0008999.json' # 22
        # self.path_bppsave = '../../zzf_save/output/new2_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter_plus0020999.json' # 22
        self.path_bppsave = '../../liutie_save/output/cheng_P2inP3outzeropad128_bpp_lambda4_finenet_iter37999.json'

        # 在第一版基础上finetune的第三版训练结果
        # self.path_bppsave = '../../zzf_save/output/new3_mask_cheng_P2inP2P3P4P5out_P2zeropad256_bpp_lambda1_iter0038999.json' # 31

        self.bpp_test5000 = {}
        self.seg = True


        ###多加载object detection训练好的CP module (P2inP3out)和PR module (P3upinP2out进来)
        self.net_belle_od = Cheng2020Anchor(N=192).cuda()
        path_saveCPmodule = '/media/data/ccr/liutie_save/output/EXP_cheng2020anchor_256chinput_P2inP3outMSE_P2zeroyouxiajiao256_lambda1_N192_7imgtrainft9999_small5Wtrain_eachdnorm_finenet_09062230/model_0037999_CPmodule.pth'
        self.net_belle_od.load_state_dict(torch.load(path_saveCPmodule))
        # self.net_belle_od = torch.load(path_saveCPmodule)
        print(next(self.net_belle_od.parameters()).device)
        print('load CPmodule trained by od: %s' %(path_saveCPmodule))

        gpu_ids = [0, 1, 2]
        self.netG_od = define_G(256, 256, 128, 'global', 0, 9, 1, 3, 'instance', gpu_ids=gpu_ids).cuda()  # 3->0 64->128
        path_savePRmodule = '/media/data/ccr/liutie_save/output/EXP_cheng2020anchor_256chinput_P2inP3outMSE_P2zeroyouxiajiao256_lambda1_N192_7imgtrainft9999_small5Wtrain_eachdnorm_finenet_09062230/model_0037999_PRmodule.pth'
        self.netG_od.load_state_dict(torch.load(path_savePRmodule))
        # self.netG_od = torch.load(path_savePRmodule)
        print(next(self.net_belle_od.parameters()).device)
        print('load PRmodule trained by od: %s' %(path_savePRmodule))

    def prepare_dir(self):
        os.makedirs(f"../../zzf_save/info/{self.set_idx}", exist_ok=True)
        os.makedirs(f"../../zzf_save/feature/{self.set_idx}_ori", exist_ok=True)
        os.makedirs(f"../../zzf_save/feature/{self.set_idx}_ds", exist_ok=True)
        # os.makedirs(f"feature/{self.set_idx}_resid", exist_ok=True)
        os.makedirs(f"../../zzf_save/output", exist_ok=True)

    def forward_front(self, inputs, images, features):
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        return self.model._postprocess(results, inputs, images.image_sizes)

    def feature_coding(self):
        print("Saving features maps...")
        print('min/max_test_size:%d/%d' % (self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST))
        filenames = []
        with tqdm(total=len(self.data_loader.dataset)) as pbar:
            for inputs in iter(self.data_loader):
                # 自己加入的5行，断了之后重新跑，提过feature的不用再提
                fname_temp = utils_copy.simple_filename(inputs[0]["file_name"])
                self.height_temp = self.numpixel_test5000[fname_temp][0]
                self.width_temp = self.numpixel_test5000[fname_temp][1]
                self.numpixel_temp = self.numpixel_test5000[fname_temp][2]
                fea_path = f"../../zzf_save/feature/{self.set_idx}_ori/"
                if os.path.isfile(f"{fea_path}{fname_temp}.png"):
                    print(f"feature extraction: {fname_temp} skip (exist)")
                    continue
                filenames.append(self._feature_coding(inputs, fname_temp))  # inputs是filename 15d64d, height 680, width 1024, image_id 19877和image这个tensor [3, 800, 1205] uint8 大于1
                pbar.update()
        tf = open(self.path_bppsave, "r")
        bpp_test5000 = json.load(tf)
        bpp_sum = 0
        i_count = 0
        for key in bpp_test5000:
            bpp_temp = bpp_test5000[key]
            bpp_sum = bpp_sum + bpp_temp[0]
            i_count = i_count + 1
            print('i_count: %d, bpp: %8.4f, %s' % (i_count, bpp_test5000[key][0], key))
        print('average bpp: %9.6f' % (bpp_sum / i_count))
        print("####################### NOT run VTM!!! ###############################")
        # print("runvtm---------------------runvtmrunvtmrunvtmrunvtmrunvtmrunvtmrunvtm")
        # run_vtm(f"feature/{self.set_idx}_ori", self.VTM_param["QP"], self.VTM_param["threads"])

        return filenames

    def _feature_coding(self, inputs, fname_temp):
        # #加了这一行
        # self.model.net_belle.eval()
        # # self.model.net_belle.train()

        images = self.model.preprocess_image(inputs)  # images: device cpu, image_sizes [800, 1205] tensor [1, 3, 800, 1216] torch.float32 cpu
        features = self.model.backbone(images.tensor)
        height_originalimage = images.image_sizes[0]
        width_originalimage = images.image_sizes[0]

        d_p2 = features['p2']  # [1, 256, 200, 304]
        d_p3 = features['p3']
        d_originalsize_p2 = d_p2
        d_originalsize_p3 = d_p3
        print(d_p2.size(), '-------------------P2 original size')
        #normlize p3 and p2
        if torch.min(d_p2) >= torch.min(d_p3): #2个数中取小的
            guiyihua_min = torch.min(d_p3)
        else:
            guiyihua_min = torch.min(d_p2)
        if torch.max(d_p2) >= torch.max(d_p3): #2个数中取大的
            guiyihua_max = torch.max(d_p2)
        else:
            guiyihua_max = torch.max(d_p3)
        guiyihua_scale = guiyihua_max - guiyihua_min
        ###pad
        # d_originalsize = d
        # d, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_zeropad_youxiajiao128(d, 16)
        # d_p2, _, _, _, _ = Pfeature_zeropad_youxiajiao128(d_p2, 16)
        d_p2, h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right = Pfeature_zeropad_youxiajiao256(d_p2, 32)
        d_p3, h_new_p3_left, h_new_p3_right, w_new_p3_left, w_new_p3_right = Pfeature_zeropad_youxiajiao128(d_p3, 16)
        d_p2 = (d_p2 - guiyihua_min) / guiyihua_scale
        d_p3 = (d_p3 - guiyihua_min) / guiyihua_scale
        d_originalsize_p2 = (d_originalsize_p2 - guiyihua_min) / guiyihua_scale
        d_originalsize_p3 = (d_originalsize_p3 - guiyihua_min) / guiyihua_scale
        print(d_p2.size(), '-------------------Cheng input (P2) size')
        # # normlize p2 and p4
        # if torch.min(d) >= torch.min(d_p4):  # 2个数中取小的
        #     guiyihua_min = torch.min(d_p4)
        # else:
        #     guiyihua_min = torch.min(d)
        # if torch.max(d) >= torch.max(d_p4):  # 2个数中取大的
        #     guiyihua_max = torch.max(d)
        # else:
        #     guiyihua_max = torch.max(d_p4)
        # guiyihua_scale = guiyihua_max - guiyihua_min
        # d = (d - guiyihua_min) / guiyihua_scale
        # d_p4 = (d_p4 - guiyihua_min) / guiyihua_scale
        # print(d.size(), '-------------------P2 original size')
        # temp_ori_size_p2 = d.shape  # P2原始尺寸
        # temp_ori_size_p4 = d_p4.shape  # P4原始尺寸
        # target_size_p2 = [d.size()[0], d.size()[1], padding_size(d.size()[2], 16), padding_size(d.size()[3], 16)]  # P2补黑边后(16的倍数) [1, 256, 208, 304]
        # d_big = torch.zeros(target_size_p2).cuda()
        # d_big[:, 0:temp_ori_size_p2[1], 0:temp_ori_size_p2[2], 0:temp_ori_size_p2[3]] = d
        # print(d_big.size(), '-------------------Cheng input (P2) size')
        # target_size_p4 = [d_p4.size()[0], d_p4.size()[1], int(target_size_p2[2] / 4.0), int(target_size_p2[3] / 4.0)]  # P2的1/4
        # d_big_p4 = torch.zeros(target_size_p4).cuda()
        # d_big_p4[:, 0:temp_ori_size_p4[1], 0:temp_ori_size_p4[2], 0:temp_ori_size_p4[3]] = d_p4
        # d_output = torch.zeros(temp_ori_size_p4)  # 用于从网络输出的tensor取出左上角
        net_belle_output = self.net_belle_od(d_p2)
        # net_belle_output = self.model.net_belle(d_p2)
        print(net_belle_output["x_hat"].size(), '-------------------Cheng output (P3) size')
        d_output = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output["x_hat"], h_new_p3_left, h_new_p3_right, w_new_p3_left, w_new_p3_right)
        print('max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f'
              % (torch.max(d_p2), torch.min(d_p2), torch.max(d_output), torch.min(d_output)))
        print(d_output.size(), '-------------------output size')
        features_cheng = features.copy()
        features_p345 = features.copy()
        # features_cheng["p4"] = d_output * guiyihua_scale + guiyihua_min
        features_p345["p3"] = d_output * guiyihua_scale + guiyihua_min
        # print('After denormlize: max/min_p2(GT)(Cheng input): %8.4f/%8.4f, max/min_p4(GT): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features["p4"]), torch.min(features["p4"]), torch.max(features_p345["p4"]), torch.min(features_p345["p4"])))
        print('After denormlize: max/min_P3(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f'
              % (torch.max(features["p3"]), torch.min(features["p3"]), torch.max(features_p345["p3"]), torch.min(features_p345["p3"])))

        fake_image_f_GT = d_p2
        upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        up_image = upsample(net_belle_output["x_hat"])
        res = self.netG_od.forward(up_image)
        # res = self.model.netG.forward(up_image)
        fake_image_f = res + up_image
        d_output_p2 = Pfeature_zeropad_youxiajiao256_reverse(fake_image_f, h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right)
        print(d_output_p2.size(), '-------------------Finenet output P2 size')
        print('max/min_P3up(Finenet input): %8.4f/%8.4f, max/min_P2(GT): %8.4f/%8.4f, max/min_P2(Finenet output): %8.4f/%8.4f' %(torch.max(up_image), torch.min(up_image), torch.max(d_p2), torch.min(d_p2), torch.max(d_output_p2), torch.min(d_output_p2)))

        features_cheng["p2"] = d_output_p2 * guiyihua_scale + guiyihua_min
        cheng_feat = quant_fix(features_cheng.copy())
        print('After denormlize: max/min_P2(GT): %8.4f/%8.4f, max/min_P2(Finenet output): %8.4f/%8.4f' %(torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_cheng["p2"]), torch.min(features_cheng["p2"])))

        l_l2 = torch.nn.MSELoss().cuda()
        loss_l2 = l_l2(d_output_p2, d_originalsize_p2)
        psnr_temp1 = 10 * math.log10(1 / loss_l2)

        up_image = Pfeature_zeropad_youxiajiao256_reverse(up_image, h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right)
        loss_l2_0 = l_l2(up_image, d_originalsize_p2)
        psnr_temp1_0 = 10 * math.log10(1 / loss_l2_0)
        dpsnr_temp = psnr_temp1 - psnr_temp1_0

        up_image_P3GT = upsample(d_p3)
        up_image_P3GT = Pfeature_zeropad_youxiajiao256_reverse(up_image_P3GT, h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right)
        loss_l2_P3GT = l_l2(up_image_P3GT, d_originalsize_p2)
        psnr_temp1_P3GT = 10 * math.log10(1 / loss_l2_P3GT)

        heigh_temp = self.height_temp
        width_temp = self.width_temp
        numpixel_temp = self.numpixel_temp
        out_criterion = self.criterion(net_belle_output, d_p3, heigh_temp, width_temp) #net_belle_output和d为pad后的
        print('image hxw: %dx%d, num_pixel: %d' % (heigh_temp, width_temp, numpixel_temp))
        # define_mse = nn.MSELoss()
        # net_belle_output["x_hat"] = d_output  # [1, 256, 208, 304]->[1, 256, 200, 304]
        # out_criterion["mse_loss"] = define_mse(net_belle_output["x_hat"], d_p4)
        # psnr_temp = mse2psnr(out_criterion['mse_loss'])
        # print('bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion["bpp_loss"].item(), out_criterion["mse_loss"].item(), psnr_temp))
        define_mse = nn.MSELoss()
        out_criterion["mse_loss"] = define_mse(d_output, d_originalsize_p3)
        psnr_temp = mse2psnr(out_criterion["mse_loss"])
        print('bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion["bpp_loss"].item(), out_criterion["mse_loss"].item(), psnr_temp))
        self.bpp_test5000[fname_temp] = [out_criterion["bpp_loss"].item()]
        tf = open(self.path_bppsave, "w")
        json.dump(self.bpp_test5000, tf)
        tf.close()

        print("FINENET MSE:%8.4f, (ori)dpsnr/psnr/psnr0: %8.4f/%8.4f/%8.4f, psnr_useP3GT: %8.4f, max/min_P2(GT): %8.4f/%8.4f, max/min_P3up(Finenet input): %8.4f/%8.4f, max/min_P2(FineNet output): %8.4f/%8.4f"
            % (loss_l2, dpsnr_temp, psnr_temp1, psnr_temp1_0, psnr_temp1_P3GT, torch.max(fake_image_f_GT), torch.min(fake_image_f_GT), torch.max(up_image), torch.min(up_image), torch.max(fake_image_f), torch.min(fake_image_f)))

        ##features_resid = features.copy()
        ##features_resid["p2"] = resid_pic
        ##resid_feat = quant_fix(features_resid.copy())
        #################################ccr added

        image_feat = quant_fix(features_p345.copy())

        fname = utils_copy.simple_filename(inputs[0]["file_name"])
        # fname_feat = f"../../liutie_save/feature/{self.set_idx}_ori/{fname}.png"  # 用于存P345
        fname_feat = f"../../zzf_save/feature/{self.set_idx}_ori/{fname}.png"  # 用于存P345
        # fname_ds = f"../../liutie_save/feature/{self.set_idx}_ds/{fname}.png"  # 用于存P2
        fname_ds = f"../../zzf_save/feature/{self.set_idx}_ds/{fname}.png"  # 用于存P2

        # with open(f"../../liutie_save/info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
        with open(f"../../zzf_save/info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
            torch.save(inputs, inputs_f)

        # utils.save_feature_map(fname_feat, image_feat)
        ####################################ccr added 3 parts
        utils_copy.save_feature_map_onlyp2(fname_ds, cheng_feat)  # 用于存P2
        # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        utils_copy.save_feature_map_p345(fname_feat, image_feat)  # 用于存P345
        ####################################liutie added 3 parts

        #  #################################ccr added
        #  compG_input = features['p2']
        #  print(compG_input.size(),'---------------CompNet_input')
        #  comp_image = self.model.compG.forward(compG_input)
        #  ####### replace CompNet(last sentence), to compare onlyCompNet(Down 2) and P2down2
        #  #comp_image = F.interpolate(compG_input, scale_factor=0.5, mode="bilinear", align_corners=False)  # [1, 256, h/4, w/4]->[1, 256, h/8, w/8]
        #  ##print( comp_image.size(),'-------------------- comp_image')
        #  upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        #  up_image = upsample(comp_image)
        # ## print(up_image.size(),'--------------------------up_image')
        # # input_fconcat = up_image
        # # res = self.model.netG.forward(input_fconcat)
        # ## print(res.size(),'------------------------res')
        #  ##fake_image_f = res + up_image
        #  ##resid_pic = compG_input - fake_image_f
        #  # features['p2'] = fake_image_f
        #  #################################ccr added
        #  print(comp_image.size(),'--------------CompNet output (before upsample)')
        #  print(up_image.size(),'--------------CompNet output (after upsample)')
        #  features_ds = features.copy()
        #  features_ds["p2"] = comp_image
        #  #print(features_ds["p2"].size(), '---------feature[p2] shape')
        #  print('max/min_p2(GT)(CompNet input): %8.4f/%8.4f, max/min_P2(CompNet output): %8.4f/%8.4f' %(torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_ds["p2"]), torch.min(features_ds["p2"])))
        #  ds_feat = quant_fix(features_ds.copy())
        #
        #  ##features_resid = features.copy()
        #  ##features_resid["p2"] = resid_pic
        #  ##resid_feat = quant_fix(features_resid.copy())
        #  #################################ccr added
        #
        #  image_feat = quant_fix(features.copy())
        #
        #  fname = utils.simple_filename(inputs[0]["file_name"])
        #  fname_feat = f"feature/{self.set_idx}_ori/{fname}.png"
        #  fname_p345 = f"feature/{self.set_idx}_p345/{fname}.png"
        #  fname_ds = f"feature/{self.set_idx}_ds/{fname}.png"
        #  fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"
        #
        #  with open(f"info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
        #      torch.save(inputs, inputs_f)
        #
        #  # utils.save_feature_map(fname_feat, image_feat)
        #  ####################################ccr added 3 parts
        #  utils.save_feature_map_onlyp2(fname_ds, ds_feat)
        #  # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        #  utils.save_feature_map_p345(fname_feat, image_feat)
        #  ####################################ccr added 3 parts
        return fname_feat

        # d = features['p2']  # [1, 256, 200, 304]
        # d_p3 = features['p3']  # [1, 256, 200, 304]
        # d_originalsize = d_p3
        # print(d.size(), '-------------------P2 original size')
        # #normlize p4 and p4
        # if torch.min(d) >= torch.min(d_p3): #2个数中取小的
        #     guiyihua_min = torch.min(d_p3)
        # else:
        #     guiyihua_min = torch.min(d)
        # if torch.max(d) >= torch.max(d_p3): #2个数中取大的
        #     guiyihua_max = torch.max(d)
        # else:
        #     guiyihua_max = torch.max(d_p3)
        # guiyihua_scale = guiyihua_max - guiyihua_min
        # ###pad
        # d, h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right = Pfeature_zeropad_youxiajiao256(d, 32)
        # d_p3, h_new_p3_left, h_new_p3_right, w_new_p3_left, w_new_p3_right = Pfeature_zeropad_youxiajiao128(d_p3, 16)
        # d = (d - guiyihua_min) / guiyihua_scale
        # d_p3 = (d_p3 - guiyihua_min) / guiyihua_scale
        # d_originalsize = (d_originalsize - guiyihua_min) / guiyihua_scale
        # print(d.size(), '-------------------Cheng input (P2) size')
        # # # normlize p2 and p4
        # # if torch.min(d) >= torch.min(d_p4):  # 2个数中取小的
        # #     guiyihua_min = torch.min(d_p4)
        # # else:
        # #     guiyihua_min = torch.min(d)
        # # if torch.max(d) >= torch.max(d_p4):  # 2个数中取大的
        # #     guiyihua_max = torch.max(d)
        # # else:
        # #     guiyihua_max = torch.max(d_p4)
        # # guiyihua_scale = guiyihua_max - guiyihua_min
        # # d = (d - guiyihua_min) / guiyihua_scale
        # # d_p4 = (d_p4 - guiyihua_min) / guiyihua_scale
        # # print(d.size(), '-------------------P2 original size')
        # # temp_ori_size_p2 = d.shape  # P2原始尺寸
        # # temp_ori_size_p4 = d_p4.shape  # P4原始尺寸
        # # target_size_p2 = [d.size()[0], d.size()[1], padding_size(d.size()[2], 16), padding_size(d.size()[3], 16)]  # P2补黑边后(16的倍数) [1, 256, 208, 304]
        # # d_big = torch.zeros(target_size_p2).cuda()
        # # d_big[:, 0:temp_ori_size_p2[1], 0:temp_ori_size_p2[2], 0:temp_ori_size_p2[3]] = d
        # # print(d_big.size(), '-------------------Cheng input (P2) size')
        # # target_size_p4 = [d_p4.size()[0], d_p4.size()[1], int(target_size_p2[2] / 4.0), int(target_size_p2[3] / 4.0)]  # P2的1/4
        # # d_big_p4 = torch.zeros(target_size_p4).cuda()
        # # d_big_p4[:, 0:temp_ori_size_p4[1], 0:temp_ori_size_p4[2], 0:temp_ori_size_p4[3]] = d_p4
        # # d_output = torch.zeros(temp_ori_size_p4)  # 用于从网络输出的tensor取出左上角
        # net_belle_output = self.model.net_belle(d)
        # print(net_belle_output["x_hat"].size(), '-------------------Cheng output (P3) size')
        # d_output = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output["x_hat"], h_new_p3_left, h_new_p3_right, w_new_p3_left, w_new_p3_right)
        # print(d_output.size(), '-------------------output size')
        # print('max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(d), torch.min(d), torch.max(d_output), torch.min(d_output)))
        # features_cheng = features.copy()
        # features_p345 = features.copy()
        # # features_cheng["p4"] = d_output * guiyihua_scale + guiyihua_min
        # features_p345["p3"] = d_output * guiyihua_scale + guiyihua_min
        # # print('After denormlize: max/min_p2(GT)(Cheng input): %8.4f/%8.4f, max/min_p4(GT): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f'
        # #       % (torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features["p4"]), torch.min(features["p4"]), torch.max(features_p345["p4"]), torch.min(features_p345["p4"])))
        # print('After denormlize: max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_p345["p3"]), torch.min(features_p345["p3"])))
        # cheng_feat = quant_fix(features_cheng.copy())
        #
        # heigh_temp = self.height_temp
        # width_temp = self.width_temp
        # numpixel_temp = self.numpixel_temp
        # out_criterion = self.criterion(net_belle_output, d_p3, heigh_temp, width_temp) #net_belle_output和d为pad后的
        # print('image hxw: %dx%d, num_pixel: %d' % (heigh_temp, width_temp, numpixel_temp))
        # # define_mse = nn.MSELoss()
        # # net_belle_output["x_hat"] = d_output  # [1, 256, 208, 304]->[1, 256, 200, 304]
        # # out_criterion["mse_loss"] = define_mse(net_belle_output["x_hat"], d_p4)
        # # psnr_temp = mse2psnr(out_criterion['mse_loss'])
        # # print('bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion["bpp_loss"].item(), out_criterion["mse_loss"].item(), psnr_temp))
        # define_mse = nn.MSELoss()
        # out_criterion["mse_loss"] = define_mse(d_output, d_originalsize)
        # psnr_temp = mse2psnr(out_criterion["mse_loss"])
        # print('bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion["bpp_loss"].item(), out_criterion["mse_loss"].item(), psnr_temp))
        # self.bpp_test5000[fname_temp] = [out_criterion["bpp_loss"].item()]
        # tf = open(self.path_bppsave, "w")
        # json.dump(self.bpp_test5000, tf)
        # tf.close()
        #
        # ##features_resid = features.copy()
        # ##features_resid["p2"] = resid_pic
        # ##resid_feat = quant_fix(features_resid.copy())
        # #################################ccr added
        #
        # image_feat = quant_fix(features_p345.copy())
        #
        # fname = utils.simple_filename(inputs[0]["file_name"])
        # # fname_feat = f"../../liutie_save/feature/{self.set_idx}_ori/{fname}.png"  # 用于存P345
        # fname_feat = f"../../zzf_save/feature/{self.set_idx}_ori/{fname}.png"  # 用于存P345
        # # fname_ds = f"../../liutie_save/feature/{self.set_idx}_ds/{fname}.png"  # 用于存P2
        # fname_ds = f"../../zzf_save/feature/{self.set_idx}_ds/{fname}.png"  # 用于存P2
        #
        # # with open(f"../../liutie_save/info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
        # with open(f"../../zzf_save/info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
        #     torch.save(inputs, inputs_f)
        #
        # # utils.save_feature_map(fname_feat, image_feat)
        # ####################################ccr added 3 parts
        # utils.save_feature_map_onlyp2(fname_ds, cheng_feat)  # 用于存P2
        # # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        # utils.save_feature_map_p345(fname_feat, image_feat)  # 用于存P345
        # ####################################liutie added 3 parts
        #
        # #  #################################ccr added
        # #  compG_input = features['p2']
        # #  print(compG_input.size(),'---------------CompNet_input')
        # #  comp_image = self.model.compG.forward(compG_input)
        # #  ####### replace CompNet(last sentence), to compare onlyCompNet(Down 2) and P2down2
        # #  #comp_image = F.interpolate(compG_input, scale_factor=0.5, mode="bilinear", align_corners=False)  # [1, 256, h/4, w/4]->[1, 256, h/8, w/8]
        # #  ##print( comp_image.size(),'-------------------- comp_image')
        # #  upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        # #  up_image = upsample(comp_image)
        # # ## print(up_image.size(),'--------------------------up_image')
        # # # input_fconcat = up_image
        # # # res = self.model.netG.forward(input_fconcat)
        # # ## print(res.size(),'------------------------res')
        # #  ##fake_image_f = res + up_image
        # #  ##resid_pic = compG_input - fake_image_f
        # #  # features['p2'] = fake_image_f
        # #  #################################ccr added
        # #  print(comp_image.size(),'--------------CompNet output (before upsample)')
        # #  print(up_image.size(),'--------------CompNet output (after upsample)')
        # #  features_ds = features.copy()
        # #  features_ds["p2"] = comp_image
        # #  #print(features_ds["p2"].size(), '---------feature[p2] shape')
        # #  print('max/min_p2(GT)(CompNet input): %8.4f/%8.4f, max/min_P2(CompNet output): %8.4f/%8.4f' %(torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_ds["p2"]), torch.min(features_ds["p2"])))
        # #  ds_feat = quant_fix(features_ds.copy())
        # #
        # #  ##features_resid = features.copy()
        # #  ##features_resid["p2"] = resid_pic
        # #  ##resid_feat = quant_fix(features_resid.copy())
        # #  #################################ccr added
        # #
        # #  image_feat = quant_fix(features.copy())
        # #
        # #  fname = utils.simple_filename(inputs[0]["file_name"])
        # #  fname_feat = f"feature/{self.set_idx}_ori/{fname}.png"
        # #  fname_p345 = f"feature/{self.set_idx}_p345/{fname}.png"
        # #  fname_ds = f"feature/{self.set_idx}_ds/{fname}.png"
        # #  fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"
        # #
        # #  with open(f"info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
        # #      torch.save(inputs, inputs_f)
        # #
        # #  # utils.save_feature_map(fname_feat, image_feat)
        # #  ####################################ccr added 3 parts
        # #  utils.save_feature_map_onlyp2(fname_ds, ds_feat)
        # #  # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        # #  utils.save_feature_map_p345(fname_feat, image_feat)
        # #  ####################################ccr added 3 parts
        # return fname_feat

        # d = features['p2']  # [1, 256, 200, 304]
        # d_p3 = features['p3']
        # d_p4 = features['p4']
        # d_p5 = features['p5']
        # d_originalsize_p2 = d
        # d_originalsize_p3 = d_p3
        # d_originalsize_p4 = d_p4
        # d_originalsize_p5 = d_p5
        # print(d.size(), '-------------------P2 original size')
        # #normlize p4 and p4
        #
        # if torch.min(d) >= torch.min(d_p3): #2个数中取小的
        #     guiyihua_min_1 = torch.min(d_p3)
        # else:
        #     guiyihua_min_1 = torch.min(d)
        # if torch.max(d) >= torch.max(d_p3): #2个数中取大的
        #     guiyihua_max_1 = torch.max(d)
        # else:
        #     guiyihua_max_1 = torch.max(d_p3)
        #
        # if torch.min(d_p4) >= torch.min(d_p5): #2个数中取小的
        #     guiyihua_min_2 = torch.min(d_p5)
        # else:
        #     guiyihua_min_2 = torch.min(d_p4)
        # if torch.max(d_p4) >= torch.max(d_p5): #2个数中取大的
        #     guiyihua_max_2 = torch.max(d_p4)
        # else:
        #     guiyihua_max_2 = torch.max(d_p5)
        #
        # guiyihua_max = max(guiyihua_max_1, guiyihua_max_2)
        # guiyihua_min = min(guiyihua_min_1, guiyihua_min_2)
        #
        # guiyihua_scale = guiyihua_max - guiyihua_min
        # ###pad
        # d, h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right = Pfeature_zeropad_youxiajiao256(d, 32)
        # d_p3, h_new_p3_left, h_new_p3_right, w_new_p3_left, w_new_p3_right = Pfeature_zeropad_youxiajiao128(d_p3, 16)
        # d_p4, h_new_p4_left, h_new_p4_right, w_new_p4_left, w_new_p4_right = Pfeature_zeropad_youxiajiao64(d_p4, 8)
        # d_p5, h_new_p5_left, h_new_p5_right, w_new_p5_left, w_new_p5_right = Pfeature_zeropad_youxiajiao32(d_p5, 4)
        # d = (d - guiyihua_min) / guiyihua_scale
        # d_p3 = (d_p3 - guiyihua_min) / guiyihua_scale
        # d_p4 = (d_p4 - guiyihua_min) / guiyihua_scale
        # d_p5 = (d_p5 - guiyihua_min) / guiyihua_scale
        # d_originalsize_p2 = (d_originalsize_p2 - guiyihua_min) / guiyihua_scale
        # d_originalsize_p3 = (d_originalsize_p3 - guiyihua_min) / guiyihua_scale
        # d_originalsize_p4 = (d_originalsize_p4 - guiyihua_min) / guiyihua_scale
        # d_originalsize_p5 = (d_originalsize_p5 - guiyihua_min) / guiyihua_scale
        # print(d.size(), '-------------------Cheng input (P2) size')
        #
        # net_belle_output = self.model.net_belle(d)
        # print(net_belle_output["x_hat"].size(), '-------------------cheng output (P3) size')
        # print(net_belle_output["x_hat_p2"].size(), '-------------------cheng output (P2) size')
        # print(net_belle_output["x_hat_p4"].size(), '-------------------cheng output (P4) size')
        # print(net_belle_output["x_hat_p5"].size(), '-------------------cheng output (P5) size')
        #
        # d_output_p3 = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output["x_hat"], h_new_p3_left, h_new_p3_right, w_new_p3_left, w_new_p3_right)
        # d_output_p2 = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output["x_hat_p2"], h_new_p2_left, h_new_p2_right, w_new_p2_left, w_new_p2_right)
        # d_output_p4 = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output["x_hat_p4"], h_new_p4_left, h_new_p4_right, w_new_p4_left, w_new_p4_right)
        # d_output_p5 = Pfeature_zeropad_youxiajiao128_reverse(net_belle_output["x_hat_p5"], h_new_p5_left, h_new_p5_right, w_new_p5_left, w_new_p5_right)
        #
        # print(d_output_p3.size(), '-------------------output_p3 size')
        # print(d_output_p2.size(), '-------------------output_p2 size')
        # print(d_output_p4.size(), '-------------------output_p4 size')
        # print(d_output_p5.size(), '-------------------output_p5 size')
        # print('max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(d), torch.min(d), torch.max(d_output_p3), torch.min(d_output_p3)))
        # print('max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P2(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(d), torch.min(d), torch.max(d_output_p2), torch.min(d_output_p2)))
        # print('max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(d), torch.min(d), torch.max(d_output_p4), torch.min(d_output_p4)))
        # print('max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P5(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(d), torch.min(d), torch.max(d_output_p5), torch.min(d_output_p5)))
        # features_cheng = features.copy()
        # features_p345 = features.copy()
        # # features_cheng["p4"] = d_output * guiyihua_scale + guiyihua_min
        # features_p345["p3"] = d_output_p3 * guiyihua_scale + guiyihua_min
        # features_p345["p2"] = d_output_p2 * guiyihua_scale + guiyihua_min
        # features_p345["p4"] = d_output_p4 * guiyihua_scale + guiyihua_min
        # features_p345["p5"] = d_output_p5 * guiyihua_scale + guiyihua_min
        # # print('After denormlize: max/min_p2(GT)(Cheng input): %8.4f/%8.4f, max/min_p4(GT): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f'
        # #       % (torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features["p4"]), torch.min(features["p4"]), torch.max(features_p345["p4"]), torch.min(features_p345["p4"])))
        # print('After denormlize: max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P3(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_p345["p3"]), torch.min(features_p345["p3"])))
        # print('After denormlize: max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_p345["p4"]), torch.min(features_p345["p4"])))
        # print('After denormlize: max/min_P2(GT)(Cheng input): %8.4f/%8.4f, max/min_P5(Cheng output): %8.4f/%8.4f'
        #       % (torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_p345["p5"]), torch.min(features_p345["p5"])))
        #
        # cheng_feat = quant_fix(features_cheng.copy())
        #
        # heigh_temp = self.height_temp
        # width_temp = self.width_temp
        # numpixel_temp = self.numpixel_temp
        # out_criterion_p3 = self.criterion_p3(net_belle_output, d_p3, heigh_temp, width_temp) #net_belle_output和d为pad后的
        # out_criterion_p2 = self.criterion_p2(net_belle_output, d, heigh_temp, width_temp)  # net_belle_output和d为pad后的
        # out_criterion_p4 = self.criterion_p4(net_belle_output, d_p4, heigh_temp, width_temp)  # net_belle_output和d为pad后的
        # out_criterion_p5 = self.criterion_p5(net_belle_output, d_p5, heigh_temp, width_temp)  # net_belle_output和d为pad后的
        # print('image hxw: %dx%d, num_pixel: %d' % (heigh_temp, width_temp, numpixel_temp))
        #
        # define_mse = nn.MSELoss()
        # out_criterion_p3["mse_loss"] = define_mse(d_output_p3, d_originalsize_p3)
        # psnr_temp_p3 = mse2psnr(out_criterion_p3["mse_loss"])
        # print('P3 bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion_p3["bpp_loss"].item(), out_criterion_p3["mse_loss"].item(), psnr_temp_p3))
        #
        # define_mse = nn.MSELoss()
        # out_criterion_p2["mse_loss"] = define_mse(d_output_p2, d_originalsize_p2)
        # psnr_temp_p2 = mse2psnr(out_criterion_p2["mse_loss"])
        # print('P2 bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (out_criterion_p2["bpp_loss"].item(), out_criterion_p2["mse_loss"].item(), psnr_temp_p2))
        #
        # define_mse = nn.MSELoss()
        # out_criterion_p4["mse_loss"] = define_mse(d_output_p4, d_originalsize_p4)
        # psnr_temp_p4 = mse2psnr(out_criterion_p4["mse_loss"])
        # print('P4 bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (
        # out_criterion_p4["bpp_loss"].item(), out_criterion_p4["mse_loss"].item(), psnr_temp_p4))
        #
        # define_mse = nn.MSELoss()
        # out_criterion_p5["mse_loss"] = define_mse(d_output_p5, d_originalsize_p5)
        # psnr_temp_p5 = mse2psnr(out_criterion_p5["mse_loss"])
        # print('P5 bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' % (
        # out_criterion_p5["bpp_loss"].item(), out_criterion_p5["mse_loss"].item(), psnr_temp_p5))
        #
        # self.bpp_test5000[fname_temp] = [out_criterion_p3["bpp_loss"].item()] # 将p3的bpp_loss作为输出
        # tf = open(self.path_bppsave, "w")
        # json.dump(self.bpp_test5000, tf)
        # tf.close()
        #
        #
        #
        # image_feat = quant_fix(features_p345.copy())
        #
        # fname = utils.simple_filename(inputs[0]["file_name"])
        # fname_feat = f"../../zzf_save/feature/{self.set_idx}_ori/{fname}.png"  # 用于存P345
        # # fname_p345 = f"feature/{self.set_idx}_p345/{fname}.png"
        # fname_ds = f"../../zzf_save/feature/{self.set_idx}_ds/{fname}.png"  # 用于存P2
        # # fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"
        #
        # with open(f"../../zzf_save/info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
        #     torch.save(inputs, inputs_f)
        #
        # # utils.save_feature_map(fname_feat, image_feat)
        # ####################################ccr added 3 parts
        # utils.save_feature_map_onlyp2(fname_ds, cheng_feat)  # 用于存P2
        # # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        # utils.save_feature_map_p345(fname_feat, image_feat)  # 用于存P345
        # ####################################liutie added 3 parts
        #
        #
        # return fname_feat

    def evaluation(self, inputs):
        with open(f"../../zzf_save/output/{self.set_idx}_coco.txt", 'w') as of:
            of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')

            coco_classes_fname = 'oi_eval/coco_classes.txt'

            with open(coco_classes_fname, 'r') as f:
                coco_classes = f.read().splitlines()

            # for fname in tqdm(inputs[:2]):
            for fname in tqdm(inputs):

                outputs = self._evaluation(fname)
                outputs = outputs[0]

                imageId = os.path.basename(fname)
                classes = outputs['instances'].pred_classes.to('cpu').numpy()
                scores = outputs['instances'].scores.to('cpu').numpy()
                bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
                H, W = outputs['instances'].image_size

                bboxes = bboxes / [W, H, W, H]
                bboxes = bboxes[:, [0, 2, 1, 3]]

                masks = outputs['instances'].pred_masks.to('cpu').numpy()

                for ii in range(len(classes)):
                    coco_cnt_id = classes[ii]
                    class_name = coco_classes[coco_cnt_id]

                    rslt = [imageId[:-4], class_name, scores[ii]] + \
                           bboxes[ii].tolist()

                    assert (masks[ii].shape[1] == W) and (masks[ii].shape[0] == H)
                    rslt += [masks[ii].shape[1], masks[ii].shape[0],
                             oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

                    o_line = ','.join(map(str, rslt))
                    print('1111')
                    of.write(o_line + '\n')

        conversion(self.set_idx)
        cmd = f"python oid_challenge_evaluation.py \
        --input_annotations_boxes   ./dataset/annotations_5k/segmentation_validation_bbox_5k.csv \
        --input_annotations_labels  ./dataset/annotations_5k/segmentation_validation_labels_5k.csv \
        --input_class_labelmap      ./dataset/annotations_5k/coco_label_map.pbtxt \
        --input_annotations_segm    ./dataset/annotations_5k/segmentation_validation_masks_5k.csv \
        --input_predictions         ../../zzf_save/output/{self.set_idx}_oi.txt \
        --output_metrics            ../../zzf_save/output/{self.set_idx}_AP.txt"
        # --input_predictions         inference/{self.set_idx}_oi.txt \
        # --output_metrics            inference/{self.set_idx}_AP.txt"
        print(">>>> cmd: ", cmd)
        subprocess.call([cmd], shell=True)

        self.summary()

        return

    def evaluation_offline(self):
        cmd = f"python oid_challenge_evaluation.py \
        --input_annotations_boxes   oi_eval/detection_validation_5k_bbox.csv \
        --input_annotations_labels  oi_eval/detection_validation_labels_5k.csv \
        --input_class_labelmap      oi_eval/coco_label_map.pbtxt \
        --input_predictions         /media/data/minglang/data/detection_result/q7_result.oid.txt \
        --output_metrics            inference_ml/origin_AP.txt"
        print(">>>> cmd: ", cmd)
        subprocess.call([cmd], shell=True)

    def _evaluation(self, fname):

        fname_simple = utils_copy.simple_filename(fname)
        print(fname)
        # fname_ds_rec = fname.replace('rec', 'ds_rec')
        # # fname_resid_rec = fname.replace('rec', 'resid_rec')
        fname_ds_rec = fname.replace('ori', 'ds')  # QP36原始
        # fname_ds_rec = fname.replace('36_ori', '42_ds') #QP36

        with open(f"../../zzf_save/info/{self.set_idx}/{fname_simple}_inputs.bin", "rb") as inputs_f:
            inputs = torch.load(inputs_f)

        images = self.model.preprocess_image(inputs)
        features = self.feat2feat_p345(fname)  # P3P4P5P6 float32
        features_ds = self.feat2feat_onlyp2(fname_ds_rec)  # P2 float32
        # features_resid = self.feat2feat_onlyp2(fname_resid_rec)
        # resid = features_resid['p2']
        features['p2'] = features_ds['p2']  # 给features(只有P3456)加上P2
        features['p2'] = features['p2'].type(torch.float64)
        features['p3'] = features['p3'].type(torch.float64)
        features['p4'] = features['p4'].type(torch.float64)
        features['p5'] = features['p5'].type(torch.float64)
        features['p6'] = features['p6'].type(torch.float64)


        outputs = self.forward_front(inputs, images, features)  # images是float64
        self.evaluator.process(inputs, outputs)
        return outputs

    def summary(self):
        with open("../../zzf_save/results.csv", "a") as result_f:
            # with open(f"inference/{self.set_idx}_AP.txt", "rt") as ap_f:
            with open(f"../../zzf_save/output/{self.set_idx}_AP.txt", "rt") as ap_f:
                ap = ap_f.readline()
                ap = ap.split(",")[1][:-1]

            size_basis = utils_copy.get_size(f'../../zzf_save/feature/{self.set_idx}_bit/')

            # ml add
            size_coeffs, size_mean, self.qp, self.DeepCABAC_qstep = 0, 0, 0, 0
            # bpp = (size_basis + size_coeffs, + size_mean)/self.pixel_num
            # ml
            bpp = 0

            print(">>>> result: ", f"{self.set_idx},{self.qp},{self.DeepCABAC_qstep},{bpp},{ap}\n")

            result_f.write(f"{self.set_idx},{self.qp},{self.DeepCABAC_qstep},{bpp},{ap}\n")

    def feat2feat(self, fname):
        pyramid = {}

        png = cv2.imread(fname, -1).astype(np.float32)
        vectors_height = png.shape[0]
        v2_h = int(vectors_height / 85 * 64)
        v3_h = int(vectors_height / 85 * 80)
        v4_h = int(vectors_height / 85 * 84)

        v2_blk = png[:v2_h, :]
        v3_blk = png[v2_h:v3_h, :]
        v4_blk = png[v3_h:v4_h, :]
        v5_blk = png[v4_h:vectors_height, :]

        pyramid["p2"] = self.feature_slice(v2_blk, [v2_blk.shape[0] // 16, v2_blk.shape[1] // 16])
        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8, v3_blk.shape[1] // 32])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4, v4_blk.shape[1] // 64])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2, v5_blk.shape[1] // 128])

        pyramid["p2"] = dequant_fix(pyramid["p2"])
        pyramid["p3"] = dequant_fix(pyramid["p3"])
        pyramid["p4"] = dequant_fix(pyramid["p4"])
        pyramid["p5"] = dequant_fix(pyramid["p5"])

        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
        pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
        pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)

        pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

        # 加了下面这几句弄到cuda
        pyramid["p2"] = pyramid["p2"].cuda()
        pyramid["p3"] = pyramid["p3"].cuda()
        pyramid["p4"] = pyramid["p4"].cuda()
        pyramid["p5"] = pyramid["p5"].cuda()
        pyramid["p6"] = pyramid["p6"].cuda()

        return pyramid

    def feat2feat_onlyp2(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        pyramid["p2"] = self.feature_slice(png, [png.shape[0] // 16, png.shape[1] // 16])
        pyramid["p2"] = dequant_fix(pyramid["p2"])
        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        # 加了下面这几句弄到cuda
        pyramid["p2"] = pyramid["p2"].cuda()
        return pyramid

    def feat2feat_p345(self, fname):
        pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)
        vectors_height = png.shape[0]
        v3_h = int(vectors_height / 21 * 16)
        v4_h = int(vectors_height / 21 * 20)

        v3_blk = png[:v3_h, :]
        v4_blk = png[v3_h:v4_h, :]
        v5_blk = png[v4_h:vectors_height, :]

        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8, v3_blk.shape[1] // 32])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4, v4_blk.shape[1] // 64])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2, v5_blk.shape[1] // 128])

        pyramid["p3"] = dequant_fix(pyramid["p3"])
        pyramid["p4"] = dequant_fix(pyramid["p4"])
        pyramid["p5"] = dequant_fix(pyramid["p5"])

        pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
        pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
        pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)
        pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

        # 加了下面这几句弄到cuda
        pyramid["p3"] = pyramid["p3"].cuda()
        pyramid["p4"] = pyramid["p4"].cuda()
        pyramid["p5"] = pyramid["p5"].cuda()
        pyramid["p6"] = pyramid["p6"].cuda()
        return pyramid

    def feature_slice(self, image, shape):
        height = image.shape[0]
        width = image.shape[1]

        blk_height = shape[0]
        blk_width = shape[1]
        blk = []

        for y in range(height // blk_height):
            for x in range(width // blk_width):
                y_lower = y * blk_height
                y_upper = (y + 1) * blk_height
                x_lower = x * blk_width
                x_upper = (x + 1) * blk_width
                blk.append(image[y_lower:y_upper, x_lower:x_upper])
        feature = torch.from_numpy(np.array(blk))
        return feature

    def clear(self):
        DatasetCatalog._REGISTERED.clear()


class DetectEval(Eval):
    def prepare_part(self, myarg, data_name="pick"):
        print("Loading", data_name, "...")
        utils_copy.pick_coco_exp(data_name, myarg)
        self.data_loader = build_detection_test_loader(self.cfg, data_name)
        self.evaluator = COCOEvaluator(data_name, self.cfg, False)
        self.evaluator.reset()
        print(data_name, "Loaded")
