# Copyright (c) Facebook, Inc. and its affiliates.
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
import random
import time

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

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

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf * (2 ** n_downsampling), output_nc, kernel_size=7, padding=0), nn.Tanh()]
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
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # n_downsampling=1

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
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
        compressaiargs_lambda = 1.0 #0.05 #指令里
        self.belle_clip_max_norm = 1.0
        # # self.net_belle = image_models[compressaiargs_model](quality=compressaiargs_quality)
        # self.net_belle = ScaleHyperpriorMulti(M=192, N=128)  # M=192, N=128,  320,192
        # # self.net_belle = ScaleHyperpriorMulti(M=512, N=320)
        # # self.net_belle = Cheng2020AttentionMulti(N=192)
        self.net_belle = Cheng2020Anchor(N=192)
        self.net_belle = self.net_belle.to(device)
        # print('#####################################PRINT NET_BELLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(self.net_belle)
        self.optimizer_belle, self.belle_aux_optimizer = configure_optimizers(self.net_belle, compressaiargs_learning_rate, compressaiargs_aux_learning_rate)
        self.belle_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_belle, milestones=[2000, 4250], gamma=0.5)
        self.belle_criterion = RateDistortionLoss(compressaiargs_lambda)
        self.i_step_count = 0
        # compressai_logdir = '/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM/tensorboard_belle/EXP_cheng2020anchor_256chinput_P4inP4outMSE_replicateyouxiajiao_lambda1_N192_ft7imgtrain4999_small5Wtrain_eachdnorm_08221010/'
        # compressai_logdir = '/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM/tensorboard_belle/EXP_cheng2020anchor_256chinput_P4inP4outMSE_zeroyouxiajiao128_lambda1_N192_7imgtrain_eachdnorm_08231650/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P4inP4outMSE_zeroyouxiajiao128_lambda1_N192_7imgtrain_eachdnorm_08231650/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P4inP4outMSE_zeroyouxiajiao128_lambda1_N192_7imgtrainft4999_small5Wtrain_eachdnorm_08231750/'
        # compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P4inP4outMSE_zeroyouxiajiao128_lambda1_N192_7imgtrain_eachdnorm_08231650_1/'
        compressai_logdir = '../../liutie_save/tensorboard_belle/EXP_cheng2020anchor_256chinput_P4inP4outMSE_zeroyouxiajiao128_lambda1_N192_7imgtrainft9999_small5Wtrain_eachdnorm_08231750_1/'
        mkdirs(compressai_logdir)
        self.belle_writer = SummaryWriter(log_dir=compressai_logdir)
        self.belle_savetensorboardfreq = 100

        # self.guiyihua_scale = 43.6045
        # self.guiyihua_min = -23.1728
        # path_belle_pretrainedmodel = '/media/data/liutie/VCM/rcnn/belle_pretrainedmodel/bmshj2018-hyperprior-4-de1b779c.pth.tar'
        # checkpoint = torch.load(path_belle_pretrainedmodel, map_location=lambda storage, loc: storage)
        # try:
        #     self.net_belle.load_state_dict(checkpoint['state_dict'])
        # except:
        #     try:
        #         for key in list(checkpoint):
        #             if key.split(".")[0] == "entropy_bottleneck":
        #                 if key.split(".")[1] == "_biases":
        #                     checkpoint[key.split(".")[0] + '._bias' + key.split(".")[2]] = checkpoint[key]
        #                     del (checkpoint[key])
        #                 if key.split(".")[1] == "_factors":
        #                     checkpoint[key.split(".")[0] + '._factor' + key.split(".")[2]] = checkpoint[key]
        #                     del (checkpoint[key])
        #                 if key.split(".")[1] == "_matrices":
        #                     checkpoint[key.split(".")[0] + '._matrix' + key.split(".")[2]] = checkpoint[key]
        #                     del (checkpoint[key])
        #         self.net_belle.load_state_dict(checkpoint)
        #         print("load codec ori")
        #     except:
        #         print("codec load none")

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
        ## ywz add for CAI forward - 使用locals()进行反射
        cai_input_tensor = features["p2"]  # float32
        cai_input_tensor_p4 = features["p4"]  # float32
        #normlize p2 and p4
        guiyihua_max = torch.max(cai_input_tensor_p4)
        guiyihua_min = torch.min(cai_input_tensor_p4)
        guiyihua_scale = guiyihua_max - guiyihua_min
        # cai_input_tensor = (cai_input_tensor - guiyihua_min) / guiyihua_scale
        # cai_input_tensor_p4 = (cai_input_tensor_p4 - guiyihua_min) / guiyihua_scale
        ###pad
        d, h_new_left, h_new_right, w_new_left, w_new_right = Pfeature_zeropad_youxiajiao128(cai_input_tensor, 64)
        d_p4, _, _, _, _ = Pfeature_zeropad_youxiajiao128(cai_input_tensor_p4, 16)
        d = (d - guiyihua_min) / guiyihua_scale
        d_p4 = (d_p4 - guiyihua_min) / guiyihua_scale
        # ###将P2crop成64的倍数
        # h_p2 = cai_input_tensor.size()[2]
        # w_p2 = cai_input_tensor.size()[3]
        # # h_p2_new = h_p2 // 64 * 64
        # # w_p2_new = w_p2 // 64 * 64
        # h_p2_new = h_p2 // 16 * 16
        # w_p2_new = w_p2 // 16 * 16
        # target_size = [cai_input_tensor.size()[0], cai_input_tensor.size()[1], h_p2_new, w_p2_new]  # [b, 256, 128, 192]
        # cai_input_tensor_new = torch.zeros(target_size)
        # cai_input_tensor_new = cai_input_tensor[:, 0:cai_input_tensor_new.size()[1], 0:cai_input_tensor_new.size()[2], 0:cai_input_tensor_new.size()[3]]
        # # print(cai_input_tensor_new.size(), '-------------------CAI P2 input new size')
        # #此2行替换成下面4行
        # # h_p4_new = int(h_p2_new / 4)
        # # w_p4_new = int(w_p2_new / 4)
        # h_p4 = cai_input_tensor_p4.size()[2]
        # w_p4 = cai_input_tensor_p4.size()[3]
        # h_p4_new = h_p4 // 16 * 16
        # w_p4_new = w_p4 // 16 * 16
        # target_size_p4 = [cai_input_tensor_p4.size()[0], cai_input_tensor_p4.size()[1], h_p4_new, w_p4_new]  # [b, 256, 128 / 4, 192 / 4]
        # cai_input_tensor_p4_new = torch.zeros(target_size_p4)
        # cai_input_tensor_p4_new = cai_input_tensor_p4[:, 0:cai_input_tensor_p4_new.size()[1], 0:cai_input_tensor_p4_new.size()[2], 0:cai_input_tensor_p4_new.size()[3]]
        # # print(cai_input_tensor_p4_new.size(), '-------------------CAI P4 input new size')
        ### train CAI framework
        self.net_belle.train()
        # self.net_belle.eval()
        d = d.to(device)
        d_p4 = d_p4.to(device)
        # print(d_p4.size(), '-------------------P4_GT size')
        print(d_p4.size(), '-------------------cheng input (P4) size')
        # time1_end = time.time()
        # time1 = time1_end - time1_start
        # time2_start = time.time()
        self.optimizer_belle.zero_grad()  # optimizer.zero_grad()
        self.belle_aux_optimizer.zero_grad()
        net_belle_output = self.net_belle(d_p4)
        print(net_belle_output["x_hat"].size(), '-------------------cheng output (P4) size')
        out_criterion = self.belle_criterion(net_belle_output, d_p4)
        out_criterion["loss"].backward()
        if self.belle_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net_belle.parameters(), self.belle_clip_max_norm)
        self.optimizer_belle.step()

        aux_loss = self.net_belle.aux_loss()
        aux_loss.backward()
        self.belle_aux_optimizer.step()
        psnr_temp = 10 * math.log10(1 / out_criterion["mse_loss"].item())
        print(
            f'Loss: {out_criterion["loss"].item():.3f} |'
            f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
            f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
            f"\tAux loss: {aux_loss.item():.2f}"
            f'\tPSNR: {psnr_temp:.3f} |'
        )
        # print("i_step: %d, max/min_P2(cheng input): %8.4f/%8.4f, max/min_P4(cheng output): %8.4f/%8.4f, max/min_P4(GT): %8.4f/%8.4f"
        #       % (self.i_step_count, torch.max(d), torch.min(d), torch.max(net_belle_output["x_hat"]), torch.min(net_belle_output["x_hat"]), torch.max(d_p4), torch.min(d_p4)))
        print("i_step: %d, max/min_P4(cheng input): %8.4f/%8.4f, max/min_P4(cheng output): %8.4f/%8.4f"
              % (self.i_step_count, torch.max(d_p4), torch.min(d_p4), torch.max(net_belle_output["x_hat"]), torch.min(net_belle_output["x_hat"])))

        #net_belle_output["x_hat"]小于0置为0，大于1置为1

        if (self.i_step_count % self.belle_savetensorboardfreq == 0): # and (channel_idx == 0):
            i_select_channel = random.randint(0, 255)
            self.belle_writer.add_scalar("training/Loss", out_criterion["loss"], global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/MSE loss", out_criterion["mse_loss"], global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/Bpp loss", out_criterion["bpp_loss"], global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/Aux loss", aux_loss.item(), global_step=self.i_step_count)
            self.belle_writer.add_scalar("training/PSNR", psnr_temp, global_step=self.i_step_count)
            self.belle_writer.add_image('images.tensor', images.tensor[0, :, :, :], global_step=self.i_step_count, dataformats='CHW')  # dataformats='HWC')
            self.belle_writer.add_image('P2_GT', d[0, i_select_channel:(i_select_channel+1), :, :], global_step=self.i_step_count, dataformats='CHW')  # dataformats='HWC')
            self.belle_writer.add_image('P4_GT', d_p4[0, i_select_channel:(i_select_channel+1), :, :], global_step=self.i_step_count, dataformats='CHW')  # dataformats='HWC')
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
        print(features['p5'],'----------------------------------------------------------------------GeneralizedRCNN inference features')

        real_image = features["p2"]
        # no seg
        compG_input = real_image
        comp_image = self.compG.forward(compG_input)
        ### tensor-level bilinear
        upsample = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        #upsample = torch.nn.Upsample(scale_factor=self.opt.alpha, mode='bilinear')
        up_image = upsample(comp_image)
        # no seg
        input_flabel = None
        # ds (ds | comp_image)
        input_fconcat = up_image
        # add compact image, so that G tries to find the best residual
        res = self.netG.forward(input_fconcat)
        fake_image_f = res + up_image
        features['p2'] = fake_image_f


        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
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
