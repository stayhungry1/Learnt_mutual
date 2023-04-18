# python3 examples/train.py -m cheng2020-attn -d /media/data/yangwenzhe/Dataset/DIV2K/DIV2K_train_HR/ -d_test /media/data/yangwenzhe/Dataset/div_after_crop/ -q 4 --lambda 0.001 --batch-size 6 -lr 1e-5 --save --cuda --exp exp_cheng_En_01_only_q4 --checkpoint /home/jjp/CompressAI/pretrained_model/cheng2020-attn/cheng2020_attn-ms-ssim-4-8b2f647e.pth.tar
import os
import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict

import logging
# from utils import util
import os
import logging
from datetime import datetime

torch.set_default_tensor_type(torch.DoubleTensor)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
from compressai.datasets import ImageFolder, FeatureFolder
from compressai.zoo import image_models
from compressai.models.Inv_arch import InvRescaleNet
from compressai.models.Subnet_constructor import subnet
from compressai.models.Enhance import EnModule
from compressai.models.ARCNN import ARCNNModel
from torchvision.transforms import ToPILImage
import numpy as np
import PIL
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from pytorch_msssim import ms_ssim
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.cpu().clamp_(0, 1).squeeze())


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
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m


class RateDistortionLoss(nn.Module): #只注释掉了109行的bpp_loss, 08021808又加上了
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    # def forward(self, output, target, lq, x_l, x_enh): #0.001
    def forward(self, output, target):  # 0.001 #, lq, x_l, x_enh
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # # out["mse_loss"] = self.mse(lq, target)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"] #lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out


class RateDistortionLoss_p2(nn.Module): #只注释掉了109行的bpp_loss, 08021808又加上了
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    # def forward(self, output, target, lq, x_l, x_enh): #0.001
    def forward(self, output, target):  # 0.001 #, lq, x_l, x_enh
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # # out["mse_loss"] = self.mse(lq, target)
        out["mse_loss"] = self.mse(output["x_hat_p2"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"] #lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out


class RateDistortionLoss_copy(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    # def forward(self, output, target, lq, x_l, x_enh): #0.001
    def forward(self, output, target):  # 0.001 #, lq, x_l, x_enh
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # out["mse_loss"] = self.mse(lq, target)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"] #lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, learning_rate, aux_learning_rate):
# def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        # lr=args.learning_rate,
        lr=learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        # lr=args.aux_learning_rate,
        lr=aux_learning_rate,
    )
    return optimizer, aux_optimizer


def Net_optimizers(net, learning_rate):
    parameters = {
        n for n, p in net.named_parameters() if p.requires_grad
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=learning_rate,
    )

    return optimizer


def train_one_epoch(
        model, IRN_net, En_net, criterion, train_dataloader, optimizer, optimizer_IRN, optimizer_En, aux_optimizer,
        epoch, clip_max_norm, logger_train, tb_logger
):
    model.train()
    IRN_net.train()
    # En_net.train()

    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        # # debug ywz
        print(d.size()) #([8, 3, 256, 256]) -> ([8,256, 256, 256])
        # d = F.pad(d, [d.size()[0], d.size()[1], padding_size(d.size()[1], 64), padding_size(d.size()[2], 64)])
        # print(d.size()) #([8, 3, 256, 256]) -> ([8,256, 256, 256])
        raise ValueError("stop")
        # ywz done
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        optimizer_IRN.zero_grad()
        # optimizer_En.zero_grad()

        x_l = IRN_inference(IRN_net, d, device)
        out_net = model(x_l)
        # x_enh = En_net(out_net["x_hat"])
        x_enh = out_net["x_hat"] #不走质量增强
        x_hat = IRN_inference(IRN_net, x_enh, device, rev=True)

        out_criterion = criterion(out_net, d, x_hat, x_l, x_enh)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            torch.nn.utils.clip_grad_norm_(IRN_net.parameters(), clip_max_norm)
            # torch.nn.utils.clip_grad_norm_(En_net.parameters(), clip_max_norm)
        optimizer.step()
        optimizer_IRN.step()
        # optimizer_En.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            # print(
            #     f"Train epoch {epoch}: ["
            #     f"{i * len(d)}/{len(train_dataloader.dataset)}"
            #     f" ({100. * i / len(train_dataloader):.0f}%)]"
            #     f'\tLoss: {out_criterion["loss"].item():.3f} |'
            #     f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
            #     f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
            #     f"\tAux loss: {aux_loss.item():.2f}"
            # )
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i * len(d):5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.4f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.4f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )

    tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), epoch)
    tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), epoch)
    tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), epoch)


def test_epoch(epoch, test_dataloader, model, IRN_net, En_net, criterion, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    psnr_1 = AverageMeter()
    psnr_2 = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            x_l = IRN_inference(IRN_net, d, device)
            out_net = model(x_l)
            # x_enh = En_net(out_net["x_hat"])
            x_enh = out_net["x_hat"]
            x_hat = IRN_inference(IRN_net, x_enh, device, rev=True)
            out_criterion = criterion(out_net, d, x_hat, x_l, x_enh)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            x_ori = x_l
            x1 = out_net["x_hat"]
            x2 = x_enh
            x_ori = torch2img(x_ori)
            x1 = torch2img(x1)
            x2 = torch2img(x2)
            # save_dir = "/media/data/yangwenzhe/ywzCompressAI/experiments/exp_cheng_En_03_only_q4/images/"
            # x_ori.save(os.path.join(save_dir, 'gt.png'))
            # x1.save(os.path.join(save_dir, 'x1.png'))
            # x2.save(os.path.join(save_dir, 'x2.png'))

            p1, m1 = compute_metrics(x_ori, x1)
            psnr_1.update(p1)
            p2, m2 = compute_metrics(x_ori, x2)
            psnr_2.update(p2)

            rec = torch2img(x_hat)
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: dpsnr'), psnr_2.avg - psnr_1.avg, epoch + 1)
    # print(
    #     f"Test epoch {epoch}: Average losses:"
    #     f"\tLoss: {loss.avg:.3f} |"
    #     f"\tMSE loss: {mse_loss.avg:.3f} |"
    #     f"\tBpp loss: {bpp_loss.avg:.4f} |"
    #     f"\tAux loss: {aux_loss.avg:.2f}\n"
    # )
    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"MSE loss: {mse_loss.avg:.4f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f} | "
        f"PSNR_1: {psnr_1.avg:.6f} | "
        f"PSNR_2: {psnr_2.avg:.6f} | "
    )
    return loss.avg


# def temp_epoch(test_dataloader, model):
#     model.eval()
#     device = next(model.parameters()).device
#
#     with torch.no_grad():
#         for i, x in enumerate(test_dataloader):
#             x = x.to(device)
#             x_forward = model(x)
#             x_l = x_forward[:,:3,:,:]
#             x_h = x_forward[:, 3:, :, :]
#             z = torch.randn(x_h.shape).to(device)
#             x_backward = model(torch.cat((x_l, z), 1), rev=True)
#             torchvision.utils.save_image(x, "/home/jjp/CompressAI/results/temp/" + 'gt.png')
#             torchvision.utils.save_image(x_backward, "/home/jjp/CompressAI/results/temp/" + '%.5d.png' % i)


def IRN_inference(model, input, device, rev=False):
    if rev:
        b, c, h, w = input.shape
        z = torch.randn([b, 3 * c, h, w]).to(device)
        out = model(torch.cat((input, z), 1), rev)

    else:
        x_forward = model(input)
        out = x_forward[:, :3, :, :]

    return out


def save_checkpoint(state, is_best, suffix, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(filename.split('/')[-1], suffix + "_checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, dest_filename)


def load_ARCNN(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net['network']['net'].items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)


def load_IRN(path, network):
    load_path = path
    load_network(load_path, network, True)


def load_network(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    load_net = torch.load(load_path)

    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)


def load_En(checkpoint, net):
    state_dicts = torch.load(checkpoint)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


def parse_args(argv=None):
    # add yaml
    import yaml
    def _parse_args_and_yaml(given_parser=None):
        if given_parser == None:
            given_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
        given_configs, remaining = given_parser.parse_known_args()
        if given_configs.cai_file:
            with open(given_configs.cai_file, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                given_parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = given_parser.parse_args(remaining)

        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text

    # add yaml done
    parser = argparse.ArgumentParser(description="Example training script.")
    # VCM file
    parser.add_argument(
        '--config-file',
        default=
        'VCMbelle_06221/examples/train_in_this.yaml',
        type=str,
        metavar='FILE',
        help='YAML config file specifying default arguments')
    # self file
    parser.add_argument(
        '--cai-file',
        default=
        'VCMbelle_06221/examples/train_in_this.yaml',
        type=str,
        metavar='FILE',
        help='YAML config file specifying default arguments')


    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", default='', type=str, required=False, help="Training dataset"
    )
    parser.add_argument(
        "-d_test", "--test_dataset", type=str, required=False, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "-exp", "--experiment", type=str, required=False, help="Experiment name"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    # args = parser.parse_args(argv)
    # return args
    args, args_text = _parse_args_and_yaml(parser)
    return args

def main_cai(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.makedirs(os.path.join('experiments', args.experiment))

    # util.setup_logger
    setup_logger('train', os.path.join('experiments', args.experiment), 'train_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)
    setup_logger('val', os.path.join('experiments', args.experiment), 'val_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    tb_logger = SummaryWriter(log_dir='/media/data/yangwenzhe/ywzCompressAI/tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('experiments', args.experiment, 'checkpoints'))
    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    # )

    # test_transforms = transforms.Compose(
    #     [transforms.ToTensor()]
    # )

    train_dataset = FeatureFolder(args.dataset, split="")#, transform=train_transforms)
    test_dataset = FeatureFolder(args.test_dataset, split="")#, transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=args.quality)
    net = net.to(device)
    IRN_net = InvRescaleNet(3, 3, subnet("DBNet", 'xavier'), [8], 1)
    # IRN_net = InvRescaleNet(256, 256, subnet("DBNet", 'xavier'), [8], 1)
    IRN_net = IRN_net.to(device)
    En_net = ARCNNModel()
    # En_net = EnModule(64, 3, 16, 10, 5)
    # En_net = En_net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
        # net = CustomDataParallel(net)
        # IRN_net = CustomDataParallel(IRN_net)
        # En_net = CustomDataParallel(En_net)

    logger_train.info(args)
    logger_train.info(net)
    logger_train.info(IRN_net)
    logger_train.info(En_net)

    # # Load IRN
    # IRN_path = "/media/data/yangwenzhe/ywzCompressAI/experiments/exp_cheng_En_01_only_q4/checkpoints/IRN_net_checkpoint_best_loss.pth.tar"
    # checkpoint = torch.load(IRN_path, map_location=lambda storage, loc: storage)
    # IRN_net.load_state_dict(checkpoint['state_dict'])
    #
    # # Load EN
    # # EN_path = "/media/data/yangwenzhe/ywzCompressAI/experiments/exp_cheng_En_02_q4/checkpoints/En_net_checkpoint_best_loss.pth.tar"
    # # checkpoint = torch.load(EN_path, map_location=lambda storage, loc: storage)
    # # En_net.load_state_dict(checkpoint['state_dict'])
    # try:
    #     EN_path = "/media/data/yangwenzhe/ywzCompressAI/pretrained_model/ARCNN/ckp_first_best.pt"
    #     load_ARCNN(EN_path, En_net)
    # except:
    #     print("error load EN_NET")
    #
    # try:
    #     IRN_path = "/media/data/yangwenzhe/ywzCompressAI/pretrained_model/IRN/x2/IRN_x2.pth"
    #     load_IRN(IRN_path, IRN_net)
    #     # temp_epoch(test_dataloader, IRN_net)
    # except:
    #     print("error load IRN")

    optimizer, aux_optimizer = configure_optimizers(net, args)
    optimizer_IRN = Net_optimizers(IRN_net, args.learning_rate)
    optimizer_En = Net_optimizers(En_net, args.learning_rate)

    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 4250], gamma=0.5)
    lr_scheduler_IRN = optim.lr_scheduler.MultiStepLR(optimizer_IRN, milestones=[2000, 4250], gamma=0.5)
    lr_scheduler_En = optim.lr_scheduler.MultiStepLR(optimizer_En, milestones=[2000, 4250], gamma=0.5)

    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['state_dict'])
    best_loss = float("inf")
    
    # for train_one_epoch 
    (
        model, IRN_net, En_net, criterion, train_dataloader, optimizer, optimizer_IRN, optimizer_En, aux_optimizer,
        epoch, clip_max_norm, logger_train, tb_logger
    ) = (
        net,
        IRN_net,
        En_net,
        criterion,
        train_dataloader,
        optimizer,
        optimizer_IRN,
        optimizer_En,
        aux_optimizer,
        1, #epoch
        args.clip_max_norm,
        logger_train,
        tb_logger
    )
    return locals()

    ###############################################
    for epoch in range(last_epoch, args.epochs):
        # print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            IRN_net,
            En_net,
            criterion,
            train_dataloader,
            optimizer,
            optimizer_IRN,
            optimizer_En,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger
        )
        if epoch % 25 == 0:
            loss = test_epoch(epoch, test_dataloader, net, IRN_net, En_net, criterion, logger_val, tb_logger)

        lr_scheduler.step()
        lr_scheduler_IRN.step()
        lr_scheduler_En.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "state_dict": net.state_dict(),
                },
                is_best,
                "net",
                os.path.join('experiments', args.experiment, 'checkpoints', "net_checkpoint.pth.tar")
            )
            save_checkpoint(
                {
                    "state_dict": IRN_net.state_dict(),
                },
                is_best,
                "IRN_net",
                os.path.join('experiments', args.experiment, 'checkpoints',
                             "IRN_net_checkpoint.pth.tar")
            )
            save_checkpoint(
                {
                    "state_dict": En_net.state_dict(),
                },
                is_best,
                "En_net",
                os.path.join('experiments', args.experiment, 'checkpoints',
                             "En_net_checkpoint.pth.tar")
            )
            if is_best:
                logger_val.info('best checkpoint saved.')


if __name__ == "__main__":
    # main_cai(sys.argv[1:])
    main_cai()
