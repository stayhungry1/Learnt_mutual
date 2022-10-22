import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
import subprocess

import utils
from quantizer import quant_fix, dequant_fix
# from VTM_encoder import run_vtm
from VTM_encoder_ccr import run_vtm
from cvt_detectron_coco_oid_vivo import conversion
import scipy.io as sio
from typing import Tuple, Union
import PIL.Image as Image
import math
import json
import scipy.io as sio


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


class Eval:
    def __init__(self, settings, index) -> None:
        self.settings = settings
        self.set_idx = index
        self.VTM_param = settings["VTM"]
        print('load model path: %s' % (settings["pkl_path"]))
        self.model, self.cfg = utils.model_loader(settings)  # load模型进来
        self.prepare_dir()
        utils.print_settings(settings, index)

        self.pixel_num = settings["pixel_num"]

        compressai_lmbda = 1.0
        self.criterion = RateDistortionLoss(lmbda=compressai_lmbda)

        # 读取文件
        path_save = 'Openimage_numpixel_test5000.json'  # new_dict[fname_simple][0] [1] [2] 分别为height, width, num_pixel fname_simple为 '000a1249af2bc5f0'
        tf = open(path_save, "r")
        self.numpixel_test5000 = json.load(tf)

        # self.path_bppsave = 'output/cheng_onlycompressP2_bpp_lambda1e0.json' #P2inP2out
        # self.path_bppsave = '../../liutie_save/output/cheng_onlycompressP2outputP4_bpp_lambda1e0.json'
        # self.path_bppsave = '../../liutie_save/output/cheng_onlycompressP4outputP4zeropad128_bpp_lambda1e0_iter9999.json'
        # self.path_bppsave = '../../liutie_save/output/cheng_P4inP4outzeropad16_bpp_lambda1e0_iter39999.json'
        # self.path_bppsave = '../../liutie_save/output/cheng_P3inP3outzeropad128_bpp_lambda1e0_iter39999.json'
        # self.path_bppsave = '../../liutie_save/output/cheng_P3inP3outzeropad128_bpp_lambda1e0_finenet_iter33999.json'
        self.path_bppsave = '../../liutie_save/output/cheng_P2inP3outzeropad128_bpp_lambda1e0_finenet_iter37999.json'
        self.bpp_test5000 = {}

        self.input_format = "BGR"

    def prepare_dir(self):
        os.makedirs(f"../../liutie_save/info/{self.set_idx}", exist_ok=True)
        os.makedirs(f"../../liutie_save/feature/{self.set_idx}_ori", exist_ok=True)
        os.makedirs(f"../../liutie_save/feature/{self.set_idx}_ds", exist_ok=True)
        # os.makedirs(f"feature/{self.set_idx}_resid", exist_ok=True)
        os.makedirs(f"../../liutie_save/output", exist_ok=True)

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

        # storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):

            # for i in input:
            #     print(i)

            # classes = outputs[0]['instances'].pred_classes.to('cpu').numpy()
            # scores = outputs[0]['instances'].scores.to('cpu').numpy()
            # bboxes = outputs[0]['instances'].pred_boxes.tensor.to('cpu').numpy()
            # H, W = outputs[0]['instances'].image_size

            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            v_gt = v_gt.overlay_instances(boxes=None)
            anno_img = v_gt.get_image()

            h_temp = img.shape[1]
            w_temp = img.shape[2]
            print('hw:[%dx%d]' %(h_temp, w_temp))

            prop['instances'].pred_boxes[:, 0] = w_temp - prop['instances'].pred_boxes[:, 0]
            prop['instances'].pred_boxes[:, 2] = w_temp - prop['instances'].pred_boxes[:, 2]
            prop['instances'].pred_boxes[:, 1] = h_temp - prop['instances'].pred_boxes[:, 1]
            prop['instances'].pred_boxes[:, 3] = h_temp - prop['instances'].pred_boxes[:, 3]
            print('[%d, %d, %d, %d]' %(prop['instances'].pred_boxes[:, 0], prop['instances'].pred_boxes[:, 1], prop['instances'].pred_boxes[:, 2], prop['instances'].pred_boxes[:, 3]))

            # box_size = min(len(prop.proposal_boxes), max_vis_prop)
            # print(prop.proposal_boxes[0:box_size].pred_boxes.tensor.cpu().numpy().shape)
            # print(prop.proposal_boxes[0:box_size].pred_boxes.tensor.cpu().numpy())
            print(prop['instances'].pred_boxes.tensor.to('cpu').numpy().shape) #[59, 4,]
            # print(prop['instances'].scores.to('cpu').numpy().shape) #[59]
            # print(prop['instances'].pred_classes.to('cpu').numpy().shape) #[59]
            box_size = min(len(prop['instances'].pred_boxes), max_vis_prop)
            print(box_size)

            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                # boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
                boxes = prop['instances'].pred_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            # vis_img = vis_img.transpose(2, 0, 1)
            # vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"

            print(vis_img.shape)
            path_savevisualize = '../../liutie_save/feature/1.png'
            print(np.max(vis_img))
            print(np.min(vis_img))
            print(vis_img.dtype)

            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_savevisualize, vis_img)
            # storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward_front(self, inputs, images, features):
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        # self.visualize_training(inputs, proposals)
        return self.model._postprocess(results, inputs, images.image_sizes)

    def feature_coding(self):
        print("Saving features maps...")
        print('min/max_test_size:%d/%d' % (self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST))
        filenames = []
        with tqdm(total=len(self.data_loader.dataset)) as pbar:
            for inputs in iter(self.data_loader):
                # 自己加入的5行，断了之后重新跑，提过feature的不用再提
                fname_temp = utils.simple_filename(inputs[0]["file_name"])
                self.height_temp = self.numpixel_test5000[fname_temp][0]
                self.width_temp = self.numpixel_test5000[fname_temp][1]
                self.numpixel_temp = self.numpixel_test5000[fname_temp][2]
                fea_path = f"../../liutie_save/feature/{self.set_idx}_ori/"
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
        print('average bpp: %8.4f' % (bpp_sum / i_count))
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
        net_belle_output = self.model.net_belle(d_p2)
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
        res = self.model.netG.forward(up_image)
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

        fname = utils.simple_filename(inputs[0]["file_name"])
        fname_feat = f"../../liutie_save/feature/{self.set_idx}_ori/{fname}.png"  # 用于存P345
        # fname_p345 = f"feature/{self.set_idx}_p345/{fname}.png"
        fname_ds = f"../../liutie_save/feature/{self.set_idx}_ds/{fname}.png"  # 用于存P2
        # fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"

        with open(f"../../liutie_save/info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
            torch.save(inputs, inputs_f)

        # utils.save_feature_map(fname_feat, image_feat)
        ####################################ccr added 3 parts
        utils.save_feature_map_onlyp2(fname_ds, cheng_feat)  # 用于存P2
        # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        utils.save_feature_map_p345(fname_feat, image_feat)  # 用于存P345
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

    def evaluation(self, inputs):
        with open(f"../../liutie_save/output/{self.set_idx}_coco.txt", 'w') as of:
            of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')

            coco_classes_fname = 'oi_eval/coco_classes.txt'

            with open(coco_classes_fname, 'r') as f:
                coco_classes = f.read().splitlines()

            # for fname in tqdm(inputs[:2]):
            for fname in tqdm(inputs):

                fname_simple_temp = utils.simple_filename(fname)
                if fname_simple_temp != '00a159a661a2f5aa':
                    continue

                outputs = self._evaluation(fname)
                outputs = outputs[0]

                imageId = os.path.basename(fname)
                classes = outputs['instances'].pred_classes.to('cpu').numpy()
                scores = outputs['instances'].scores.to('cpu').numpy()
                bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
                H, W = outputs['instances'].image_size

                bboxes = bboxes / [W, H, W, H]
                bboxes = bboxes[:, [0, 2, 1, 3]]

                for ii in range(len(classes)):
                    coco_cnt_id = classes[ii]
                    class_name = coco_classes[coco_cnt_id]

                    rslt = [imageId[:-4], class_name, scores[ii]] + \
                           bboxes[ii].tolist()

                    o_line = ','.join(map(str, rslt))

                    of.write(o_line + '\n')

        conversion(self.set_idx)
        cmd = f"python oid_challenge_evaluation.py \
        --input_annotations_boxes   oi_eval/detection_validation_5k_bbox.csv \
        --input_annotations_labels  oi_eval/detection_validation_labels_5k.csv \
        --input_class_labelmap      oi_eval/coco_label_map.pbtxt \
        --input_predictions         ../../liutie_save/output/{self.set_idx}_oi.txt \
        --output_metrics            ../../liutie_save/output/{self.set_idx}_AP.txt"
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

        fname_simple = utils.simple_filename(fname)
        print(fname)
        # fname_ds_rec = fname.replace('rec', 'ds_rec')
        # # fname_resid_rec = fname.replace('rec', 'resid_rec')
        fname_ds_rec = fname.replace('ori', 'ds')  # QP36原始
        # fname_ds_rec = fname.replace('36_ori', '42_ds') #QP36
        ##lambda1: 33+103
        # fname_p4p5 = fname.replace('33_ori', '103_ori')  #P4P5
        ##lambda2: 35+104
        # fname_p4p5 = fname.replace('35_ori', '104_ori')  #P4P5
        ## lambda1chu2: 42+106
        # fname_p4p5 = fname.replace('42_ori', '106_ori')  # P4P5
        ## lambda1chu2: 47+106
        # fname_p4p5 = fname.replace('47_ori', '106_ori')  # P4P5
        ## lambda1chu4: 43+107
        # fname_p4p5 = fname.replace('43_ori', '107_ori')  # P4P5
        ## lambda1chu4: 43+107
        # fname_p4p5 = fname.replace('43_ori', '111_ori')  # P4P5
        ## lambda1chu8: 48+108
        # fname_p4p5 = fname.replace('48_ori', '108_ori')  # P4P5
        ## lambda1chu8: 48+108
        # fname_p4p5 = fname.replace('55_ori', '113_ori')  # P4P5
        # lambda4: 51+109
        fname_p4p5 = fname.replace('51_ori', '109_ori')  # P4P5
        ## lambda4: 50+109
        # fname_p4p5 = fname.replace('50_ori', '109_ori')  # P4P5
        ## lambda8: 53+112
        # fname_p4p5 = fname.replace('53_ori', '112_ori')  # P4P5

        with open(f"../../liutie_save/info/{self.set_idx}/{fname_simple}_inputs.bin", "rb") as inputs_f:
            inputs = torch.load(inputs_f)

        images = self.model.preprocess_image(inputs)
        features = self.feat2feat_p345(fname)  # P3P4P5P6 float32
        features_ds = self.feat2feat_onlyp2(fname_ds_rec)  # P2 float32
        features_p4p5 = self.feat2feat_p345(fname_p4p5)
        # features_resid = self.feat2feat_onlyp2(fname_resid_rec)
        # resid = features_resid['p2']
        features['p2'] = features_ds['p2']  # 给features(只有P3456)加上P2
        features['p4'] = features_p4p5['p4']  #把103_ori的P4拿来用
        features['p5'] = features_p4p5['p5']  #把103_ori的P5拿来用
        features['p2'] = features['p2'].type(torch.float64)
        features['p3'] = features['p3'].type(torch.float64)
        features['p4'] = features['p4'].type(torch.float64)
        features['p5'] = features['p5'].type(torch.float64)
        features['p6'] = features['p6'].type(torch.float64)

        # upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        # print(features_ds['p2'].size(), '-------------FineNet input (before upsample)')
        # up_image = upsample(features_ds['p2'])
        # print(up_image.size(), '-----------------FineNet input')
        # res = self.model.netG.forward(up_image)
        # print(res.size(), '------------------FineNet output')
        # fake_image_f = res + up_image
        # # real_image = fake_image_f + resid
        # features['p2'] = fake_image_f
        # print('max/min_P2(Upsample before FineNet): %8.4f/%8.4f, max/min_P2(FineNet output): %8.4f/%8.4f' %(torch.max(up_image), torch.min(up_image), torch.max(fake_image_f), torch.min(fake_image_f)))
        # fname_finenetoutput = f"feature/{self.set_idx}_finenetoutput/{fname_simple}.png"
        # ##print(fname_finenetoutput)
        # #features_finenetoutput = quant_fix(features.copy())
        # # utils.save_feature_map_onlyp2(fname_finenetoutput, features_finenetoutput) #ds_feat包括p2p3p4p5p6，进去后会删去p3p4p5p6
        #
        # #features['p2'] = up_image
        # ###################################################ccr added
        # # features_345 = self.feat2feat(fname_345)
        # # compG_input = features['p2']
        # # comp_image = self.model.compG.forward(compG_input)
        # # upsample = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        # # up_image = upsample(comp_image)
        # # input_fconcat = up_image
        # # res = self.model.netG.forward(input_fconcat)
        # # fake_image_f = res + up_image
        # # features['p2'] = fake_image_f
        # # features['p3'] = features_345['p3']
        # # features['p4'] = features_345['p4']
        # # features['p5'] = features_345['p5']
        # ###################################################ccr added
        outputs = self.forward_front(inputs, images, features)  # images是float64
        self.evaluator.process(inputs, outputs)
        self.visualize_training(inputs, outputs)
        return outputs

    def summary(self):
        with open("../../liutie_save/results.csv", "a") as result_f:
            # with open(f"inference/{self.set_idx}_AP.txt", "rt") as ap_f:
            with open(f"../../liutie_save/output/{self.set_idx}_AP.txt", "rt") as ap_f:
                ap = ap_f.readline()
                ap = ap.split(",")[1][:-1]

            size_basis = utils.get_size(f'../../liutie_save/feature/{self.set_idx}_bit/')

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
        utils.pick_coco_exp(data_name, myarg)
        self.data_loader = build_detection_test_loader(self.cfg, data_name)
        self.evaluator = COCOEvaluator(data_name, self.cfg, False)
        self.evaluator.reset()
        print(data_name, "Loaded")
