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

import utils
from quantizer import quant_fix, dequant_fix
# from VTM_encoder import run_vtm
from VTM_encoder_ccr import run_vtm
from cvt_detectron_coco_oid import conversion
import scipy.io as sio
from typing import Tuple, Union
import PIL.Image as Image
import math
import json
import scipy.io as sio

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


class RateDistortionLoss(nn.Module): #只注释掉了109行的bpp_loss, 08021808又加上了
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    # def forward(self, output, target, lq, x_l, x_enh): #0.001
    def forward(self, output, target, height, width):  # 0.001 #, lq, x_l, x_enh
        N, _, _, _ = target.size()
        out = {}
        # num_pixels = N * H * W
        num_pixels = N * height * width

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


class Eval:
    def __init__(self, settings, index) -> None:
        self.settings = settings
        self.set_idx = index
        self.VTM_param = settings["VTM"]
        print('load model path: %s' %(settings["pkl_path"]))
        self.model, self.cfg = utils.model_loader(settings) #load模型进来
        self.prepare_dir()
        utils.print_settings(settings, index)
        
        self.pixel_num = settings["pixel_num"]

        compressai_lmbda = 1.0
        self.criterion = RateDistortionLoss(lmbda=compressai_lmbda)

        # 读取文件
        path_save = 'Openimage_numpixel_test5000.json' #new_dict[fname_simple][0] [1] [2] 分别为height, width, num_pixel fname_simple为 '000a1249af2bc5f0'
        tf = open(path_save, "r")
        self.numpixel_test5000 = json.load(tf)

        # self.path_bppsave = 'output/cheng_onlycompressP2_bpp_lambda1e0.json' #P2inP2out
        self.path_bppsave = 'output/cheng_onlycompressP2outputP4_bpp_lambda1e0.json'
        self.bpp_test5000 = {}

    def prepare_dir(self):
        os.makedirs(f"info/{self.set_idx}", exist_ok=True)
        os.makedirs(f"feature/{self.set_idx}_ori", exist_ok=True)
        os.makedirs(f"feature/{self.set_idx}_ds", exist_ok=True)
        # os.makedirs(f"feature/{self.set_idx}_resid", exist_ok=True)
        os.makedirs(f"output", exist_ok=True)

    def forward_front(self, inputs, images, features):
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        return self.model._postprocess(results, inputs, images.image_sizes)

    def feature_coding(self):   
        print("Saving features maps...")
        print('min/max_test_size:%d/%d' %(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST))
        filenames = []
        with tqdm(total=len(self.data_loader.dataset)) as pbar:
            for inputs in iter(self.data_loader):
                #自己加入的5行，断了之后重新跑，提过feature的不用再提
                fname_temp = utils.simple_filename(inputs[0]["file_name"])
                self.height_temp = self.numpixel_test5000[fname_temp][0]
                self.width_temp = self.numpixel_test5000[fname_temp][1]
                self.numpixel_temp = self.numpixel_test5000[fname_temp][2]
                fea_path = f"feature/{self.set_idx}_ori/"
                if os.path.isfile(f"{fea_path}{fname_temp}.png"):
                    print(f"feature extraction: {fname_temp} skip (exist)")
                    continue
                filenames.append(self._feature_coding(inputs, fname_temp)) #inputs是filename 15d64d, height 680, width 1024, image_id 19877和image这个tensor [3, 800, 1205] uint8 大于1
                pbar.update()
        tf = open(self.path_bppsave, "r")
        bpp_test5000 = json.load(tf)
        bpp_sum = 0
        i_count = 0
        for key in bpp_test5000:
            bpp_temp = bpp_test5000[key]
            bpp_sum = bpp_sum + bpp_temp[0]
            i_count = i_count + 1
            print('i_count: %d, bpp: %8.4f, %s' %(i_count, bpp_test5000[key][0], key))
        print('average bpp: %8.4f' %(bpp_sum / i_count))
        print("####################### NOT run VTM!!! ###############################")
        # print("runvtm---------------------runvtmrunvtmrunvtmrunvtmrunvtmrunvtmrunvtm")
        # run_vtm(f"feature/{self.set_idx}_ori", self.VTM_param["QP"], self.VTM_param["threads"])

        return filenames

    def _feature_coding(self, inputs, fname_temp):
        # #加了这一行
        # self.model.net_belle.eval()
        # # self.model.net_belle.train()

        images = self.model.preprocess_image(inputs) #images: device cpu, image_sizes [800, 1205] tensor [1, 3, 800, 1216] torch.float32 cpu
        features = self.model.backbone(images.tensor)
        height_originalimage = images.image_sizes[0]
        width_originalimage = images.image_sizes[0]

        d = features['p2'] #[1, 256, 200, 304]
        d_p4 = features['p4'] #[1, 256, 200, 304]
        # guiyihua_min = torch.min(d)
        # guiyihua_scale = torch.max(d) - torch.min(d)
        # d = (d - guiyihua_min) / guiyihua_scale #d在0-1之间
        #normlize p2 and p4
        if torch.min(d) >= torch.min(d_p4): #2个数中取小的
            guiyihua_min = torch.min(d_p4)
        else:
            guiyihua_min = torch.min(d)
        if torch.max(d) >= torch.max(d_p4): #2个数中取大的
            guiyihua_max = torch.max(d)
        else:
            guiyihua_max = torch.max(d_p4)
        guiyihua_scale = guiyihua_max - guiyihua_min
        d = (d - guiyihua_min) / guiyihua_scale
        d_p4 = (d_p4 - guiyihua_min) / guiyihua_scale
        print(d.size(), '-------------------P2 original size')
        temp_ori_size_p2 = d.shape #P2原始尺寸
        temp_ori_size_p4 = d_p4.shape #P4原始尺寸
        target_size_p2 = [d.size()[0], d.size()[1], padding_size(d.size()[2], 16), padding_size(d.size()[3], 16)] #P2补黑边后(16的倍数) [1, 256, 208, 304]
        d_big = torch.zeros(target_size_p2).cuda()
        d_big[:, 0:temp_ori_size_p2[1], 0:temp_ori_size_p2[2], 0:temp_ori_size_p2[3]] = d
        print(d_big.size(), '-------------------Cheng input (P2) size')
        target_size_p4 = [d_p4.size()[0], d_p4.size()[1], int(target_size_p2[2] / 4.0), int(target_size_p2[3] / 4.0)] #P2的1/4
        d_big_p4 = torch.zeros(target_size_p4).cuda()
        d_big_p4[:, 0:temp_ori_size_p4[1], 0:temp_ori_size_p4[2], 0:temp_ori_size_p4[3]] = d_p4
        d_output = torch.zeros(temp_ori_size_p4) #用于从网络输出的tensor取出左上角
        net_belle_output = self.model.net_belle(d_big)
        print(net_belle_output["x_hat"].size(), '-------------------Cheng output (P4) size')
        d_output = net_belle_output["x_hat"][:, :, 0:temp_ori_size_p4[2], 0:temp_ori_size_p4[3]]
        print(d_output.size(), '-------------------output P4 size')
        print('max/min_p2(GT)(Cheng input): %8.4f/%8.4f, max/min_p4(GT): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f' %(torch.max(d), torch.min(d), torch.max(d_p4), torch.min(d_p4), torch.max(d_output), torch.min(d_output)))
        features_cheng = features.copy()
        features_p345 = features.copy()
        features_p345["p4"] = d_output * guiyihua_scale + guiyihua_min
        print('After denormlize: max/min_p2(GT)(Cheng input): %8.4f/%8.4f, max/min_p4(GT): %8.4f/%8.4f, max/min_P4(Cheng output): %8.4f/%8.4f' %(torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features["p4"]), torch.min(features["p4"]), torch.max(features_p345["p4"]), torch.min(features_p345["p4"])))

        d_output_p2 = torch.zeros(temp_ori_size_p2) #用于从refinenet输出的tensor取出左上角
        fake_image_f_GT = d #没有补黑边的P2
        upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        up_image = upsample(net_belle_output["x_hat"])
        res = self.model.netG.forward(up_image)
        fake_image_f = res + up_image
        d_output_p2 = fake_image_f[:, :, 0:temp_ori_size_p2[2], 0:temp_ori_size_p2[3]]
        print(d_output_p2.size(), '-------------------Finenet output P2 size')
        print('max/min_p4up(Finenet input): %8.4f/%8.4f, max/min_p2(GT): %8.4f/%8.4f, max/min_P2(Finenet output): %8.4f/%8.4f' %(torch.max(up_image), torch.min(up_image), torch.max(d), torch.min(d), torch.max(d_output_p2), torch.min(d_output_p2)))

        features_cheng["p2"] = d_output_p2 * guiyihua_scale + guiyihua_min
        cheng_feat = quant_fix(features_cheng.copy())
        print('After denormlize: max/min_p2(GT): %8.4f/%8.4f, max/min_P2(Finenet output): %8.4f/%8.4f' %(torch.max(features["p2"]), torch.min(features["p2"]), torch.max(features_cheng["p2"]), torch.min(features_cheng["p2"])))

        l_l2 = torch.nn.MSELoss().cuda()
        loss_l2 = l_l2(fake_image_f[:, :, 0:temp_ori_size_p2[2], 0:temp_ori_size_p2[3]], fake_image_f_GT)
        psnr_temp1 = 10 * math.log10(1 / loss_l2)

        loss_l2_0 = l_l2(up_image[:, :, 0:temp_ori_size_p2[2], 0:temp_ori_size_p2[3]], fake_image_f_GT)
        psnr_temp1_0 = 10 * math.log10(1 / loss_l2_0)
        dpsnr_temp = psnr_temp1 - psnr_temp1_0

        up_image_P4GT = upsample(d_p4)
        loss_l2_P4GT = l_l2(up_image_P4GT, fake_image_f_GT)
        psnr_temp1_P4GT = 10 * math.log10(1 / loss_l2_P4GT)

        heigh_temp = self.height_temp
        width_temp = self.width_temp
        numpixel_temp = self.numpixel_temp
        out_criterion = self.criterion(net_belle_output, d_big_p4, heigh_temp, width_temp)
        define_mse = nn.MSELoss()
        net_belle_output["x_hat"] = d_output #[1, 256, 208, 304]->[1, 256, 200, 304]
        out_criterion["mse_loss"] = define_mse(net_belle_output["x_hat"], d_p4)
        psnr_temp = mse2psnr(out_criterion['mse_loss'])
        print('bpp: %8.4f, MSE: %8.4f, PSNR: %8.4f' %(out_criterion["bpp_loss"].item(), out_criterion["mse_loss"].item(), psnr_temp))
        self.bpp_test5000[fname_temp] = [out_criterion["bpp_loss"].item()]
        tf = open(self.path_bppsave, "w")
        json.dump(self.bpp_test5000, tf)
        tf.close()

        print("FINENET MSE:%8.4f, dpsnr/psnr/psnr0: %8.4f/%8.4f/%8.4f, psnr_useP4GT: %8.4f, max/min_P2(GT): %8.4f/%8.4f, max/min_P4up(Finenet input): %8.4f/%8.4f, max/min_P2(FineNet output): %8.4f/%8.4f"
            % (loss_l2, dpsnr_temp, psnr_temp1, psnr_temp1_0, psnr_temp1_P4GT, torch.max(fake_image_f_GT), torch.min(fake_image_f_GT), torch.max(up_image), torch.min(up_image), torch.max(fake_image_f), torch.min(fake_image_f)))

        ##features_resid = features.copy()
        ##features_resid["p2"] = resid_pic
        ##resid_feat = quant_fix(features_resid.copy())
        #################################ccr added

        image_feat = quant_fix(features_p345.copy())

        fname = utils.simple_filename(inputs[0]["file_name"])
        fname_feat = f"feature/{self.set_idx}_ori/{fname}.png" #用于存P345
        # fname_p345 = f"feature/{self.set_idx}_p345/{fname}.png"
        fname_ds = f"feature/{self.set_idx}_ds/{fname}.png" #用于存P2
        # fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"

        with open(f"info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
            torch.save(inputs, inputs_f)

        # utils.save_feature_map(fname_feat, image_feat)
        ####################################ccr added 3 parts
        utils.save_feature_map_onlyp2(fname_ds, cheng_feat) #输入dict包括P2345，用于存P2
        # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        utils.save_feature_map_p345(fname_feat, image_feat) #输入dict包括P2345，用于存P345
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
        with open(f"./output/{self.set_idx}_coco.txt", 'w') as of:
            of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')

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

                for ii in range(len(classes)):
                    coco_cnt_id = classes[ii]
                    class_name = coco_classes[coco_cnt_id]

                    rslt = [imageId[:-4], class_name, scores[ii]] + \
                        bboxes[ii].tolist()

                    o_line = ','.join(map(str,rslt))

                    of.write(o_line + '\n')

        conversion(self.set_idx)
        cmd = f"python oid_challenge_evaluation.py \
        --input_annotations_boxes   oi_eval/detection_validation_5k_bbox.csv \
        --input_annotations_labels  oi_eval/detection_validation_labels_5k.csv \
        --input_class_labelmap      oi_eval/coco_label_map.pbtxt \
        --input_predictions         output/{self.set_idx}_oi.txt \
        --output_metrics            output/{self.set_idx}_AP.txt"
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
        fname_ds_rec = fname.replace('ori', 'ds') #QP36原始
        # fname_ds_rec = fname.replace('36_ori', '42_ds') #暂时没用:QP36不仅用P4，还用压缩后的P2

        with open(f"info/{self.set_idx}/{fname_simple}_inputs.bin", "rb") as inputs_f:
            inputs = torch.load(inputs_f)

        images = self.model.preprocess_image(inputs)
        features = self.feat2feat_p345(fname) #P3P4P5P6 float32
        features_ds = self.feat2feat_onlyp2(fname_ds_rec) #P2 float32
        # features_resid = self.feat2feat_onlyp2(fname_resid_rec)
        # resid = features_resid['p2']
        features['p2'] = features_ds['p2'] #给features(只有P3456)加上P2
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
        outputs = self.forward_front(inputs, images, features) #images是float64
        self.evaluator.process(inputs, outputs)
        return outputs
    
    def summary(self):
        with open("results.csv", "a") as result_f:
            # with open(f"inference/{self.set_idx}_AP.txt", "rt") as ap_f:
            with open(f"output/{self.set_idx}_AP.txt", "rt") as ap_f:
                ap = ap_f.readline()
                ap = ap.split(",")[1][:-1]

            size_basis = utils.get_size(f'feature/{self.set_idx}_bit/')

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

        pyramid["p2"] = self.feature_slice(v2_blk, [v2_blk.shape[0] // 16, v2_blk.shape[1] // 16 ])
        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

        pyramid["p2"] = dequant_fix(pyramid["p2"])
        pyramid["p3"] = dequant_fix(pyramid["p3"])
        pyramid["p4"] = dequant_fix(pyramid["p4"])
        pyramid["p5"] = dequant_fix(pyramid["p5"])

        pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
        pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
        pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
        pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)

        pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

        #加了下面这几句弄到cuda
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
        #加了下面这几句弄到cuda
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

        pyramid["p3"] = self.feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
        pyramid["p4"] = self.feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
        pyramid["p5"] = self.feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

        pyramid["p3"] = dequant_fix(pyramid["p3"])
        pyramid["p4"] = dequant_fix(pyramid["p4"])
        pyramid["p5"] = dequant_fix(pyramid["p5"])

        pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
        pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
        pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)
        pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

        #加了下面这几句弄到cuda
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
