import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
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
from VTM_encoder import run_vtm
from cvt_detectron_coco_oid import conversion
import scipy.io as sio

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
                fea_path = f"feature/{self.set_idx}_ori/"
                if os.path.isfile(f"{fea_path}{fname_temp}.png"):
                    print(f"feature extraction: {fname_temp} skip (exist)")
                    continue
                filenames.append(self._feature_coding(inputs)) #inputs是filename 15d64d, height 680, width 1024, image_id 19877和image这个tensor [3, 800, 1205] uint8 大于1
                pbar.update()
        print("runvtm---------------------runvtmrunvtmrunvtmrunvtmrunvtmrunvtmrunvtm")
        run_vtm(f"feature/{self.set_idx}_ori", self.VTM_param["QP"], self.VTM_param["threads"])

        return filenames

    def _feature_coding(self, inputs):
        images = self.model.preprocess_image(inputs) #images: device cpu, image_sizes [800, 1205] tensor [1, 3, 800, 1216] torch.float32 cpu
        features = self.model.backbone(images.tensor)

        #################################ccr added
        compG_input = features['p2']
        print(compG_input.size(),'---------------CompNet_input')
        comp_image = self.model.compG.forward(compG_input)
        ####### replace CompNet(last sentence), to compare onlyCompNet(Down 2) and P2down2
        #comp_image = F.interpolate(compG_input, scale_factor=0.5, mode="bilinear", align_corners=False)  # [1, 256, h/4, w/4]->[1, 256, h/8, w/8]
        ##print( comp_image.size(),'-------------------- comp_image')
        upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        up_image = upsample(comp_image)
       ## print(up_image.size(),'--------------------------up_image')
       # input_fconcat = up_image
       # res = self.model.netG.forward(input_fconcat)
       ## print(res.size(),'------------------------res')
        ##fake_image_f = res + up_image
        ##resid_pic = compG_input - fake_image_f
        # features['p2'] = fake_image_f
        #################################ccr added
        print(comp_image.size(),'--------------CompNet output (before upsample)')
        print(up_image.size(),'--------------CompNet output (after upsample)')
        features_ds = features.copy()
        features_ds["p2"] = comp_image
        ds_feat = quant_fix(features_ds.copy())

        ##features_resid = features.copy()
        ##features_resid["p2"] = resid_pic
        ##resid_feat = quant_fix(features_resid.copy())
        #################################ccr added

        image_feat = quant_fix(features.copy())
        
        fname = utils.simple_filename(inputs[0]["file_name"])
        fname_feat = f"feature/{self.set_idx}_ori/{fname}.png"
        fname_p345 = f"feature/{self.set_idx}_p345/{fname}.png"
        fname_ds = f"feature/{self.set_idx}_ds/{fname}.png"
        fname_resid = f"feature/{self.set_idx}_resid/{fname}.png"

        with open(f"info/{self.set_idx}/{fname}_inputs.bin", "wb") as inputs_f:
            torch.save(inputs, inputs_f)

        # utils.save_feature_map(fname_feat, image_feat)
        ####################################ccr added 3 parts
        utils.save_feature_map_onlyp2(fname_ds, ds_feat)
        # utils.save_feature_map_onlyp2(fname_resid, resid_feat)
        utils.save_feature_map_p345(fname_feat, image_feat)
        ####################################ccr added 3 parts
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
        fname_ds_rec = fname.replace('rec', 'ds_rec')
        fname_resid_rec = fname.replace('rec', 'resid_rec')

        with open(f"info/{self.set_idx}/{fname_simple}_inputs.bin", "rb") as inputs_f:
            inputs = torch.load(inputs_f)

        images = self.model.preprocess_image(inputs)
        features = self.feat2feat_p345(fname)
        features_ds = self.feat2feat_onlyp2(fname_ds_rec)
        # features_resid = self.feat2feat_onlyp2(fname_resid_rec)
        # resid = features_resid['p2']

        upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        print(features_ds['p2'].size(), '-------------FineNet input (before upsample)')
        up_image = upsample(features_ds['p2'])
        print(up_image.size(), '-----------------FineNet input')
        res = self.model.netG.forward(up_image)
        print(res.size(), '------------------FineNet output')
        fake_image_f = res + up_image
        # real_image = fake_image_f + resid
        features['p2'] = fake_image_f
        #features['p2'] = up_image
        ###################################################ccr added
        # features_345 = self.feat2feat(fname_345)
        # compG_input = features['p2']
        # comp_image = self.model.compG.forward(compG_input)
        # upsample = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        # up_image = upsample(comp_image)
        # input_fconcat = up_image
        # res = self.model.netG.forward(input_fconcat)
        # fake_image_f = res + up_image
        # features['p2'] = fake_image_f
        # features['p3'] = features_345['p3']
        # features['p4'] = features_345['p4']
        # features['p5'] = features_345['p5']
        ###################################################ccr added
        outputs = self.forward_front(inputs, images, features)
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
