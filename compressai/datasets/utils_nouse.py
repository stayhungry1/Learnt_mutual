# m58786/utils.py
import glob
import json
import os
import shutil
from datetime import datetime
from tkinter import ttk
import torch.nn as nn
import torch.nn.functional as F

import imagesize
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
import numpy as np
import torch
# import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import copy

def simple_filename(filename_ext):
    filename_base = os.path.basename(filename_ext)
    filename_noext = os.path.splitext(filename_base)[0]
    return filename_noext

def model_loader(settings):
    cfg = get_cfg()
    cfg.merge_from_file(settings["yaml_path"])
    cfg.MODEL.WEIGHTS = settings["pkl_path"]
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    model = DefaultPredictor(cfg).model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, cfg

def pick_coco_exp(name, targetlist):
    # print(">>>> name: ", name)
    if os.path.isdir(name): # ml commented
        shutil.rmtree(name)
    os.makedirs(name, exist_ok=True)

    coco_path = os.environ["DETECTRON2_DATASETS"]
    anno_path = "./dataset/annotations/instances_OpenImage_v6.json"    

    file_list = glob.glob(os.path.join(coco_path, "*.jpg"))
    file_list = [x for x in file_list if simple_filename(x) in targetlist]

    file_name_list = [os.path.basename(x) for x in file_list]
    with open(anno_path, "r") as anno_file:
        coco_json = json.load(anno_file)
    my_json = {}
    my_json["info"] = coco_json["info"]
    my_json["licenses"] = coco_json["licenses"]
    my_json["images"] = []
    my_json["annotations"] = []
    my_json["categories"] = coco_json["categories"]

    my_json["images"].extend(
        [x for x in coco_json["images"] if x["file_name"] in file_name_list]
    )
    image_id_list = [x["id"] for x in my_json["images"]]
    my_json["annotations"].extend(
        [x for x in coco_json["annotations"] if x["image_id"] in image_id_list]
    )

    for filepath in file_list:
        shutil.copy(filepath, name)
    with open(f"{name}/my_anno.json", "w") as my_file:
        my_file.write(json.dumps(my_json))
    register_coco_instances(name, {}, f"{name}/my_anno.json", name)

def print_settings(settings, index):
    model_name = settings["model_name"]
    VTM_param = settings["VTM"]
    print()
    print("Evaluation of proposed methods for:", model_name.upper())
    print(f"Settings ID: {index}")
    print(f"VTM paramerters      : {VTM_param}")


import cv2
import numpy as np

def save_feature_map(filename, features, data_type='np.uint16', save_channel_num=256):
    features_draw = features.copy()
    del features_draw["p6"]
    _save_feature_map(filename, features_draw, data_type=data_type, save_channel_num=save_channel_num)

def compute_min_val_index(list_val):
    min_number = []
    min_index = []
    
    t = copy.deepcopy(list_val)
    for _ in range(len(list_val)):
        number = min(t)
        index = t.index(number)
        t[index] = 10000000000
        min_number.append(number)
        min_index.append(index)

    return min_index, min_number

def cal_varience(fetures):
    fetures = fetures.cpu().numpy()
    variences = [np.var(i) for i in fetures]
    
    return variences


def set_zeros(features, save_channel_num):
    variences = cal_varience(features)
    min_index, min_val = compute_min_val_index(variences)
    zeros_index = min_index[:save_channel_num]
    for i in zeros_index:
        features[i, :, :] = 0

    return features

def resize_feature(features, save_channel_num):
    _, h_out1, w_out1 = features.size()
    features_resize = copy.deepcopy(features)
    #
    if save_channel_num == 111:
        pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        unpool = nn.MaxUnpool2d(2, stride=2)
        output, indices = pool(features_resize)
        features_resize = unpool(output, indices)
    elif save_channel_num in [112, 113, 114, 115, 116]:
        # max_val = torch.max(features_resize)
        # min_val = torch.min(features_resize)
        features_resize = torch.unsqueeze(features_resize, 0)
        features_resize = F.interpolate(features_resize, scale_factor=(0.5,0.5), mode="bilinear", align_corners=False)
        features_resize = F.interpolate(features_resize, scale_factor=(2,2), mode="bilinear", align_corners=False)
        features_resize = torch.squeeze(features_resize, 0)
    # a = 1
    # tt
    return features_resize

def _save_feature_map(filename, features, debug=False, data_type='np.uint16', save_channel_num=256):
    feat = [features["p2"].squeeze(), features["p3"].squeeze(), features["p4"].squeeze(), features["p5"].squeeze()]

    if not data_type in ['np.uint16'] and save_channel_num >= 1:
        if save_channel_num in [111, 112]:
            features_zeroed = resize_feature(feat[0], save_channel_num)
            feat[0] = features_zeroed
        elif save_channel_num in [113]: # do bilinear for P3
            features_zeroed = resize_feature(feat[1], save_channel_num)
            feat[1] = features_zeroed
        elif save_channel_num in [114]: # do bilinear for P4
            features_zeroed = resize_feature(feat[2], save_channel_num)
            feat[2] = features_zeroed
        elif save_channel_num in [115]: # do bilinear for P5
            features_zeroed = resize_feature(feat[3], save_channel_num)
            feat[3] = features_zeroed
        elif save_channel_num in [116]: # do bilinear for P5
            features_zeroed_p2 = resize_feature(feat[0], save_channel_num)
            features_zeroed_p3 = resize_feature(feat[1], save_channel_num)
            features_zeroed_p4 = resize_feature(feat[2], save_channel_num)
            features_zeroed_p5 = resize_feature(feat[3], save_channel_num)

            feat[0] = features_zeroed_p2
            feat[1] = features_zeroed_p3
            feat[2] = features_zeroed_p4
            feat[3] = features_zeroed_p5
        else:
            features_zeroed = set_zeros(feat[0], save_channel_num)
            feat[0] = features_zeroed

        # features_zeroed = feat[0]
        # for i in range(256):
        #     feat[1][i, :, :] = 0
        #     feat[2][i, :, :] = 0
        #     feat[3][i, :, :] = 0

    width_list = [16, 32, 64, 128]
    height_list = [16, 8, 4, 2]
    tile_big = np.empty((0, feat[0].shape[2] * width_list[0]))
    for blk, width, height in zip(feat, width_list, height_list):
        big_blk = np.empty((0, blk.shape[2] * width))
        for row in range(height):
            big_blk_col = np.empty((blk.shape[1], 0))
            for col in range(width):
                tile = blk[col + row * width].cpu().numpy()
                if debug:
                    cv2.putText(
                        tile,
                        f"{col + row * width}",
                        (32, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                big_blk_col = np.hstack((big_blk_col, tile))
            big_blk = np.vstack((big_blk, big_blk_col))
        tile_big = np.vstack((tile_big, big_blk))
    
    if data_type in ['np.uint16']:
        tile_big = tile_big.astype(np.uint16)
        cv2.imwrite(filename, tile_big)
    else:
        # print(">>>> filename: ", filename)
        # sub_path = filename[:-4]
        # if not os.path.exists(sub_path):
        #     os.makedirs(sub_path)
        
        # # p2
        # filename_p2 = sub_path + "/p2.npy"
        # filename_p3 = sub_path + "/p3.npy"
        # filename_p4 = sub_path + "/p4.npy"
        # filename_p5 = sub_path + "/p5.npy"

        # np.save(filename_p2, features["p2"].cpu().numpy())
        # np.save(filename_p3, features["p3"].cpu().numpy())
        # np.save(filename_p4, features["p4"].cpu().numpy())
        # np.save(filename_p5, features["p5"].cpu().numpy())

        tile_big = tile_big.astype(np.float32)

        # plt.imshow(tile_big)
        # plt.show()
        # plt.savefig('t1_raw.jpg')
        # tt

        np.save(filename, tile_big)





def result_in_list(settings, number, result, set_index):
    res = list(result.values())[0]
    ap = res["AP"]
    ap50 = res["AP50"]
    aps = res["APs"]
    apm = res["APm"]
    apl = res["APl"]

    return [
        datetime.now(),
        set_index,
        number,
        f"{ap:.3f}",
        f"{ap50:.3f}",
        f"{aps:.3f}",
        f"{apm:.3f}",
        f"{apl:.3f}",
    ]

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
