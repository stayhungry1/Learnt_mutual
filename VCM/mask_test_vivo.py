import argparse
import glob
import json
import os

import utils
from eval_vivo_mask import DetectEval

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/OpenImageV6-5K' #vivo
os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/zzf/vcm/VCM/dataset/validation' #vivo 59942提案数据
# os.environ["DETECTRON2_DATASETS"] = '/media/data/ccr/testimg2' #vivo  # 测试的两张图
# os.environ["DETECTRON2_DATASETS"] = '/media/data/liutie/VCM/OpenImageV6-5K' #30901
# os.environ["DETECTRON2_DATASETS"] = '/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/m57343_objdet_small_twoimage' #30901
# os.environ["DETECTRON2_DATASETS"] = '/media/data/liutie/VCM/rcnn/testimg2' #30901

#####同时跑多个记得修改这两处: 本文件的pick和eval的bppjson名字
#####如果用P2残差的话，改eval里面的QP和p2res_min这两处
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", default=31, type=int)
    parser.add_argument("-n", "--number", default=5000, type=int)
    parser.add_argument("-m", "--mode", default='feature_coding')
    # parser.add_argument("-m", "--mode", default='evaluation')

    args = parser.parse_args()
    set_idx = args.index
    number = args.number
    mode = args.mode

    with open(f"settings/{set_idx}.json", "r") as setting_json:
    # with open("settings/{set_idx}.json", "r") as setting_json:
        settings = json.load(setting_json)  # setting 中是model_name, yaml_path和pkl_path

    if settings["model_name"] == "x101":
        methods_eval = DetectEval(settings, set_idx)
        picklist = sorted(glob.glob(os.path.join(os.environ["DETECTRON2_DATASETS"], "*.jpg")))[:number]
        picklist = [utils.simple_filename(x) for x in picklist]
        # methods_eval.prepare_part(picklist, data_name="pick") # 14  18 new_train 21
        methods_eval.prepare_part(picklist, data_name="pick1") # 13  19 finetune 31 20 42 22
        # methods_eval.prepare_part(picklist, data_name="pick2") #15 41 43 23

    if mode == "feature_coding":
        filenames = methods_eval.feature_coding()

    elif mode == "evaluation":
        # filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/feature/{set_idx}_rec/*.png")
        # filenames = glob.lob(f"feature/{set_idx}_rec/*.png")
        filenames = glob.glob(f"../../zzf_save/feature/{set_idx}_ori/*.png") #cheng2020anchor只压缩P2, 所以用P345 original的文件夹42_ori
        methods_eval.evaluation(filenames)

    elif mode == "evaluation_offline":
        methods_eval.evaluation_offline()