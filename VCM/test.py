import argparse
import csv
import glob
import json
import os

import utils
from eval_vivo_xxx import DetectEval

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["DETECTRON2_DATASETS"] = '../../OpenImageV6-5K'

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
        settings = json.load(setting_json)

    if settings["model_name"] == "x101":
        methods_eval = DetectEval(settings, set_idx)
        picklist = sorted(glob.glob(os.path.join(os.environ["DETECTRON2_DATASETS"], "*.jpg")))[:number]
        picklist = [utils.simple_filename(x) for x in picklist]
        methods_eval.prepare_part(picklist, data_name="pick")

    if mode == "feature_coding":
        filenames = methods_eval.feature_coding()

    elif mode == "evaluation":
        filenames = glob.glob(f"../../feature_save/feature/{set_idx}_ori/*.png")
        methods_eval.evaluation(filenames)

    elif mode == "evaluation_offline":
        methods_eval.evaluation_offline()
