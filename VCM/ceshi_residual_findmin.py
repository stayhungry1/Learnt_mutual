import os
import glob
import utils
from PIL import Image
import numpy as np
import cv2

def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size

def simple_filename(filename_ext):
    filename_base = os.path.basename(filename_ext)
    filename_noext = os.path.splitext(filename_base)[0]
    return filename_noext

# set_idx = 35
filenames = glob.glob(f"/media/data/liutie/VCM/OpenImageV6-5K/*.jpg")
# filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/m57343_objdet_small_twoimage/*.jpg")
# filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCM_EE1.2_P-layer_feature_map_anchor_generation_137th_MPEG-VCM-main/feature/{set_idx}_rec/*.png")
num_img = len(filenames)
path_smallF_qianzhui = './feature/40_down2ressmall_ori/' #QP43 smallF
path_bigF_qianzhui = './feature/40_down2resbig_ori/' #QP43 bigF
path_saveres_qianzhui = './feature/40_down2res_ori/' #QP43 res
# path_smallF_qianzhui = './feature/43_down2res1small_ori/' #QP43 smallF
# path_bigF_qianzhui = './feature/43_down2res1big_ori/' #QP43 bigF
# path_saveres_qianzhui = './feature/43_down2res1_ori/' #QP43 res
os.makedirs(path_saveres_qianzhui, exist_ok=True)  # 用于保存原始分辨率的img输入

#0706短边416长边832时的_min
_min = -476.0
# #0710短边400长边800时的_min allres_minmax[-491.488, 462.918]
# _min = -491.5

i_count = 0
i_count_valid = 0
i_count_invalid = 0
bit_sum = 0
num_pixel_sum = 0
bpp_sum = 0
bit_all = np.zeros((num_img))
num_pixel_all = np.zeros((num_img))
bpp_all = np.zeros((num_img))
feat_res_all_min = 0
feat_res_all_max = 0
for fname in filenames:
    fname_simple = utils.simple_filename(fname) #000a1249af2bc5f0
    #debug确认用
    # if fname_simple != '000a1249af2bc5f0':
    #     continue
    # if fname_simple == '682d35f9b90a2b4c':
    #     continue
    path_smallF = path_smallF_qianzhui + fname_simple + '.png'
    path_bigF = path_bigF_qianzhui + fname_simple + '.png'
    path_res = path_saveres_qianzhui + fname_simple + '.png'
    # feat_small = Image.open(path_smallF).convert('RGB')
    # feat_big = Image.open(path_bigF).convert('RGB')
    feat_small = cv2.imread(path_smallF, -1).astype(np.float32)  # ndarray [2040, 2432]
    feat_big = cv2.imread(path_bigF, -1).astype(np.float32)  # ndarray [2040, 2432]
    # h_small = feat_small.size[1] #2210 int
    # w_small = feat_small.size[0] #2560 int
    # h_big = feat_big.size[1] #4420 int
    # w_big = feat_big.size[0] #5120 int
    h_small = feat_small.shape[0] #2210 int
    w_small = feat_small.shape[1] #2560 int
    h_big = feat_big.shape[0] #4420 int
    w_big = feat_big.shape[1] #5120 int
    #不再判断是否长宽均为严格2倍
    feat_small_resize = cv2.resize(feat_small, (w_big, h_big), interpolation=cv2.INTER_CUBIC).astype(np.float32)  # ndarray [2040, 2432]
    # if (h_small*2.0 == h_big) and (w_small*2.0 == w_big):
    #     # feat_small_resize = feat_small.resize((w_big, h_big))
    #     feat_small_resize = cv2.resize(feat_small, (w_big, h_big), interpolation=cv2.INTER_CUBIC).astype(np.float32) #ndarray [2040, 2432]
    #     i_count_valid = i_count_valid + 1
    # else:
    #     i_count_invalid = i_count_invalid + 1
    #     print('!!!hw_smallF not 1/2 of hw_bigF!!! %d/%d, smallF_hw[%dx%d], bigF_hw[%dx%d], imgname: %s'
    #           %(i_count, num_img, h_small, w_small, h_big, w_big, fname_simple))
    #     continue
    feat_res = feat_big - feat_small_resize
    # 注释掉这2行和下面imwrite来求阈值
    feat_res = feat_res - _min
    feat_res = feat_res.astype(np.uint16) #uint16
    feat_res_min = np.min(feat_res)
    feat_res_max = np.max(feat_res)
    if i_count == 0:
        feat_res_all_min = feat_res_min
        feat_res_all_max = feat_res_max
    else:
        if feat_res_min <= feat_res_all_min:
            feat_res_all_min = feat_res_min
        if feat_res_max >= feat_res_all_max:
            feat_res_all_max = feat_res_max
    i_count = i_count + 1
    cv2.imwrite(path_res, feat_res) #cv2.imwrite得到的png为uint8或uint16，像素最小值大于0
    # print('%d/%d, smallF_hw[%dx%d], bigF_hw[%dx%d], imgname: %s, res_minmax[%d, %d], allres_minmax[%d, %d], count_all/valid/invalid: [%d/%d/%d]'
    #       %(i_count, num_img, h_small, w_small, h_big, w_big, fname_simple, feat_res_min, feat_res_max, feat_res_all_min, feat_res_all_max, i_count, i_count_valid, i_count_invalid))
    print('%d/%d, smallF_hw[%dx%d], bigF_hw[%dx%d], imgname: %s, res_minmax[%7.3f, %7.3f], allres_minmax[%7.3f, %7.3f]'
          %(i_count, num_img, h_small, w_small, h_big, w_big, fname_simple, feat_res_min, feat_res_max, feat_res_all_min, feat_res_all_max))




