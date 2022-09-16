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
# path_smallF_qianzhui = './feature/40_down2ressmall_ori/' #QP40 smallF
# path_bigF_qianzhui = './feature/40_down2resbig_ori/' #QP40 bigF
# path_saveres_qianzhui = './feature/40_down2res_ori/' #QP40 res
# #下面3行为smallF(rec)resize+bigR(rec)
# path_smallF_qianzhui = './feature/43_down2res1small_rec/' #QP43 smallF
# path_bigR_qianzhui = './feature/43_down2res1_rec/' #QP43 bigR
# path_savebigF_qianzhui = './feature/43_down2res1big_addsaverec/' #QP43 bigF
#下面3行为smallF(ori)resize+bigR(rec)
# path_smallF_qianzhui = './feature/43_down2res1small_ori/' #QP43 smallF
path_smallF_qianzhui = './feature/43_down2res1small_rec/' #QP43 smallF
path_bigR_qianzhui = './feature/43_down2res1_ori/' #QP43 bigR
# path_savebigF_qianzhui = './feature/43_down2res1big_smallForiADDbigRrecsave/' #QP43 bigF
path_savebigF_qianzhui = './feature/43_down2res1big_smallFrecADDbigRorisave/' #QP43 bigF

# #0706短边416长边832时的_min
# _min = -476.0
#0710短边400长边800时的_min allres_minmax[-491.488, 462.918]
_min = -491.5

i_count = 0
i_count_valid = 0
i_count_invalid = 0
bit_sum = 0
num_pixel_sum = 0
bpp_sum = 0
bit_all = np.zeros((num_img))
num_pixel_all = np.zeros((num_img))
bpp_all = np.zeros((num_img))
feat_bigF_all_min = 0
feat_bigF_all_max = 0
for fname in filenames:
    fname_simple = utils.simple_filename(fname) #000a1249af2bc5f0
    # #debug确认用
    # if fname_simple != '000a1249af2bc5f0':
    #     continue
    path_smallF = path_smallF_qianzhui + fname_simple + '.png'
    path_bigR = path_bigR_qianzhui + fname_simple + '.png'
    path_bigF = path_savebigF_qianzhui + fname_simple + '.png'
    # feat_small = Image.open(path_smallF).convert('RGB')
    # feat_big = Image.open(path_bigF).convert('RGB')
    feat_small = cv2.imread(path_smallF, -1).astype(np.float32)  # ndarray [2040, 2432]
    feat_bigR = cv2.imread(path_bigR, -1).astype(np.float32)  # ndarray [2040, 2432]
    feat_bigR = feat_bigR + _min
    h_small = feat_small.shape[0] #2210 int
    w_small = feat_small.shape[1] #2560 int
    h_big = feat_bigR.shape[0] #4420 int
    w_big = feat_bigR.shape[1] #5120 int
    #不再判断是否长宽均为严格2倍
    feat_small_resize = cv2.resize(feat_small, (w_big, h_big), interpolation=cv2.INTER_CUBIC).astype(np.float32)  # ndarray [2040, 2432]
    feat_bigF = feat_small_resize + feat_bigR
    ## feat_res.save(path_res)
    feat_bigF = feat_bigF.astype(np.uint16)
    feat_bigF_min = np.min(feat_bigF)
    feat_bigF_max = np.max(feat_bigF)
    if i_count == 0:
        feat_bigF_all_min = feat_bigF_min
        feat_bigF_all_max = feat_bigF_max
    else:
        if feat_bigF_min <= feat_bigF_all_min:
            feat_bigF_all_min = feat_bigF_min
        if feat_bigF_max >= feat_bigF_all_max:
            feat_bigF_all_max = feat_bigF_max
    i_count = i_count + 1
    cv2.imwrite(path_bigF, feat_bigF) #cv2.imwrite得到的png为uint8或uint16，像素最小值大于0
    print('%d/%d, smallF_hw[%dx%d], bigF_hw[%dx%d], imgname: %s, bigF_minmax[%d, %d], allbigF_minmax[%d, %d], count_all/valid/invalid: [%d/%d/%d]'
          %(i_count, num_img, h_small, w_small, h_big, w_big, fname_simple, feat_bigF_min, feat_bigF_max, feat_bigF_all_min, feat_bigF_all_max, i_count, i_count_valid, i_count_invalid))
# bit_avg = bit_sum / num_img
# num_pixel_avg = num_pixel_sum / num_img
# bpp_avg = bpp_sum / num_img
# print('avg_bit: {:.3f}, sum_bit: {:.0f}, num_image: {:.0f}'.format(bit_avg, bit_sum, num_img))
# print('avg_numpixel: {:.0f}, sum_numpixel: {:.0f}, num_image: {:.0f}'.format(num_pixel_avg, num_pixel_sum, num_img))
# # print('[xiaotu] avg_bpp: {:.4f}, sum_bpp: {:.4f}, num_image: {:.0f}'.format(bpp_avg*4.0, bpp_sum, num_img))
# print('[yuantu] avg_bpp: {:.4f}, sum_bpp: {:.4f}, num_image: {:.0f}'.format(bpp_avg, bpp_sum, num_img))



