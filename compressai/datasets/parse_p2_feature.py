import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import math
# from .quantizer import dequant_fix

#最小公倍数
def gcd(a,b):
    if a < b:
        temp = b
        b = a 
        a = temp
    remainder = a % b
    if remainder == 0:
        return b
    else:
        return gcd(remainder,b)

def gys(a,b):
    remainder = gcd(a,b)
    return (a*b/remainder)
# gys(a,b)


# quantizer
import numpy as np
import torch

# _min = -23.1728
# _max = 20.3891

_scale = 23.4838
_min = -23.1728

def quant_fix(features):
    if type(features) == type({}):
        for name, pyramid in features.items():
            pyramid_q = (pyramid-_min) * _scale
            features[name] = pyramid_q
    elif type(features) == type([]):
        for ii in range(len(features)):
            pyramid = features[ii]
            pyramid_q = (pyramid-_min) * _scale
            features[ii] = pyramid_q
            # raise ValueError("max:{}, min:{}".format(pyramid_q.max(), pyramid_q.min()))
            # if np.sum(features[ii] > 60000):
            #     raise ValueError("find max > 60000")
    return features
    
def dequant_fix(x):
    return x.type(torch.float32)/_scale + _min
#######

def feature_slice(image, shape):
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

def feat2feat(fname):
    pyramid = {}

    # np load
    # png = np.load(fname)
    # png load
    fname = str(fname)
    try:
        png = cv2.imread(fname, -1).astype(np.float32)
    except:
        print("-------------")
        print(fname)
        raise ValueError("stop_fname")

    # ml
    # print(">>>> fname: ", fname)
    # np.save("save_channel_num_{}.png", png)
    # plt.imshow(png)
    # plt.savefig("save_channel_num_{}.png".format(save_channel_num))
    # tt

    vectors_height = png.shape[0]
    v2_h = int(vectors_height / 85 * 64)
    v3_h = int(vectors_height / 85 * 80)
    v4_h = int(vectors_height / 85 * 84)
    # raise ValueError("feature stop,png feature:{}".format(png))
    v2_blk = png[:v2_h, :]
    v3_blk = png[v2_h:v3_h, :]
    v4_blk = png[v3_h:v4_h, :]
    v5_blk = png[v4_h:vectors_height, :]

    pyramid["p2"] = feature_slice(v2_blk, [v2_blk.shape[0] // 16, v2_blk.shape[1] // 16 ])
    pyramid["p3"] = feature_slice(v3_blk, [v3_blk.shape[0] // 8 , v3_blk.shape[1] // 32 ])
    pyramid["p4"] = feature_slice(v4_blk, [v4_blk.shape[0] // 4 , v4_blk.shape[1] // 64 ])
    pyramid["p5"] = feature_slice(v5_blk, [v5_blk.shape[0] // 2 , v5_blk.shape[1] // 128])

    pyramid["p2"] = dequant_fix(pyramid["p2"])
    pyramid["p3"] = dequant_fix(pyramid["p3"])
    pyramid["p4"] = dequant_fix(pyramid["p4"])
    pyramid["p5"] = dequant_fix(pyramid["p5"])
    # raise ValueError("feature stop,png feature:{}".format(pyramid["p2"]))
    pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
    pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
    pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
    pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)

    pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

    # pyramid["p2"] = pyramid["p2"].cuda()
    # pyramid["p3"] = pyramid["p3"].cuda()
    # pyramid["p4"] = pyramid["p4"].cuda()
    # pyramid["p5"] = pyramid["p5"].cuda()
    # pyramid["p6"] = pyramid["p6"].cuda()

    return pyramid

# --------------------------------
#add by ywz
def padding_size(ori_size, factor_size):
    if ori_size % factor_size == 0:
        return ori_size
    else:
        return factor_size * (ori_size // factor_size + 1)

def _joint_split_features(feature, out_channels = 3):
    # feature = torch.zeros([1,256,272,198])
    #B,C,H,W
    factor_pixel = 64
    temp_channels = feature.shape[1]
    if feature.shape[2] == 200:
        # 200的边->8 = 1600 能够被64整除 整下的往另一个方向补 & 补64整数倍的黑边
        mode_factor = int(gys(feature.shape[2], factor_pixel)) // feature.shape[2] #整除边的贴的倍数
        target_mode_shape = mode_factor * feature.shape[2]
        target_mode_shape = padding_size(target_mode_shape, factor_pixel)
        #
        mode = "first" #
        the_other_shape = feature.shape[3]
    # elif feature.shape[3] == 200:
    else:
        # 200的边->8 = 1600 能够被64整除 整下的往另一个方向补 & 补64整数倍的黑边
        mode_factor = int(gys(feature.shape[3], factor_pixel)) // feature.shape[3] #整除边的贴的倍数
        target_mode_shape = mode_factor * feature.shape[3]
        target_mode_shape = padding_size(target_mode_shape, factor_pixel)
        #
        mode = "second" #
        the_other_shape = feature.shape[2]
    # print(mode)
    #256/8 = 32 (+1) >/ 3 = 11
    # 以下非hardcode
    the_other_factor = math.ceil((temp_channels/mode_factor) / out_channels) #另一个边的倍数
    target_other_shape = math.ceil(the_other_shape * the_other_factor / factor_pixel) * factor_pixel
    if mode == "first":
        #[2]
        height_factor = mode_factor
        width_factor = the_other_factor
        target_height = target_mode_shape
        target_width = target_other_shape
    elif mode == "second":
        #[3]
        height_factor = the_other_factor
        width_factor = mode_factor
        target_height = target_other_shape
        target_width = target_mode_shape
    # print("target height&width: {}&{}".format(target_height, target_width))
    merge_feature = None #np.zeros([feature.shape[0], out_channels, target_height, target_width])
    for batch_idx in range(feature.shape[0]):
        tile_big = None
        for channel_idx in range(out_channels):
            big_blk = None #np.empty((0, target_width))
            for row in range(height_factor):
                big_blk_col = None
                for col in range(width_factor):
                    temp_idx = channel_idx*width_factor*height_factor + row * width_factor + col
                    if temp_idx >= temp_channels:
                        # 补黑边
                        tile = np.zeros((feature.shape[-2],feature.shape[-1]))
                    else:
                        # 整个原channel贴过来
                        # tile = feature[batch_idx][temp_idx].cpu().numpy()
                        tile = feature[batch_idx][temp_idx].detach().cpu().numpy()
                    # print("tile shape:{}".format(tile.shape))
                    # print("big_blk_col {}, tile:{}".format(big_blk_col.shape if type(big_blk_col) != type(None) else None, tile.shape))
                    big_blk_col = np.concatenate((big_blk_col, tile), axis=-1) if type(big_blk_col) != type(None) else tile
                    # print("width concat:{}".format(big_blk_col.shape))
                # big_blk = np.vstack((big_blk, big_blk_col))
                big_blk = np.concatenate((big_blk, big_blk_col), axis=-2) if type(big_blk) != type(None) else big_blk_col
                # print("height concat:{}".format(big_blk.shape))
            # 通道维度拼
            tile_big = np.concatenate((tile_big, np.expand_dims(big_blk,0)), axis=-3) if type(tile_big) != type(None) else np.expand_dims(big_blk,0)
            # print("channel concat:{}".format(tile_big.shape)) #[3,1600,3344]
        merge_feature = np.concatenate((merge_feature, np.expand_dims(tile_big,0)), axis=0) if type(merge_feature) != type(None) else np.expand_dims(tile_big,0)

    merge_feature_pad = np.zeros([feature.shape[0], out_channels, target_height, target_width])
    merge_feature_pad[:,:,0:merge_feature.shape[-2],0:merge_feature.shape[-1]] = merge_feature[:,:,0:merge_feature.shape[-2],0:merge_feature.shape[-1]]
    # return merge_feature
    # print(merge_feature_pad.shape)
    return merge_feature_pad, mode
def _split_features_from_joint(feature, mode_input, ori_shape):
    # ------------------------------
    # 和joint部分相呼应，适配两种mode
    # feature = torch.zeros([1,256,272,198])
    #B,C,H,W
    factor_pixel = 64
    ori_channels = ori_shape[1]
    if ori_shape[2] == 200:
        # 200的边->8 = 1600 能够被64整除 整下的往另一个方向补 & 补64整数倍的黑边
        mode_factor = int(gys(ori_shape[2], factor_pixel)) // ori_shape[2] #整除边的贴的倍数
        target_mode_shape = mode_factor * ori_shape[2]
        target_mode_shape = padding_size(target_mode_shape, factor_pixel)
        #
        mode = "first" #
        the_other_shape = ori_shape[3]
    # elif ori_shape[3] == 200:
    else:
        # 200的边->8 = 1600 能够被64整除 整下的往另一个方向补 & 补64整数倍的黑边
        mode_factor = int(gys(ori_shape[3], factor_pixel)) // ori_shape[3] #整除边的贴的倍数
        target_mode_shape = mode_factor * ori_shape[3]
        target_mode_shape = padding_size(target_mode_shape, factor_pixel)
        #
        mode = "second" #
        the_other_shape = ori_shape[2]
    # print(mode)
    if mode != mode_input:
        raise ValueError("mode is not same when split")
    #256/8 = 32 (+1) >/ 3 = 11
    # 以下非hardcode
    the_other_factor = math.ceil((ori_channels/mode_factor) / feature.shape[1]) #另一个边的倍数
    target_other_shape = math.ceil(the_other_shape * the_other_factor / factor_pixel) * factor_pixel
    if mode == "first":
        #[2]
        height_factor = mode_factor
        width_factor = the_other_factor
        target_height = target_mode_shape
        target_width = target_other_shape
    elif mode == "second":
        #[3]
        height_factor = the_other_factor
        width_factor = mode_factor
        target_height = target_other_shape
        target_width = target_mode_shape
    # ---------------------------------------------------
    # 以上部分的变量注意名称意义可能不真实，下面是split过程真正的定义变量
    out_channels = ori_shape[-3]
    out_height = ori_shape[-2]
    out_width = ori_shape[-1]
    out_feature = np.zeros([feature.shape[0], out_channels, out_height, out_width])
    # split
    for batch_idx in range(feature.shape[0]):
        # split in one batch
        for out_feature_channel_idx in range(out_feature.shape[1]):
            # cal channel, start_height, start_width
            temp_channel_idx = out_feature_channel_idx // (height_factor*width_factor)
            temp_start_heigh_factor = (out_feature_channel_idx % (height_factor*width_factor)) // width_factor
            temp_start_width_factor = out_feature_channel_idx % width_factor
            # copy from big merge
            out_feature[batch_idx, out_feature_channel_idx, :, :] = feature[batch_idx, temp_channel_idx, temp_start_heigh_factor*out_height: (temp_start_heigh_factor+1)*out_height, temp_start_width_factor*out_width:(temp_start_width_factor+1)*out_width]
    return out_feature
# --------------------------------

def parse_p2_feature():
    img_list = os.listdir(feature_path)
    for i_img, name in enumerate(img_list):
        read_path = feature_path +  name
        features = feat2feat(read_path)
        p2_feature = features["p2"]
        
        print(">>>> p2_feature: ", p2_feature.size())
        
        #test joint
        # p2_feature = torch.squeeze(p2_feature, 0)
        #test joint
        # [1,256,272,200]
        np_val, mode = _joint_split_features(p2_feature,3)
        print(np_val.shape)
        print(np_val.dtype)
        
        # test split_from_joint
        p2_feature_reversion = _split_features_from_joint(np_val, mode, p2_feature.shape)
        print(p2_feature_reversion.shape)
        print(p2_feature_reversion.dtype)

        return

# add by minglang


def save_png(features, name, path, player):
    path_save = path + name #[:-4]

    # dir_list = [path_save]
    # for i_dir in dir_list:
    #     if not os.path.exists(i_dir):
    #         os.makedirs(i_dir)
    # print(">>>> features[0].astype(np.uint16): ", np.max(features[0].astype(np.uint16)), np.min(features[0].astype(np.uint16)))
    if player in ['all']:
        cv2.imwrite(path_save + '_p2.png', features[0].astype(np.uint16))
        cv2.imwrite(path_save + '_p3.png', features[1].astype(np.uint16))
        cv2.imwrite(path_save + '_p4.png', features[2].astype(np.uint16))
        cv2.imwrite(path_save + '_p5.png', features[3].astype(np.uint16))
    elif player in ['p4']:
        cv2.imwrite(path_save + '_p4.png', features[2].astype(np.uint16))
    elif player in ['p5']:
        cv2.imwrite(path_save + '_p5.png', features[3].astype(np.uint16))
    elif player in ['single']:
        cv2.imwrite(path_save + '.png', features.astype(np.uint16))

def fea2png(features = None, debug=False, feat = None):

    png_feature = []
    if feat:
        feat = feat
    else:
        feat = [features["p2"].squeeze(), features["p3"].squeeze(), features["p4"].squeeze(), features["p5"].squeeze()]

    #quant_fix - add by ywz
    feat = quant_fix(feat)
    #
    width_list = [16, 32, 64, 128]
    height_list = [16, 8, 4, 2]

    tile_big = np.empty((0, feat[0].shape[2] * width_list[0]))
    for blk, width, height in zip(feat, width_list, height_list):
        big_blk = np.empty((0, blk.shape[2] * width))
        for row in range(height):
            big_blk_col = np.empty((blk.shape[1], 0))
            for col in range(width):
                tile = blk[col + row * width]
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
        # print(">>>> tile_big: ", np.shape(tile_big), np.shape(big_blk))
        tile_big = np.vstack((tile_big, big_blk))

        png_feature.append(big_blk)

    # return png_feature
    return tile_big


if __name__ == "__main__":
    # feature_path = "/media/data/minglang/data/feature_compression/set_feature_zero_0525/rm_0_channels/"
    # feature_path = "/media/data/yangwenzhe/rm_0_channels/"
    feature_path = "../../ywzfiles/" #debug
    parse_p2_feature()