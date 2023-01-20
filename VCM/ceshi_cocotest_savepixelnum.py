import os
import glob
import utils
from PIL import Image
import numpy as np
import scipy.io as sio
import numpy as np
import json

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

filenames = glob.glob(f"/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM/datasets/coco/val2017/*.jpg")
num_img = len(filenames)
path_img_qianzhui = '/media/data/liutie/VCM/rcnn/VCMbelle_0622/VCM/datasets/coco/val2017/'

i_count = 0
bit_sum = 0
num_pixel_sum = 0
bpp_sum = 0
bit_all = np.zeros((num_img))
num_pixel_all = np.zeros((num_img))
bpp_all = np.zeros((num_img))
numpixel_test5000 = {}
for fname in filenames:
    fname_simple = utils.simple_filename(fname) #000a1249af2bc5f0
    # #debug确认用
    # if fname_simple != '000a1249af2bc5f0':
    #     continue
    path_img = path_img_qianzhui + fname_simple + '.jpg'
    im = Image.open(path_img).convert('RGB')
    height = im.size[1] #678 int
    width = im.size[0] #1024 int
    num_pixel = height * width #int
    numpixel_test5000[fname_simple] = [height, width, num_pixel]
    num_pixel_all[i_count] = num_pixel
    print('%d/%d, hw[%dx%d], pixel: %d, imgname: %s' %((i_count+1), num_img, height, width, num_pixel, fname_simple))
    i_count = i_count + 1
print('i_count: %d' %(i_count))
print('sum_numpixel: {:.0f}'.format(np.sum(num_pixel_all)))
fname_simple = '000000001296'
print(numpixel_test5000[fname_simple][0])
print(numpixel_test5000[fname_simple][1])
print(numpixel_test5000[fname_simple][2])
# sio.savemat('Openimage_numpixel_test5000.mat', {'numpixel_test5000': numpixel_test5000})
# np.save(path_save, numpixel_test5000)
path_save = 'cocotest5000_numpixel.json'
tf = open(path_save, "w")
json.dump(numpixel_test5000, tf)
tf.close()
