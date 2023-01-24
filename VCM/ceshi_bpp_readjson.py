######cocotest5000_bpp
import json

path_save = '/media/data/liutie/VCM/rcnn/liutie_save/output/chenganchor_bpp_lambda4_cocofinetune_cocotest5000_hw576.json'
# path_save = '/media/data/liutie/VCM/rcnn/liutie_save/output/chenganchor_bpp_lambda1_cocotest5000_hw576_P5.json'
tf = open(path_save, "r")
bpp_alltest5000 = json.load(tf)
i_count_image = 0
sum_bpp = 0
for k,v in bpp_alltest5000.items():
    i_count_image = i_count_image + 1
    sum_bpp = sum_bpp + v[0]
    print('%d: %6.4f, %s' %(i_count_image, v[0], k))
average_bpp = sum_bpp / i_count_image
print('num_image: %d, average_bpp: %6.4f, sum_bpp: %8.4f' % (i_count_image, average_bpp, sum_bpp))


