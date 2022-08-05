
# vivo共用账号密码
内网gitlab的账号root密码buaamc2vivo 上传nginx网站的账号admin 密码a  三台内网机器互相ssh/scp也开了，账号root密码r

# vivo环境相关
https://vchat.vivo.xyz:8443/vivo/down/pc/index.html#/

72162669
ywz6.62607

https://vdi.vivo.xyz
打开网页内置的谷歌浏览器
https://jus.vivo.xyz/

172.16.35.60
172.16.35.61
172.16.35.62

```
docker commit 容器 镜像
docker save 镜像名[:标签] | gzip > <myimage>.tar.gz
gunzip -c vivo_with_ywz_compressai.tar.gz | docker load

#如果需要拆分-北航云盘同步有大小限制，可以参考如下切分合并
split -b 1024m mydocker.tar.gz "prefixxx."
cat prefixxx* > mydocker.tar.gz

#上传
python setup.py install #bhpan https://github.com/xdedss/dist_bhpan
# bhpan upload [本地文件/文件夹] [远程文件夹(必须不存在,home是自己的根目录)]
bhpan upload -r vivo_with_ywz_compressai_docker home/share/vivo/vivo_docker2

#vivo侧
/usr/local/bin/docker-compose #https://mirrors.aliyun.com/docker-toolbox/linux/compose/1.21.2/docker-compose-Linux-x86_64

#普通文件夹打包
tar -cvvzf ywzvivo.tar.gz ywzvivo
mkdir ywzvivofile
split -b 1024m ywzvivo.tar.gz "ywzvivofile/ywzvivofile."
cat ywzvivofile/ywzvivofile.* > ywzvivo.tar.gz
tar -xvvzf ywzvivo.tar.gz


#vivo侧docker-compose映射
/data01/ywz/ywzvivo
/data01/ccr/media/data/ccr   /OIdataset, model_final.pth, VCM...
```


# 核心文件
```
examples/train.py
compressai/datasets/feature.py
```

# 测试脚本
jjp-930:
```
python3 examples/train.py -m cheng2020-attn -d /media/data/yangwenzhe/Dataset/DIV2K/DIV2K_train_HR/ -d_test /media/data/yangwenzhe/Dataset/div_after_crop/ -q 4 --lambda 0.001 --batch-size 8 -lr 1e-4 --save --cuda --exp exp_cheng_En_03_only_q4 --checkpoint /home/jjp/CompressAI/experiments/exp_cheng_En_01_only_q4/checkpoints/net_checkpoint_best_loss.pth.tar
```

Balle:
python3 examples/train.py -m bmshj2018-hyperprior -d /media/data/yangwenzhe/Dataset/DIV2K/DIV2K_train_HR/ -d_test /media/data/yangwenzhe/Dataset/div_after_crop/ -q 4 --lambda 0.001 --batch-size 8 -lr 1e-4 --save --cuda --exp exp_cheng_En_03_only_q4 --checkpoint /home/jjp/CompressAI/experiments/exp_cheng_En_01_only_q4/checkpoints/net_checkpoint_best_loss.pth.tar

Feature:
python3 examples/train.py -m bmshj2018-hyperprior -d /media/disk2/jjp/ywzVivo/rm_0_channels/ -d_test /media/disk2/jjp/ywzVivo/rm_0_channels/ -q 4 --lambda 0.001 --batch-size 1 -lr 1e-4 --save --cuda  --exp exp_ywz_balle_En_03_only_q4

##change to ywz-docker
Feature:
python3 examples/train.py -m bmshj2018-hyperprior -d /media/data/yangwenzhe/rm_0_channels/ -d_test /media/data/yangwenzhe/rm_0_channels/ -q 4 --lambda 0.001 --batch-size 1 -lr 1e-4 --save --cuda  --exp exp_ywz_balle_En_03_only_q4

#cpu
python3 examples/train.py -m bmshj2018-hyperprior -d /media/data/yangwenzhe/rm_0_channels/ -d_test /media/data/yangwenzhe/rm_0_channels/ -q 4 --lambda 0.001 --batch-size 1 -lr 1e-4 --save  --exp exp_ywz_balle_En_03_only_q4

python3 train_in_this.py
# VCM

#train_vcm_ywz.py
python3 train_vcm_ywz.py -m bmshj2018-hyperprior -d /media/data/yangwenzhe/rm_0_channels/ -d_test /media/data/yangwenzhe/rm_0_channels/ -q 4 --lambda 0.001 --batch-size 1 -lr 1e-4 --save  --exp exp_ywz_balle_En_03_only_q4

#ywz with ccr
python train_vcm.py --config-file /media/data/ccr/VCM/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x_vcm.yaml

python train_vcm.py --config-file /media/data/yangwenzhe/ywzCompressAI/VCM/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x_vcm.yaml