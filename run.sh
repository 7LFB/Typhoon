#! /bin/bash

echo '---recognition---'
CUDA_VISIBLE_DEVICES=0 python recognition.py --img_dir /home/comp/chongyin/DataSets/Typhoon/centerLocation/test
echo '---location---'
CUDA_VISIBLE_DEVICES=0 python location.py --img_dir /home/comp/chongyin/DataSets/Typhoon/centerLocation/test
