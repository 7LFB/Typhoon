# using data balance 

import imgaug as ia
from imgaug import augmenters as iaa
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
from skimage import data
import sklearn.utils
import cv2
import threading
import random
import glob
import sys
import os
import pdb


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

class BatchGenerator:
    def __init__(self,img_dir, pos_list, neg_list, batch_size,target_h,target_w,data_balance=0.5):
         # Set class members.

        self.batch_size=batch_size
        self.data_balance=data_balance
        self.pos_step=int(self.batch_size*self.data_balance)
        self.neg_step=int(self.batch_size-self.pos_step)
        self.img_dir = img_dir

        self.target_h=target_h
        self.target_w=target_w

        self.pos_list=open(pos_list).readlines()
        self.neg_list=open(neg_list).readlines()

    # A generator that loads batches from the hard drive.
    def __call__(self):

        current1=0
        current2=0

        while True:
            batch_img, batch_mask=[], []
            if current1 >= len(self.pos_list) or current1+self.pos_step>len(self.pos_list):
                current1 = 0
                self.pos_list= sklearn.utils.shuffle(self.pos_list)

            if current2 >= len(self.neg_list) or current2+self.neg_step>len(self.neg_list):
                current2 = 0
                self.neg_list= sklearn.utils.shuffle(self.neg_list)

            # Get the image filepaths for this batch.
            filenames_pos = self.pos_list[current1:current1+self.pos_step]
            filenames_neg = self.neg_list[current2:current2+self.neg_step]
            filenames=filenames_pos+filenames_neg

            current1 += self.pos_step
            current2 += self.neg_step


            # Load the images for this batch.
            for file in filenames:
                file=file.strip()
                img,mask=self.read_img_and_mask(file)

                batch_img.append(img)
                batch_mask.append(mask)
            
            yield np.array(batch_img)/255, np.array(batch_mask)/255
    
   
    def augmentNumpy(self, imgs, masks):

        augseq = iaa.Sequential([
            sometimes(iaa.Fliplr(0.5)),
            sometimes(iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-1, 1), # rotate by -15 to +15 degrees
            # shear=(-5, 5)# shear by -5 to +5 degrees
            # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            ],random_order=False)

        seq_imgs_deterministic = augseq.to_deterministic()
                
        imgs_aug = seq_imgs_deterministic.augment_images(imgs)

        return imgs_aug


    def read_img_and_mask(self,path):
        ss=path.split(' ')
        p1=ss[0]
        p2=int(ss[1])
        img1=cv2.imread(os.path.join(self.img_dir,'08',p1+'.jpg'),0)
        img2=cv2.imread(os.path.join(self.img_dir,'10',p1+'.jpg'),0)
        img3=cv2.imread(os.path.join(self.img_dir,'14',p1+'.jpg'),0)
        img=np.concatenate([np.expand_dims(img1,-1),np.expand_dims(img2,-1),np.expand_dims(img3,-1)],-1)
        # pdb.set_trace()
        if p2==1:
            mask=cv2.imread(os.path.join(self.img_dir,'mask',p1+'.jpg'),0)
            mask=cv2.resize(mask,(self.target_h,self.target_w))
        else:
            mask=np.zeros((self.target_h,self.target_w))
        img=cv2.resize(img,(self.target_h,self.target_w))
        mask=np.expand_dims(mask,-1)
        
    
        return img, mask
        