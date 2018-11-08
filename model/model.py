#
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from .ops import *


class TyphoonModel(object):

    def __init__(self, params, img, mask):
        self.params = params
        self.img=img
        self.mask=mask
        self.current_batch_size=int(self.params.batch_size/self.params.num_gpus)

        self.build_model()
        self.build_outputs()
  
    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def get_img(self, x, c=1):
        img = conv_(x, c, 3, 1, tf.nn.sigmoid) # output channel = 6, means two image
        # img.set_shape([self.params.batch_size,-1,-1,c])
        return img

    def build_vgg(self):
        #set convenience functions
        conv=conv_
        if self.params.use_deconv:
            upconv = deconv_
        else:
            upconv = upconv_

        with tf.variable_scope('encoder'):
            conv1 = conv_block(self.model_input,  32, 7) # H/2
            conv2 = conv_block(conv1,             64, 5) # H/4
            conv3 = conv_block(conv2,            128, 3) # H/8
            conv4 = conv_block(conv3,            256, 3) # H/16
            conv5 = conv_block(conv4,            512, 3) # H/32
            conv6 = conv_block(conv5,            512, 3) # H/64
            conv7 = conv_block(conv6,            512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            self.img4 = self.get_img(iconv4)
            uimg4  = upsample_nn(self.img4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, uimg4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.img3 = self.get_img(iconv3)
            uimg3  = upsample_nn(self.img3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, uimg3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            self.img2 = self.get_img(iconv2)
            uimg2  = upsample_nn(self.img2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, uimg2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.img_pred = self.get_img(iconv1)

    def build_resnet50(self):
        #set convenience functions
        conv=conv_
        if self.params.use_deconv:
            upconv = deconv_
        else:
            upconv = upconv_

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            self.img4 = self.get_img(iconv4)
            uimg4  = upsample_nn(self.img4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, uimg4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            self.img3 = self.get_img(iconv3)
            uimg3  = upsample_nn(self.img3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, uimg3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            self.img2 = self.get_img(iconv2)
            uimg2  = upsample_nn(self.img2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, uimg2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.img_pred = self.get_img(iconv1)
    
    def build_unet(self):

        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(self.model_input)
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = layers.Cropping2D(cropping=(ch,cw))(conv4)
        up6 = layers.concatenate([up_conv5, crop_conv4], axis=3)
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = layers.Cropping2D(cropping=(ch,cw))(conv3)
        up7 = layers.concatenate([up_conv6, crop_conv3], axis=3) 
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = layers.Cropping2D(cropping=(ch,cw))(conv2)
        up8 = layers.concatenate([up_conv7, crop_conv2], axis=3)
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = layers.Cropping2D(cropping=(ch,cw))(conv1)
        up9 = layers.concatenate([up_conv8, crop_conv1], axis=3)
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = self.get_crop_shape(self.model_input, conv9)
        conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = layers.Conv2D(1, (3, 3),activation='sigmoid', padding='same')(conv9)

        self.img_pred=conv10


    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model'):
               
                self.model_input=self.img

                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                elif self.params.encoder == 'unet':
                    self.build_unet()
                else:
                    return None     

    def build_outputs(self):
        self.mask_pred=self.img_pred


    def build_pixel_losses(self):
        with tf.variable_scope('pixel_losses'):
            # IMAGE RECONSTRUCTION
            pred= self.img_pred
            gt  = self.mask
            # L1
            # l1_reconstruction_loss=tf.reduce_mean(tf.abs(pred-gt))
            l2_reconstruction_loss=tf.reduce_mean(tf.square(pred-gt))
            # l2_reconstruction_loss=tf.reduce_sum(tf.reduce_mean(tf.square(pred-gt),0))
            
            pixel_loss = l2_reconstruction_loss 

        return pixel_loss

    def build_PSCE_losses(self):
        pred=tf.clip_by_value(self.img_pred,1e-9,1-(1e-9))
        gt = self.mask
        loss_=-gt*tf.log(pred)-(1-gt)*tf.log((1-pred))
        # loss=tf.reduce_sum(tf.reduce_mean(loss_,0))
        loss=tf.reduce_mean(loss_)
        return loss

    
    def build_losses(self,alpha=0):
        with tf.variable_scope('losses'):
            self.pixel_loss=self.build_pixel_losses()
            self.psce_loss=self.build_PSCE_losses()

        # self.loss=self.pixel_loss
        self.loss=self.pixel_loss + alpha* self.psce_loss

    def build_summary(self,train_val):
        s1=tf.summary.image(train_val+'/img', self.img * 255, 4)
        s2=tf.summary.image(train_val+'/gt', self.mask * 255, 4)
        s3=tf.summary.image(train_val+'/pred', self.mask_pred * 255, 4)
        return [s1,s2,s3]
