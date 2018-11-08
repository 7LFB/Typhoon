
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pdb
import numpy as np 

def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class Dataloader(object):
    """monodepth dataloader"""

    def __init__(self, img_dir,  pos_list, neg_list, batch_size, h,w, data_balance, num_threads):
        self.img_dir = img_dir
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.data_balance=data_balance
        self.target_h=h
        self.target_w=w

        self.img  = None
        self.mask = None
    
        input_queue_pos = tf.train.string_input_producer([self.pos_list], shuffle=False)
        line_reader_pos = tf.TextLineReader()

        input_queue_neg = tf.train.string_input_producer([self.neg_list], shuffle=False)
        line_reader_neg = tf.TextLineReader()

        _, line_pos = line_reader_pos.read(input_queue_pos)
        _, line_neg = line_reader_neg.read(input_queue_neg)
        #pdb.set_trace()

        split_line_pos = tf.string_split([line_pos]).values
        split_line_neg = tf.string_split([line_neg]).values

        seed=tf.random_uniform([],0,1)
        split_line=tf.cond(seed>self.data_balance,lambda:tf.identity(split_line_neg),lambda:tf.identity(split_line_pos))

        image_08_path  = tf.string_join([self.img_dir,'/08/', split_line[0],'.jpg'])
        image_10_path  = tf.string_join([self.img_dir,'/10/', split_line[0],'.jpg'])
        image_14_path  = tf.string_join([self.img_dir,'/14/', split_line[0],'.jpg'])
        
        mask_path=tf.string_join([self.img_dir,'/mask-100/',split_line[0],'.jpg'])
        
        xy=tf.string_to_number(split_line[1])
        y =tf.string_to_number(split_line[2])
        x =tf.string_to_number(split_line[3])

        xx=tf.cast(x/800*self.target_w,tf.int64)
        yy=tf.cast(y/800*self.target_w,tf.int64)


        image_08 = self.read_image(image_08_path)
        image_10 = self.read_image(image_10_path)
        image_14 = self.read_image(image_14_path)
        image=tf.concat([image_08,image_10,image_14],-1)


        mask=tf.cond(xy>0,lambda:self.read_image(mask_path),lambda:tf.zeros([self.target_w,self.target_h,1]),tf.int64)
        # mask=tf.cond(xy>0,lambda:self.construct_gauss_mask(yy,xx,32),lambda:tf.zeros([self.target_w,self.target_h,1]),tf.int64)
        # mask=tf.expand_dims(mask,-1)
        
        do_augment  = tf.random_uniform([], 0, 1)
        image, mask = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(image, mask), lambda: (image, mask))

        image.set_shape( [self.target_h, self.target_w, 3])
        mask.set_shape([self.target_h, self.target_w, 1])

        # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 2048
        capacity = min_after_dequeue + (num_threads+4) * batch_size
        self.img, self.mask = tf.train.shuffle_batch([image, mask],
                    batch_size, capacity, min_after_dequeue, num_threads)

       
    def augment_image_pair(self, image, mask):

        # flip
        flip_lr = tf.random_uniform([], 0, 1)
        image  = tf.cond(flip_lr > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(flip_lr > 0.5, lambda: tf.image.flip_left_right(mask),  lambda: mask)
        
        flip_ud = tf.random_uniform([], 0, 1)
        image  = tf.cond(flip_ud > 0.5, lambda: tf.image.flip_up_down(image), lambda: image)
        mask = tf.cond(flip_ud > 0.5, lambda: tf.image.flip_up_down(mask),  lambda: mask)

        # rotate
        pi=tf.constant(np.pi)
        degree=tf.random_uniform([], 0, 1)*90/pi
        rotate = tf.random_uniform([], 0, 1)
        image  = tf.cond(rotate > 0.5, lambda: tf.contrib.image.rotate(image,degree), lambda: image)
        mask = tf.cond(rotate > 0.5, lambda: tf.contrib.image.rotate(mask,degree),  lambda: mask)
        
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        image_aug  = image  ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        image_aug  =  image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        image_aug  *= color_image

        # saturate
        image_aug  = tf.clip_by_value(image_aug,  0, 1)
    
        return image_aug, mask

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        image  = tf.image.convert_image_dtype(image,  tf.float32) # convert into [0,1] auto
        image  = tf.image.resize_images(image,  [self.target_h, self.target_w], tf.image.ResizeMethod.AREA)

        return image

    def construct_pixel_mask(self,crows,ccols):

        mask=tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[crows, ccols]], values=[1.0], dense_shape=[self.target_w, self.target_h]))

        return tf.expand_dims(mask,-1)

    def construct_gauss_mask(self,crows,ccols,sigma):
        temp=tf.linspace(1.,self.target_w,self.target_w)
        cols=tf.expand_dims(temp,0)
        rows=tf.expand_dims(temp,-1)

        w=tf.tile(cols,[1,self.target_w])-tf.cast(crows,tf.float32)
        h=tf.tile(rows,[self.target_h,1])-tf.cast(ccols,tf.float32)
        D=tf.square(w)+tf.square(h)
        E=0.5*tf.square(1./sigma)
        Exp=D*E
        mask=tf.exp(-Exp)
        mask=tf.expand_dims(mask,-1)

        return mask





