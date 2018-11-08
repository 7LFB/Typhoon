# test for final results
# using 1...n index to access image

from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
import glob
import cv2
import pdb



from model.model import *
from model.average_gradients import *
from utils.colored import *
from utils.tools import *
from utils.params import *


def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h=args.h
    w=args.w

    coord = tf.train.Coordinator()

    img = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    mask = tf.placeholder(tf.float32, shape=[None, h, w, 1])
   
    model = TyphoonModel(args, img, None)

    # Gets moving_mean and moving_variance update operations from
    # COUNT PARAMS
    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print(toCyan('number of trainable parameters: {}'.format(total_num_parameters)))

    # Set up tf session and initialize variables.
    #
    config = tf.ConfigProto(allow_soft_placement=True)  # Chong
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    init_local = tf.local_variables_initializer()
    init = tf.global_variables_initializer()

    # init
    sess.run([init_local, init])

    # Saver for storing checkpoints of the model.
    var = tf.global_variables()
    # fine_tune_var=[val for val in var if ('conv6_cls' not in val.name and 'sub4_out' not in val.name and 'sub24_out' not in val.name )]
    saver = tf.train.Saver(var_list=var, max_to_keep=5)


    print(toGreen('./ckpt/recognition'))
    loader = tf.train.Saver(var_list=var)
    load(loader, sess, './ckpt/recognition')
    

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    files=glob.glob(os.path.join(args.img_dir,'08/*.jpg'))
    txtfile='recognition.txt'
    ff=open(os.path.join(args.img_dir,txtfile),'w')
    for ii in range(len(files)):
        start_time = time.time()
        # if not os.path.exists(os.path.join(args.img_dir,'mask',str(ii+1)+'.jpg')):
        #     continue
        image=read_tri_image_by_index(ii+1,args.img_dir,args.h,args.w)
        feed_dict = {img:image}

        heatmap = sess.run(
            model.mask_pred, feed_dict=feed_dict)
        pred,result=parse_heatmap(heatmap,args.threshold)

        duration = time.time() - start_time
        print('\r',toCyan('{:d}:{:f}'.format(ii+1,duration)),end='')
        ff.write(result)

    ff.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()
