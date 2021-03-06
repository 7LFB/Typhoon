# using BatchGenerator for loading data


from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np


from model.model import *
from model.average_gradients import *
from data.myBatchGenerator import *
from utils.colored import *
from utils.tools import *
from utils.params import *


def main():
    """Create the model and start the training."""
    args = get_arguments()
    args.snapshot_dir=args.snapshot_dir.replace('Typhoon/','Typhoon/'+args.model_name+'-')
    print(toMagenta(args.snapshot_dir))
    start_steps = args.start_steps

    h, w = map(int, args.input_size.split(','))
    args.h=h
    args.w=w

    # construct data generator
    t1 = open(args.train_pos_list)
    t2 = open(args.train_neg_list)
    train_num_images = len(t1.readlines()) + len(t2.readlines())

    v1 = open(args.val_pos_list)
    v2 = open(args.val_neg_list)
    val_num_images = len(v1.readlines()) + len(v2.readlines())
    
    t1.close()
    t2.close()
    v1.close()
    v2.close()

    steps_per_epoch = int((train_num_images / args.batch_size))
    num_steps = int(steps_per_epoch * args.num_epochs)
    val_num_steps = int(val_num_images / args.batch_size)

    print(toCyan('train images: {:d}, test images {:d}'.format(
        train_num_images, val_num_images)))
    print(toCyan('steps_per_epoch x num_epochs:{:d} x {:d}'.format(
        steps_per_epoch, args.num_epochs)))

    myTrainBatchGenerator=BatchGenerator(args.img_dir+'train',args.train_pos_list,args.train_neg_list,args.batch_size,h,w,args.data_balance)()
    myValBatchGenerator=BatchGenerator(args.img_dir+'verification',args.val_pos_list,args.val_neg_list,args.batch_size,h,w,0.8)()
    
    coord = tf.train.Coordinator()


    img = tf.placeholder(tf.float32, shape=[args.batch_size, h, w, 3])
    mask = tf.placeholder(tf.float32, shape=[args.batch_size, h, w, 1])
    
    img_splits = tf.split(img, args.num_gpus, 0)
    mask_splits = tf.split(mask, args.num_gpus, 0)
   
    # Using Poly learning rate policy
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow(
        (1 - step_ph / num_steps), args.power))
    opt_step = tf.train.MomentumOptimizer(learning_rate, args.momentum)

    # construct model
    tower_grads = []
    tower_losses = []

    for i in range(args.num_gpus):
       with tf.device('/gpu:%d' % i):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
               model = TyphoonModel(args, img_splits[i], mask_splits[i])
               model.build_losses(args.loss_balance)
            if i == 0:
                train_summary = model.build_summary('train')
                val_summary = model.build_summary('val')
            loss_ = model.loss
            all_trainable =[v for v in tf.trainable_variables() if 'losses' not in v.name]

            tower_losses.append(loss_)

            grads=opt_step.compute_gradients(loss_,all_trainable)

            tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    loss=tf.reduce_mean(tower_losses)


    # Gets moving_mean and moving_variance update operations from
    # tf.GraphKeys.UPDATE_OPS
    if args.no_update_mean_var == True:
        update_ops = None
    else:
        print(toMagenta('updating mean and var in batchnorm'))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = opt_step.apply_gradients(grads)

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

    # construct summary
    train_summary.append(tf.summary.scalar(
        'train/learning_rate', learning_rate))
    train_summary.append(tf.summary.scalar('train/loss', loss))


    train_merged = tf.summary.merge(train_summary)
    val_merged = tf.summary.merge(val_summary)
    FinalSummary = tf.summary.FileWriter(args.snapshot_dir, sess.graph)

    # init
    sess.run([init_local, init])

    # Saver for storing checkpoints of the model.
    var = tf.global_variables()
    # fine_tune_var=[val for val in var if ('conv6_cls' not in val.name and 'sub4_out' not in val.name and 'sub24_out' not in val.name )]
    saver = tf.train.Saver(var_list=var, max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path and args.resume:
        loader = tf.train.Saver(var_list=var)
        load_step = int(os.path.basename(
            ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    elif args.load_pretrained:
        print(toRed('Restore from pre-trained model...'))
        net.load_for_fine_tune(args.restore_from, sess,
                               fine_tune_var)  # Chong:0531

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    
    # Iterate over training steps.
    loss_history = 10000
    for step in range(start_steps, num_steps):
        img1, img2 = next(myTrainBatchGenerator)
        start_time = time.time()
        feed_dict = {img:img1,mask:img2,
                        step_ph: step}

        summary, total_loss, _ = sess.run(
            [train_merged, loss, train_op], feed_dict=feed_dict)
        FinalSummary.add_summary(summary, step)
        duration = time.time() - start_time
        print('\r', toCyan('{:s}:{:d}-{:d}-{:d} total loss = {:.3f},({:.3f} sec/step)'.format(args.model_name,step %
                                                                                         steps_per_epoch, step // steps_per_epoch, args.num_epochs, total_loss, duration)), end='')

        if step % args.test_every == 0:
            losses = []
            for jj in range(val_num_steps):
                img1, img2 = next(myValBatchGenerator)
                feed_dict = {img:img1,mask:img2}
                summary, total_loss = sess.run(
                    [val_merged, loss], feed_dict=feed_dict)
                losses.append(total_loss)
            FinalSummary.add_summary(summary, step)
            losses = np.array(losses)
            loss_ = np.mean(losses)

            test_summary = tf.Summary()
            test_summary.value.add(tag='val/loss', simple_value=loss_)
            FinalSummary.add_summary(test_summary, step)

            if loss_ < loss_history:
                save(saver, sess, args.snapshot_dir, step)
                loss_history = loss_

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
