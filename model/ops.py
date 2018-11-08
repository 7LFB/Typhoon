import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def upsample_nn(x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = tf.shape(img)
    h = s[1]
    w = s[2]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
    return scaled_imgs

def conv_(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

def conv_block(x, num_out_layers, kernel_size):
    conv1 = conv_(x,     num_out_layers, kernel_size, 1)
    conv2 = conv_(conv1, num_out_layers, kernel_size, 2)
    return conv2

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def resconv(x, num_layers, stride):
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv_(x,         num_layers, 1, 1)
    conv2 = conv_(conv1,     num_layers, 3, stride)
    conv3 = conv_(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv_(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.elu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

def upconv_(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    conv = conv_(upsample, num_out_layers, kernel_size, 1)
    return conv

def deconv_(x, num_out_layers, kernel_size, scale):
    p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
    return conv[:,3:-1,3:-1,:]

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)





