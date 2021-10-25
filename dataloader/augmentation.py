# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:46:20 2021

@author: innat
"""
import tensorflow as tf 
import config 
        
# ref: https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu
# .. modified ..
def cut_mix(batch_of_image, batch_of_label, prob = 1.0):
    if tf.random.uniform([]) < 0.5:
        # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
        # output - a batch of images with mixup applied
        imgs = []; labs = []
        for j in range(batch_size):
            # DO MIXUP WITH PROBABILITY DEFINED ABOVE
            P = tf.cast( tf.random.uniform([],0,1) <= prob, tf.int32)

            # CHOOSE RANDOM
            k = tf.cast( tf.random.uniform([],0, batch_size), tf.int32)
            a = tf.random.uniform([], 0, 1)*tf.cast(P, tf.float32) # this is beta dist with alpha=1.0

            # MAKE MIXUP IMAGE
            img1 = batch_of_image[j,]
            img2 = batch_of_image[k,]
            imgs.append((1-a)*img1 + a*img2)

            if len(batch_of_label.shape) == 1:
                lab1 = tf.one_hot(batch_of_label[j], config.CLASSES)
                lab2 = tf.one_hot(batch_of_label[k], config.CLASSES)
            else:
                lab1 = batch_of_label[j,]
                lab2 = batch_of_label[k,]

            lab1 = tf.cast(lab1, tf.float32)
            lab2 = tf.cast(lab2, tf.float32)

            # MAKE CUTMIX LABEL
            labs.append((1-a)*lab1 + a*lab2)

        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
        mixed_images = tf.reshape(tf.stack(imgs), (config.BATCH_SIZE, config.INP_SIZE[0], config.INP_SIZE[1], 3))
        mixed_labels = tf.reshape(tf.stack(labs), (config.BATCH_SIZE, config.CLASSES))
        return mixed_images, mixed_labels
    
    else:
        # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
        # output - a batch of images with cutmix applied
        imgs = []; labs = []
        for j in range(batch_size):
            # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
            P = tf.cast( tf.random.uniform([],0,1) <= prob, tf.int32)

            # CHOOSE RANDOM IMAGE TO CUTMIX WITH
            k = tf.cast( tf.random.uniform([], 0, batch_size), tf.int32)

            # CHOOSE RANDOM LOCATION
            x = tf.cast( tf.random.uniform([],0, img_size),tf.int32)
            y = tf.cast( tf.random.uniform([],0, img_size),tf.int32)

            b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

            width = tf.cast( img_size * tf.math.sqrt(1-b), tf.int32) * P
            ya = tf.math.maximum(0,   y-width//2)
            yb = tf.math.minimum(img_size, y+width//2)
            xa = tf.math.maximum(0,   x-width//2)
            xb = tf.math.minimum(img_size, x+width//2)

            # MAKE CUTMIX IMAGE
            one    = batch_of_image[j, ya:yb, 0:xa,        :]
            two    = batch_of_image[k, ya:yb, xa:xb,       :]
            three  = batch_of_image[j, ya:yb, xb:img_size, :]
            middle = tf.concat([one, two, three], axis=1)
            img    = tf.concat([batch_of_image[j, 0:ya, :, :],
                                middle,
                                batch_of_image[j, yb:img_size, :, :]], axis=0)
            imgs.append(img)

            # MAKE CUTMIX LABEL
            a = tf.cast(width*width/img_size/img_size, tf.float32)

            if len(batch_of_label.shape) == 1:
                lab1 = tf.one_hot(batch_of_label[j], config.CLASSES)
                lab2 = tf.one_hot(batch_of_label[k], config.CLASSES)
            else:
                lab1 = batch_of_label[j,]
                lab2 = batch_of_label[k,]

            labs.append((1-a)*lab1 + a*lab2)

        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
        cutmixed_imgs  = tf.reshape(tf.stack(imgs), (config.BATCH_SIZE, config.INP_SIZE[0], config.INP_SIZE[1], 3))
        cutmixed_label = tf.reshape(tf.stack(labs), (config.BATCH_SIZE, config.CLASSES))
        return cutmixed_imgs, cutmixed_label   
        