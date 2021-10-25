# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:41:27 2021

@author: innat
"""

import pathlib
import config 
import tensorflow as tf 
from tensorflow.keras import layers 

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print('Total Samples: ', image_count)


class_number = 5
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=config.INP_SIZE,
    batch_size=config.BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=config.INP_SIZE,
    batch_size=config.BATCH_SIZE
)


from augmentation import cut_mix
# for train set : augmentation 
keras_aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(.2, .3),
        layers.RandomRotation((0.2, 0.3), fill_mode="reflect")
    ]
)

train_ds = train_ds.shuffle(10 * config.BATCH_SIZE)
train_ds = train_ds.map(lambda x, y: (keras_aug(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(cut_mix, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)


def k_hot(x, y): 
    return x, tf.one_hot(y, class_number)

val_ds = val_ds.map(k_hot)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        
        
        
        
        
        
        
        
        
        
        