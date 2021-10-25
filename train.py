import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras import Input, Model, Sequential, layers
import tensorflow_hub as hub

from model.learnable_resizer import get_learnable_resizer
learnable_resizer = get_learnable_resizer(num_res_blocks=config.num_res_blocks_in_trainable_resizer)
learnable_resizer.summary()


from dataloader.data import *

tcls_names, vcls_names = train_ds.class_names , val_ds.class_names
for images, labels in train_ds.take(5):
    print(images.shape, labels.shape)
    plt.figure(figsize=(20, 20))
    for i in range(8):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy())
        plt.axis("off")
    plt.show()
    
plt.figure(figsize=(20, 20))
for images, labels in train_ds.take(1):
    print(images.shape, labels.shape)
    for i in range(8):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(tcls_names[labels[i]])
        plt.axis("off")
        
        
handle="https://tfhub.dev/sayakpaul/vit_s16_fe/1"
def get_model(plot_modal, print_summary, with_compile):
    hub_layer = hub.KerasLayer(handle, trainable=True)
    backbone = Sequential(
        [
            layers.InputLayer((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
            hub_layer
        ], name='vit'
    )
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)
    
    tail = Sequential(
            [
                layers.Dropout(0.2),
                layers.BatchNormalization(),
                layers.Dense(1, activation = 'sigmoid')
            ], name='head'
        )
    
    model = Model(inputs, tail(outputs))
    
    if plot_modal:
        display(tf.keras.utils.plot_model(model, show_shapes=True, 
                                          show_layer_names=True,  expand_nested=True))
    if print_summary:
        print(model.summary())
        
    if with_compile:
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr),  
            loss = tf.keras.losses.BinaryCrossentropy(), 
            metrics = [tf.keras.metrics.RootMeanSquaredError('rmse')])  
    return model 


get_model(plot_modal=True, print_summary=True, with_compile=False)



from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras import callbacks


from tensorflow.keras import losses, optimizers , metrics
from tensorflow.keras import callbacks

def get_lr_callback(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr
    return callbacks.LearningRateScheduler(lrfn, verbose=True)


rlr = callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.3, patience=2)
# rlr =  get_lr_callback(batch_size)


# compile and run 
model.compile(
    loss=losses.CategoricalCrossentropy(from_logits=True),
    optimizer=optimizers.Adam(), 
    metrics=['accuracy']
)
history = model.fit(train_ds, 
                    epochs=epochs,
                    callbacks=[rlr], 
                    validation_data = val_ds, verbose=2)


acc     = history.history['accuracy']
val_acc = history.history['val_accuracy'] 

loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()