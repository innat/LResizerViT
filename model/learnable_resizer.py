
from tensorflow.keras import layers, Model
from config import * 

# ref: https://keras.io/examples/vision/learnable_resizer/
def residual_block(x):
    shortcut = x

    def conv_bn_leaky(inputs, filters, kernel_size, strides):
        x = layers.Conv2D(filters, kernel_size, strides=strides, 
                          use_bias=False, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x 
    
    def conv_bn(inputs, filters, kernel_size, strides):
        x = layers.Conv2D(filters, kernel_size, strides, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        return x 
    
    x = conv_bn_leaky(x, 16, 3, 1)
    x = conv_bn(x, 16, 3, 1)
    x = layers.add([shortcut, x])
    return x


def get_learnable_resizer(filters=16, num_res_blocks=1, interpolation=INTERPOLATION):
    inputs = layers.Input(shape=[None, None, 3])

    # First, perform naive resizing.
    naive_resize = layers.Resizing(
        *TARGET_SIZE, interpolation=interpolation
    )(inputs)

    # First convolution block without batch normalization.
    x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)

    # Second convolution block with batch normalization.
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    # Intermediate resizing as a bottleneck.
    bottleneck = layers.Resizing(
        *TARGET_SIZE, interpolation=interpolation
    )(x)
    
    # Residual passes.
    x = residual_block(bottleneck)
    for i in range(1, num_res_blocks):
        x = residual_block(x)
        
    # Projection.
    x = layers.Conv2D(
        filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # Skip connection.
    x = layers.Add()([bottleneck, x])

    # Final resized image.
    x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    final_resize = layers.Add()([naive_resize, x])
    return Model(inputs, final_resize, name="learnable_resizer")
