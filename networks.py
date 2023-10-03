import os
import tensorflow as tf


class PreprocessingResNet(tf.keras.layers.Layer):
    
    def call(self, x, training=True):
        std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
        mean= tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))
        if training:
            x = tf.cast(x, tf.float32)/255.
            x = tf.image.random_flip_left_right(x)
            x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
            x = tf.image.random_crop(x, (tf.shape(x)[0], 32, 32, 3))
            x = (x - mean) / std
        else:
            x = tf.cast(x, tf.float32)/255.
            x = (x - mean) / std
        return x


class LeNet(tf.keras.Sequential):

    def __init__(self,
             input_shape=None, 
             last_units=10,
             last_activation=None,
             units=100,
             activation="relu"):
        
        super().__init__()
        self.add(tf.keras.layers.Reshape((28,28,1), input_shape=input_shape))
        self.add(tf.keras.layers.Conv2D(32, 5, activation=None))
        self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.MaxPooling2D(2, 2))
        self.add(tf.keras.layers.Conv2D(48, 5, activation=None))
        self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.MaxPooling2D(2, 2))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units, activation=None))
        self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.Dense(units, activation=None))
        self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.Dense(last_units, activation=last_activation))
        self.add(tf.keras.layers.Activation(last_activation))


class DenseNet(tf.keras.Sequential):
    
    def __init__(self,
                 input_shape=None, 
                 last_units=1,
                 last_activation=None,
                 layers=3,
                 activation="relu",
                 units=100):
    
        super().__init__()
        self.add(tf.keras.layers.Flatten(input_shape=input_shape))
        for _ in range(layers):
            self.add(tf.keras.layers.Dense(units, activation=None))
            self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.Dense(last_units, activation=None))
        self.add(tf.keras.layers.Activation(last_activation))


def shortcut(x, filters, stride):
    if x.shape[-1] == filters:
        return x
    else:
        return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x,
                  paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
    

def basic_block(x, filters, stride=1, regularizer=None):
    y = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same',
                               kernel_regularizer=regularizer,
                               kernel_initializer='he_normal',
                               use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding='same',
                               kernel_regularizer=regularizer,
                               kernel_initializer='he_normal',
                               use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    x = shortcut(x, filters, stride)
    return tf.keras.layers.ReLU()(x + y)


def group_of_blocks(x, num_blocks, filters, stride, regularizer=None, block_idx=0):    
    x = basic_block(x, filters, stride, regularizer)
    for i in range(num_blocks-1):
        x = basic_block(x, filters, 1, regularizer)
    return x


class ResNet32(tf.keras.Model):

    def __init__(self, 
                 input_shape=(32, 32, 3),
                 n_classes=10,
                 l2_reg=1e-4,
                 group_sizes=(5, 5, 5),
                 features=(16, 32, 64),
                 strides=(1, 2, 2)):
        
        regularizer = tf.keras.regularizers.l2(l2_reg)
        
        inputs = tf.keras.layers.Input(input_shape)
        preprocessed_inputs = PreprocessingResNet()(inputs)
        
        x = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same',
                                      kernel_regularizer=regularizer,
                                      kernel_initializer='he_normal',
                                      use_bias=False)(preprocessed_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        for block_idx, (group_size, filters, stride) in enumerate(zip(group_sizes, features, strides)):
            x = group_of_blocks(x,
                                num_blocks=group_size,
                                block_idx=block_idx,
                                filters=filters,
                                stride=stride,
                                regularizer=regularizer)
            
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(n_classes, kernel_regularizer=regularizer)(x)
        
        super().__init__(inputs, x)