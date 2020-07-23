from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


class ConvolutionalDiscriminator(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, input_shape):
        discriminator = models.Sequential(name='discriminator')

        discriminator.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                input_shape=input_shape))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))

        discriminator.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same'))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))

        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(units=1))

        return (discriminator)
