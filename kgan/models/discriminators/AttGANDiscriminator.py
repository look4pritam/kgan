from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


class AttGANDiscriminator(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, input_shape):
        discriminator = models.Sequential(name='discriminator')

        discriminator.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                input_shape=input_shape,
                name='block-1-conv2d'))
        discriminator.add(layers.LeakyReLU(alpha=0.2, name='block-1-lrelu'))

        discriminator.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                name='block-2-conv2d'))
        discriminator.add(layers.LeakyReLU(alpha=0.2, name='block-2-lrelu'))

        discriminator.add(layers.GlobalMaxPooling2D(name='gap'))
        discriminator.add(layers.Flatten(name='flatten'))

        discriminator.add(layers.Dense(units=1, name='prediction'))

        return (discriminator)
