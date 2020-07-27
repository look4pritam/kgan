from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


class DenseDiscriminator(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, input_shape):
        discriminator = models.Sequential(name='discriminator')

        discriminator.add(
            layers.Flatten(input_shape=input_shape, name='block-1-flatten'))

        discriminator.add(layers.Dense(units=1024, name='block-2-dense'))
        discriminator.add(layers.LeakyReLU(alpha=0.2, name='block-2-lrelu'))

        discriminator.add(layers.Dense(units=512, name='block-3-dense'))
        discriminator.add(layers.LeakyReLU(alpha=0.2, name='block-3-lrelu'))

        discriminator.add(layers.Dense(units=256, name='block-4-dense'))
        discriminator.add(layers.LeakyReLU(alpha=0.2, name='block-4-lrelu'))

        discriminator.add(
            layers.Dense(units=1, activation='sigmoid', name='prediction'))
        return (discriminator)
