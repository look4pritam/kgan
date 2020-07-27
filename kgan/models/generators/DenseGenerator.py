from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

import numpy as np


class DenseGenerator(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, input_shape, latent_dimension):
        generator = models.Sequential(name='generator')

        generator.add(
            layers.Dense(
                units=256,
                input_shape=(latent_dimension, ),
                name='block-1-dense'))
        generator.add(layers.LeakyReLU(alpha=0.2, name='block-1-lrelu'))
        generator.add(
            layers.BatchNormalization(momentum=0.8, name='block-1-bn'))

        generator.add(layers.Dense(units=512, name='block-2-dense'))
        generator.add(layers.LeakyReLU(alpha=0.2, name='block-2-lrelu'))
        generator.add(
            layers.BatchNormalization(momentum=0.8, name='block-2-bn'))

        generator.add(layers.Dense(units=1024, name='block-3-dense'))
        generator.add(layers.LeakyReLU(alpha=0.2, name='block-3-lrelu'))
        generator.add(
            layers.BatchNormalization(momentum=0.8, name='block-3-bn'))

        input_size = np.prod(input_shape)
        generator.add(
            layers.Dense(input_size, activation='tanh', name='block-4-dense'))
        generator.add(layers.Reshape(input_shape, name='fake-image'))

        return (generator)
