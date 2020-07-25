from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

import numpy as np


class SimpleGenerator(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, input_shape, latent_dimension):
        generator = models.Sequential(name='generator')

        generator.add(
            layers.Dense(units=256, input_shape=(latent_dimension, )))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.BatchNormalization(momentum=0.8))

        generator.add(layers.Dense(units=512))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.BatchNormalization(momentum=0.8))

        generator.add(layers.Dense(units=1024))
        generator.add(layers.LeakyReLU(alpha=0.2))
        generator.add(layers.BatchNormalization(momentum=0.8))

        input_size = np.prod(input_shape)
        generator.add(layers.Dense(input_size, activation='tanh'))
        generator.add(layers.Reshape(input_shape))

        return (generator)
