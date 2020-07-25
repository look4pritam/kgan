from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

import numpy as np


class ConvolutionalGenerator(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, latent_dimension):
        generator_shape = (7, 7, 256)
        generator_size = np.prod(generator_shape)

        generator = models.Sequential(name='generator')

        generator.add(
            layers.Dense(
                units=generator_size,
                use_bias=False,
                input_shape=(latent_dimension, )))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())

        generator.add(layers.Reshape(generator_shape))

        generator.add(
            layers.Conv2DTranspose(
                filters=128,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding='same',
                use_bias=False))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())

        generator.add(
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())

        generator.add(
            layers.Conv2DTranspose(
                filters=1,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                activation='tanh'))

        return (generator)
