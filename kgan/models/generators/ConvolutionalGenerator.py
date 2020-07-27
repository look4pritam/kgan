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
        generator_shape = (7, 7, 128)
        generator_size = np.prod(generator_shape)

        generator = models.Sequential(name='generator')

        generator.add(
            layers.Dense(
                units=generator_size,
                input_shape=(latent_dimension, ),
                name='block-1-dense'))
        generator.add(layers.LeakyReLU(alpha=0.2, name='block-1-lrelu'))

        generator.add(
            layers.Reshape(generator_shape, name='block-1-2-reshape'))

        generator.add(
            layers.Conv2DTranspose(
                filters=128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name='block-2-deconv2d'))
        generator.add(layers.LeakyReLU(alpha=0.2, name='block-2-lrelu'))

        generator.add(
            layers.Conv2DTranspose(
                filters=128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                name='block-3-deconv2d'))
        generator.add(layers.LeakyReLU(alpha=0.2, name='block-3-lrelu'))

        generator.add(
            layers.Conv2D(
                filters=1,
                kernel_size=(7, 7),
                padding='same',
                activation='sigmoid',
                name='fake-image'))

        return (generator)
