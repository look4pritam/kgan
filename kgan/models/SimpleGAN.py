from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.SimpleDiscriminator import SimpleDiscriminator

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from tensorflow.keras.optimizers import Adam

import numpy as np


class SimpleGAN(object):
    @classmethod
    def name(cls):
        return ('gan')

    @classmethod
    def create_discriminator(cls, input_shape):
        discriminator = SimpleDiscriminator.create(input_shape)
        return (discriminator)

    @classmethod
    def create_generator(cls, input_shape, latent_shape):
        generator = models.Sequential(name='generator')

        generator.add(layers.Dense(units=256, input_shape=latent_shape))
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

    def create_optimizer(learning_rate=0.0002):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
        return (optimizer)

    def __init__(self, input_shape, latent_shape):
        self._input_shape = input_shape
        self._latent_shape = latent_shape

        self._discriminator = SimpleGAN.create_discriminator(
            self.input_shape())
        self._generator = SimpleGAN.create_generator(self.input_shape(),
                                                     self.latent_shape())

    def train_on_batch(self, input_batch):
        pass

    def input_shape(self):
        return (self._input_shape)

    def latent_shape(self):
        return (self._latent_shape)
