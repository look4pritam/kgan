from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.SimpleDiscriminator import SimpleDiscriminator
from kgan.models.SimpleGenerator import SimpleGenerator

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from tensorflow.keras.optimizers import Adam

import numpy as np


class SimpleGAN(object):
    @classmethod
    def name(cls):
        return ('gan')

    def _create_optimizer(learning_rate=0.0002):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
        return (optimizer)

    def __init__(self, input_shape, latent_shape):
        self._input_shape = input_shape
        self._latent_shape = latent_shape

        self._discriminator = self._create_discriminator(self.input_shape())
        self._generator = self._create_generator(self.input_shape(),
                                                 self.latent_shape())

    def _create_discriminator(self, input_shape):
        discriminator = SimpleDiscriminator.create(input_shape)
        return (discriminator)

    def _create_generator(self, input_shape, latent_shape):
        generator = SimpleGenerator.create(input_shape, latent_shape)
        return (generator)

    def train_on_batch(self, input_batch):
        pass

    def input_shape(self):
        return (self._input_shape)

    def latent_shape(self):
        return (self._latent_shape)
