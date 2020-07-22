from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def create_discriminator(input_shape):
    discriminator = models.Sequential(name='discriminator')

    discriminator.add(layers.Flatten(input_shape=input_shape))

    discriminator.add(layers.Dense(units=1024))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Dense(units=512))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Dense(units=256))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Dense(units=1, activation='sigmoid'))
    return (discriminator)
