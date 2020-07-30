from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.GAN import GAN

from kgan.models.discriminators.ConvolutionalDiscriminator import ConvolutionalDiscriminator
from kgan.models.generators.ConvolutionalGenerator import ConvolutionalGenerator

import tensorflow as tf


class DCGAN(GAN):
    @classmethod
    def name(cls):
        return ('dcgan')

    def __init__(self, input_shape, latent_dimension):
        super(DCGAN, self).__init__(input_shape, latent_dimension)
        pass

    def _create_discriminator(self):
        self._discriminator = ConvolutionalDiscriminator.create(
            self.input_shape())
        return (True)

    def _create_generator(self):
        generator = ConvolutionalGenerator.create(self.latent_dimension())
        return (generator)
