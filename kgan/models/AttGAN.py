from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.WGANGP import WGANGP

from kgan.models.discriminators.AttGANDiscriminator import AttGANDiscriminator
from kgan.models.generators.ConvolutionalGenerator import ConvolutionalGenerator

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class AttGAN(WGANGP):
    @classmethod
    def name(cls):
        return ('attgan')

    def __init__(self, input_shape, latent_dimension):
        super(AttGAN, self).__init__(input_shape, latent_dimension)
        pass

    def _create_discriminator(self):
        self._discriminator = AttGANDiscriminator.create(self.input_shape())
        return (True)

    def _create_generator(self):
        self._generator = ConvolutionalGenerator.create(
            self.latent_dimension())
        return (True)

    def _normalize_dataset(self, image, attributes):
        return (image, attributes)

    def _update_learning_rate(self, current_epoch, number_of_epochs):
        return (True)

    def _create_generator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
        return (optimizer)

    def _train_on_batch(self, input_batch):

        if ((self.current_step() % self.cycle_number()) == 0):
            # Update generator weights.
            generator_loss = self._update_generator(input_batch)
            return {'generator': generator_loss}
        else:
            # Update discriminator weights.
            discriminator_loss = self._update_discriminator(input_batch)
            return {'discriminator': discriminator_loss}
