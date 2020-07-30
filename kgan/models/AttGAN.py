from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.WGANGP import WGANGP

from kgan.models.discriminators.AttGANDiscriminator import AttGANDiscriminator

from kgan.models.generators.AttGANEncoder import AttGANEncoder
from kgan.models.generators.AttGANDecoder import AttGANDecoder

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class AttGAN(WGANGP):
    @classmethod
    def name(cls):
        return ('attgan')

    def __init__(self, input_shape, latent_dimension):
        super(AttGAN, self).__init__(input_shape, latent_dimension)
        self._encoder = None
        self._decoder = None

    def _create_discriminator(self):
        self._discriminator = AttGANDiscriminator.create(self.input_shape())
        self._discriminator.summary()
        return (True)

    def _create_generator(self):
        self._generator = None

        self._encoder = AttGANEncoder.create()
        self._decoder = AttGANDecoder.create()

        input_images = np.zeros((input_shape))
        input_images = np.expand_dims(input_images, axis=0)

        image_attributes = np.zero(self.latent_dimension())
        image_attributes = np.expand_dims(image_attributes, axis=0)

        image_features = self._encoder.predict(input_images)
        generated_images = self._decoder.predict(
            [image_features, image_attributes])

        self._encoder.summary()
        self._decoder.summary()
        return (True)

    def _normalize_dataset(self, image, attributes):
        return (image, attributes)

    def _update_learning_rate(self, current_epoch, number_of_epochs):
        start_decay_epoch = number_of_epochs // 2
        if (current_epoch >= start_decay_epoch):
            self._learning_rate = self.base_learning_rate() * (
                1 - 1 / (number_of_epochs - start_decay_epoch + 1) *
                (current_epoch - start_decay_epoch + 1))
        else:
            self._learning_rate = self.base_learning_rate()

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
