from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.ImageGAN import ImageGAN

from kgan.models.discriminators.SimpleDiscriminator import SimpleDiscriminator
from kgan.models.generators.SimpleGenerator import SimpleGenerator

import tensorflow as tf


class SimpleGAN(ImageGAN):
    @classmethod
    def name(cls):
        return ('gan')

    def __init__(self, input_shape, latent_dimension):
        super(SimpleGAN, self).__init__(input_shape, latent_dimension)
        pass

    def _create_discriminator(self):
        discriminator = SimpleDiscriminator.create(self.input_shape())
        return (discriminator)

    def _create_generator(self):
        generator = SimpleGenerator.create(self.input_shape(),
                                           self.latent_dimension())
        return (generator)

    def _create_generator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
        return (optimizer)

    def _train_on_batch(self, input_batch):
        real_samples = input_batch
        generator_inputs = tf.random.normal(
            [self.batch_size(), self.latent_dimension()])

        with tf.GradientTape() as generator_tape, tf.GradientTape(
        ) as discriminator_tape:
            fake_samples = self._generator(generator_inputs, training=True)

            real_predictions = self._discriminator(real_samples, training=True)
            fake_predictions = self._discriminator(fake_samples, training=True)

            generator_loss = self._generator_loss(fake_predictions)
            discriminator_loss = self._discriminator_loss(
                real_predictions, fake_predictions)

        gradients_of_generator = generator_tape.gradient(
            target=generator_loss, sources=self._generator.trainable_variables)
        gradients_of_discriminator = discriminator_tape.gradient(
            target=discriminator_loss,
            sources=self._discriminator.trainable_variables)

        self._generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self._generator.trainable_variables))
        self._discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator,
                self._discriminator.trainable_variables))

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }
