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

    def _update_discriminator(self, input_batch):
        real_samples = input_batch
        generator_inputs = tf.random.normal(
            [self.batch_size(), self.latent_dimension()])

        self._generator.trainable = False
        self._discriminator.trainable = True
        with tf.GradientTape(persistent=True) as tape:
            fake_samples = self._generator(generator_inputs, training=False)

            real_predictions = self._discriminator(real_samples, training=True)
            fake_predictions = self._discriminator(fake_samples, training=True)

            discriminator_loss = self._discriminator_loss(
                real_predictions, fake_predictions)

        discriminator_loss_gradients = tape.gradient(
            target=discriminator_loss,
            sources=self._discriminator.trainable_variables)

        self._discriminator_optimizer.apply_gradients(
            zip(discriminator_loss_gradients,
                self._discriminator.trainable_variables))

        return (discriminator_loss)

    def _update_generator(self, input_batch):
        generator_inputs = tf.random.normal(
            [self.batch_size() * 2,
             self.latent_dimension()])

        self._generator.trainable = True
        self._discriminator.trainable = False
        with tf.GradientTape(persistent=True) as tape:
            fake_samples = self._generator(generator_inputs, training=True)
            fake_predictions = self._discriminator(
                fake_samples, training=False)
            generator_loss = self._generator_loss(fake_predictions)

        generator_loss_gradients = tape.gradient(
            target=generator_loss, sources=self._generator.trainable_variables)

        self._generator_optimizer.apply_gradients(
            zip(generator_loss_gradients, self._generator.trainable_variables))

        return (generator_loss)

    def _train_on_batch(self, input_batch):
        discriminator_loss = self._update_discriminator(input_batch)
        generator_loss = self._update_generator(input_batch)

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }

    '''
    def _train_on_batch(self, input_batch):
        real_samples = input_batch
        generator_inputs = tf.random.normal(
            [self.batch_size(), self.latent_dimension()])

        with tf.GradientTape(persistent=True) as tape:
            fake_samples = self._generator(generator_inputs, training=True)

            real_predictions = self._discriminator(real_samples, training=True)
            fake_predictions = self._discriminator(fake_samples, training=True)

            generator_loss = self._generator_loss(fake_predictions)

            discriminator_loss = self._discriminator_loss(
                real_predictions, fake_predictions)

        generator_loss_gradients = tape.gradient(
            target=generator_loss, sources=self._generator.trainable_variables)
        discriminator_loss_gradients = tape.gradient(
            target=discriminator_loss,
            sources=self._discriminator.trainable_variables)

        self._generator_optimizer.apply_gradients(
            zip(generator_loss_gradients, self._generator.trainable_variables))
        self._discriminator_optimizer.apply_gradients(
            zip(discriminator_loss_gradients,
                self._discriminator.trainable_variables))

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }
    '''
