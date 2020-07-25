from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.ImageGAN import ImageGAN

from kgan.models.discriminators.ConvolutionalDiscriminator import ConvolutionalDiscriminator
from kgan.models.generators.ConvolutionalGenerator import ConvolutionalGenerator

import tensorflow as tf


class DCGAN(ImageGAN):
    @classmethod
    def name(cls):
        return ('dcgan')

    def __init__(self, input_shape, latent_dimension):
        super(DCGAN, self).__init__()

        self._discriminator = self._create_discriminator()
        self._generator = self._create_generator()

    def _create_discriminator(self):
        discriminator = ConvolutionalDiscriminator.create(self.input_shape())
        return (discriminator)

    def _create_generator(self):
        generator = ConvolutionalGenerator.create(self.latent_dimension())
        return (generator)

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
