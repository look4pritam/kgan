from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.AbstractGAN import AbstractGAN

from kgan.models.discriminators.SimpleDiscriminator import SimpleDiscriminator
from kgan.models.generators.SimpleGenerator import SimpleGenerator

import tensorflow as tf

import cv2

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class SimpleGAN(AbstractGAN):
    @classmethod
    def name(cls):
        return ('gan')

    def __init__(self, input_shape, latent_dimension):
        super(SimpleGAN, self).__init__()

        self._input_shape = input_shape
        self._latent_dimension = latent_dimension

        self._discriminator = self._create_discriminator()
        self._generator = self._create_generator()

    def _create_discriminator(self):
        discriminator = SimpleDiscriminator.create(self.input_shape())
        return (discriminator)

    def _create_generator(self):
        generator = SimpleGenerator.create(self.input_shape(),
                                           self.latent_dimension())
        return (generator)

    def input_shape(self):
        return (self._input_shape)

    def latent_dimension(self):
        return (self._latent_dimension)

    def _generator_loss(self, fake_predictions):
        generator_loss = cross_entropy(
            tf.ones_like(fake_predictions), fake_predictions)
        return (generator_loss)

    def _discriminator_loss(self, real_predictions, fake_predictions):
        real_loss = cross_entropy(
            tf.ones_like(real_predictions), real_predictions)
        fake_loss = cross_entropy(
            tf.zeros_like(fake_predictions), fake_predictions)
        discriminator_loss = 0.5 * (real_loss + fake_loss)
        return (discriminator_loss)

    def generate(self, number_of_samples):
        generator_inputs = tf.random.normal(
            [number_of_samples, self.latent_dimension()])
        generated_images = self._generator.predict(generator_inputs)
        generated_images = generated_images.reshape(
            number_of_samples, self._input_shape[0], self._input_shape[1])
        generated_images = ((generated_images + 1.) / 2.) * 255.
        return (generated_images)

    def save_generated(self, number_of_samples=10):
        generated_images = self.generate(number_of_samples)
        for index, image in enumerate(generated_images):
            filename = 'image-' + str(index) + '.png'
            cv2.imwrite(filename, image)

    def _print_losses(self, losses):
        for key_value, loss_value in losses.items():
            print(key_value, '-', loss_value.numpy())

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
