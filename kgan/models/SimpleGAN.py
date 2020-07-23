from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.SimpleDiscriminator import SimpleDiscriminator
from kgan.models.SimpleGenerator import SimpleGenerator

import tensorflow as tf
import matplotlib.pyplot as plot

from tensorflow.keras.optimizers import Adam

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class SimpleGAN(object):
    @classmethod
    def name(cls):
        return ('gan')

    def __init__(self, input_shape, latent_dimension):
        self._input_shape = input_shape
        self._latent_dimension = latent_dimension

        self._discriminator = self._create_discriminator()
        self._generator = self._create_generator()

        self._batch_size = 0

    def _create_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        return (optimizer)

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

    def batch_size(self):
        return (self._batch_size)

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
        return (generated_images)

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
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss
        }

    def show_generated(self,
                       number_of_samples=10,
                       dim=(1, 10),
                       figsize=(12, 2)):
        generated_images = self.generate(number_of_samples)

        plot.figure(figsize=figsize)
        for i in range(number_of_samples):
            plot.subplot(dim[0], dim[1], i + 1)
            plot.imshow(
                generated_images[i], interpolation='nearest', cmap='gray_r')
            plot.axis('off')

        plot.tight_layout()
        plot.show()

    def train(self,
              train_dataset,
              batch_size,
              epochs,
              learning_rate=0.0001,
              validation_dataset=None):

        self._batch_size = batch_size
        generation_frequency = 2000

        self._generator_optimizer = self._create_optimizer(learning_rate)
        self._discriminator_optimizer = self._create_optimizer(learning_rate)
        batch_index = 0
        for current_epoch in range(epochs):
            for current_batch in train_dataset:
                losses = self._train_on_batch(current_batch)
                batch_index = batch_index + 1
                if (batch_index % generation_frequency == 0):
                    #generated_images = self.generate(10)
                    self.show_generated()

            #print('generator loss -', losses['generator_loss'].numpy())
            #print('discriminator loss -', losses['discriminator_loss'].numpy())

        return (True)
