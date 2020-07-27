from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.ImageGAN import ImageGAN

from kgan.models.discriminators.ConvolutionalDiscriminator import ConvolutionalDiscriminator
from kgan.models.generators.ConvolutionalGenerator import ConvolutionalGenerator

import tensorflow as tf
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class DCGAN(ImageGAN):
    @classmethod
    def name(cls):
        return ('dcgan')

    def __init__(self, input_shape, latent_dimension):
        super(DCGAN, self).__init__(input_shape, latent_dimension)
        pass

    def _create_discriminator(self):
        discriminator = ConvolutionalDiscriminator.create(self.input_shape())
        return (discriminator)

    def _create_generator(self):
        generator = ConvolutionalGenerator.create(self.latent_dimension())
        return (generator)

    def _update_discriminator(self, input_batch):
        real_images, real_labels = input_batch

        # Sample random points in the latent space.
        generator_inputs = tf.random.normal(
            shape=(self.batch_size(), self.latent_dimension()))

        # Generate fake images using these random points.
        generated_images = self._generator(generator_inputs)

        # Train the discriminator.
        with tf.GradientTape() as tape:
            # Compute discriminator's predictions for real images.
            real_predictions = self._discriminator(real_images)

            # Compute discriminator's predictions for generated images.
            fake_predictions = self._discriminator(generated_images)

            # Compute discriminator loss.
            discriminator_loss = self._discriminator_loss(
                real_predictions, fake_predictions)

        gradients = tape.gradient(discriminator_loss,
                                  self._discriminator.trainable_weights)
        self._discriminator_optimizer.apply_gradients(
            zip(gradients, self._discriminator.trainable_weights))

        return (discriminator_loss)

    def _update_generator(self, input_batch):
        # Sample random points in the latent space.
        generator_inputs = tf.random.normal(
            shape=(self.batch_size(), self.latent_dimension()))

        # Train the generator.
        with tf.GradientTape() as tape:
            # Generate fake images using these random points.
            generated_images = self._generator(generator_inputs)

            # Compute discriminator's predictions for generated images.
            fake_predictions = self._discriminator(generated_images)

            # Compute generator loss using these fake predictions.
            generator_loss = self._generator_loss(fake_predictions)

        # Compute gradients of generator loss using trainable weights of generator.
        gradients = tape.gradient(generator_loss,
                                  self._generator.trainable_weights)

        # Apply gradients to trainable weights of generator.
        self._generator_optimizer.apply_gradients(
            zip(gradients, self._generator.trainable_weights))

        return (generator_loss)

    def _train_on_batch(self, input_batch):
        # Update discriminator weights.
        discriminator_loss = self._update_discriminator(input_batch)

        # Update generator weights.
        generator_loss = self._update_generator(input_batch)

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }
