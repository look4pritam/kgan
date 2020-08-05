from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.ImageGAN import ImageGAN

from kgan.models.discriminators.DenseDiscriminator import DenseDiscriminator
from kgan.models.generators.DenseGenerator import DenseGenerator

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class GAN(ImageGAN):
    @classmethod
    def name(cls):
        return ('gan')

    def __init__(self, input_shape, latent_dimension):
        super(GAN, self).__init__(input_shape, latent_dimension)

        self._random_seed = 7
        tf.random.set_seed(self.random_seed())

    def random_seed(self):
        return (self._random_seed)

    def _create_discriminator(self):
        self._discriminator = DenseDiscriminator.create(self.input_shape())
        self._discriminator.summary()
        return (True)

    def _create_generator(self):
        self._generator = DenseGenerator.create(self.input_shape(),
                                                self.latent_dimension())
        self._generator.summary()
        return (True)

    def _create_generator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
        return (optimizer)

    def _update_discriminator(self, input_batch):
        real_images, real_labels = input_batch

        # Sample random points in the latent space.
        generator_inputs = self._create_generator_inputs(self.batch_size())

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
        generator_inputs = self._create_generator_inputs(self.batch_size())

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

        # Update generator weights.
        generator_loss = self._update_generator(input_batch)

        # Update discriminator weights.
        discriminator_loss = self._update_discriminator(input_batch)

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }
