from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.GAN import GAN

from kgan.layers.BatchNormalization import BatchNormalization

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from tensorflow.keras.optimizers import RMSprop

import numpy as np


class WGANGP(GAN):
    @classmethod
    def name(cls):
        return ('wgangp')

    def __init__(self, input_shape, latent_dimension):
        super(WGANGP, self).__init__(input_shape, latent_dimension)

        self._gradient_penalty_weight = 10.

    def _create_discriminator(self):
        discriminator = models.Sequential(name='discriminator')

        discriminator.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=0.02),
                use_bias=True,
                bias_initializer=tf.keras.initializers.Constant(value=0.0),
                input_shape=self.input_shape()))
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=0.02),
                use_bias=True,
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        discriminator.add(BatchNormalization(is_training=True))
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(layers.Flatten())

        discriminator.add(
            layers.Dense(
                units=1024,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        discriminator.add(BatchNormalization(is_training=True))
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(
            layers.Dense(
                units=1,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))

        self._discriminator = discriminator

        return (True)

    def _create_generator(self):
        generator_shape = (7, 7, 128)
        generator_size = np.prod(generator_shape)

        generator = models.Sequential(name='generator')

        generator.add(
            layers.Dense(
                units=generator_size,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                bias_initializer=tf.keras.initializers.Constant(value=0.0),
                input_shape=(self.latent_dimension(), ),
            ))
        generator.add(BatchNormalization(is_training=True))
        generator.add(layers.ReLU())

        generator.add(layers.Reshape(generator_shape))

        generator.add(
            layers.Conv2DTranspose(
                filters=64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                use_bias=True,
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        generator.add(BatchNormalization(is_training=True))
        generator.add(layers.ReLU())

        generator.add(
            layers.Conv2DTranspose(
                filters=1,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                use_bias=True,
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        generator.add(layers.Activation(tf.keras.activations.tanh))

        self._generator = generator

        return (True)

    def _create_generator_optimizer(self, learning_rate):
        optimizer = RMSprop(learning_rate=5 * learning_rate)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = RMSprop(learning_rate=learning_rate)
        return (optimizer)

    def set_gradient_penalty_weight(self, gradient_penalty_weight):
        self._gradient_penalty_weight = gradient_penalty_weight

    def gradient_penalty_weight(self):
        return (self._gradient_penalty_weight)

    def _discriminator_loss(self, real_predictions, fake_predictions):
        # Compute discriminator loss.
        discriminator_loss = tf.reduce_mean(
            -real_predictions) + tf.reduce_mean(fake_predictions)
        return (discriminator_loss)

    def _generator_loss(self, fake_predictions):
        # Compute generator loss.
        generator_loss = tf.reduce_mean(-fake_predictions)
        return (generator_loss)

    def _create_generator_inputs(self, input_batch):
        generator_inputs = tf.random.uniform(
            [self.batch_size(), self.latent_dimension()], minval=-1, maxval=1)
        return (generator_inputs)

    def _gradient_penalty(self, real_images, fake_images):
        with tf.GradientTape() as gp_tape:
            alpha = tf.random.uniform([self.batch_size()],
                                      0.,
                                      1.,
                                      dtype=tf.float32)
            alpha = tf.reshape(alpha, (-1, 1, 1, 1))
            sample_images = real_images + alpha * (fake_images - real_images)

            gp_tape.watch(sample_images)
            sample_predictions = self._discriminator(
                sample_images, training=False)

        gradients = gp_tape.gradient(sample_predictions, sample_images)
        gradients_l2_norm = tf.sqrt(
            tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gradients_l2_norm - 1)**2)

        return (gradient_penalty)

    def _update_discriminator(self, input_batch):
        real_images, _ = input_batch

        # Sample random points in the latent space.
        generator_inputs = self._create_generator_inputs(input_batch)

        # Generate fake images using these random points.
        fake_images = self._generator(generator_inputs)

        # Train the discriminator.
        with tf.GradientTape() as tape:

            # Compute discriminator's predictions for real images.
            real_predictions = self._discriminator(real_images)

            # Compute discriminator's predictions for generated images.
            fake_predictions = self._discriminator(fake_images)

            #discriminator_loss = tf.reduce_mean(-real_predictions) + tf.reduce_mean(fake_predictions)
            discriminator_loss = self._discriminator_loss(
                real_predictions, fake_predictions)

            # Compute gradient penalty using real and fake images.
            gradient_penalty = self._gradient_penalty(real_images, fake_images)

            # Update discriminator loss using gradient penalty value.
            discriminator_loss = discriminator_loss + self.gradient_penalty_weight(
            ) * gradient_penalty

        gradients_of_discriminator = tape.gradient(
            discriminator_loss, self._discriminator.trainable_variables)

        self._discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator,
                self._discriminator.trainable_variables))

        return (discriminator_loss)
