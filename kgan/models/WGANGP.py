from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.ImageGAN import ImageGAN

from kgan.layers.BatchNormalization import BatchNormalization

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from tensorflow.keras.optimizers import RMSprop

import numpy as np


class WGANGP(ImageGAN):
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

        return (discriminator)

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

        return (generator)

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

    def _train_on_batch(self, input_batch):
        real_samples, _ = input_batch
        generator_inputs = tf.random.uniform(
            [self.batch_size(), self.latent_dimension()], minval=-1, maxval=1)

        with tf.GradientTape() as generator_tape, tf.GradientTape(
        ) as discriminator_tape:
            fake_samples = self._generator(generator_inputs, training=True)

            fake_predictions = self._discriminator(fake_samples, training=True)
            real_predictions = self._discriminator(real_samples, training=True)

            #discriminator_loss = tf.reduce_mean(-real_predictions) + tf.reduce_mean(fake_predictions)
            discriminator_loss = self._discriminator_loss(
                real_predictions, fake_predictions)

            #generator_loss = tf.reduce_mean(-fake_predictions)
            generator_loss = self._generator_loss(fake_predictions)

            with tf.GradientTape() as gp_tape:
                alpha = tf.random.uniform([self.batch_size()],
                                          0.,
                                          1.,
                                          dtype=tf.float32)
                alpha = tf.reshape(alpha, (-1, 1, 1, 1))
                sample_images = real_samples + alpha * (
                    fake_samples - real_samples)

                gp_tape.watch(sample_images)
                sample_predictions = self._discriminator(
                    sample_images, training=False)

            gradients = gp_tape.gradient(sample_predictions, sample_images)
            grad_l2 = tf.sqrt(
                tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((grad_l2 - 1)**2)
            discriminator_loss += self.gradient_penalty_weight(
            ) * gradient_penalty

        gradients_of_discriminator = discriminator_tape.gradient(
            discriminator_loss, self._discriminator.trainable_variables)
        gradients_of_generator = generator_tape.gradient(
            generator_loss, self._generator.trainable_variables)

        self._discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator,
                self._discriminator.trainable_variables))
        self._generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self._generator.trainable_variables))

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }
