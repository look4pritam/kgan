from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.ImageGAN import ImageGAN

from kgan.layers.BatchNormalization import BatchNormalization

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.optimizers import RMSprop


class WGANGP(ImageGAN):
    @classmethod
    def name(cls):
        return ('wgangp')

    def __init__(self, input_shape, latent_dimension):
        super(WGANGP, self).__init__()

        self._discriminator = self._create_discriminator()
        self._generator = self._create_generator()

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
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
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
        discriminator.add(BatchNormalization(is_training=true))
        discriminator.add(layers.LeakyReLU(alpha=0.2))

        discriminator.add(layers.Flatten())

        discriminator.add(
            layers.Dense(
                units=1024,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        discriminator.add(BatchNormalization(is_training=true))
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
            Dense(
                units=1024,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        generator.add(BatchNormalization(is_training=true))
        generator.add(layers.ReLU())

        generator.add(
            Dense(
                units=generator_size,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        generator.add(BatchNormalization(is_training=true))
        generator.add(layers.ReLU())

        generator.add(layers.Reshape(generator_shape))

        generator.add(
            UpConv2D(
                filters=64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                use_bias=True,
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        generator.add(BatchNormalization(is_training=true))
        generator.add(layers.ReLU())

        generator.add(
            UpConv2D(
                filters=1,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                use_bias=True,
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        generator.add(tf.keras.activations.tanh())

        return (generator)

    def _create_generator_optimizer(self, learning_rate):
        optimizer = RMSprop(learning_rate=5 * learning_rate)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = RMSprop(learning_rate=learning_rate)
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
