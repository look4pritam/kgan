from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.AbstractGAN import AbstractGAN

from kgan.layers.BatchNormalization import BatchNormalization

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class WGANGP(AbstractGAN):
    @classmethod
    def name(cls):
        return ('wgangp')

    def __init__(self, input_shape, latent_dimension):
        super(WGANGP, self).__init__()

        self._input_shape = input_shape
        self._latent_dimension = latent_dimension

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
        generator = ConvolutionalGenerator.create(self.latent_dimension())

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
                units=128 * 7 * 7,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.02),
                bias_initializer=tf.keras.initializers.Constant(value=0.0)))
        generator.add(BatchNormalization(is_training=true))
        generator.add(layers.ReLU())

        generator.add(layers.Reshape((7, 7, 128)))

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
        discriminator_loss = real_loss + fake_loss
        return (discriminator_loss)

    def generate(self, number_of_samples):
        generator_inputs = tf.random.normal(
            [number_of_samples, self.latent_dimension()])
        generated_images = self._generator.predict(generator_inputs)
        generated_images = generated_images.reshape(
            number_of_samples, self._input_shape[0], self._input_shape[1])
        generated_images = generated_images * 127.5 + 127.5
        return (generated_images)

    def save_generated(self, number_of_samples=10):
        generated_images = self.generate(number_of_samples)
        for index, image in enumerate(generated_images):
            filename = 'image-' + str(index) + '.png'
            cv2.imwrite(filename, image)

    def _print_losses(self, losses):
        for key_value, loss_value in losses.items():
            print(key_value, '-', loss_value.numpy())

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
