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

    def _normalize_dataset(self, dataset_sample):
        image, label = dataset_sample
        image = tf.cast(image, tf.float32) / 255.
        return (image, label)

    def _train_on_batch(self, input_batch):
        real_images, real_labels = input_batch

        # Sample random points in the latent space.
        generator_inputs = tf.random.normal(
            shape=(self.batch_size(), self.latent_dimension()))

        # Decode these random points to fake images.
        generated_images = self._generator(generator_inputs)

        # Combine generated images with real images.
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Create combined labels for discriminating real images from generated images.
        labels = tf.concat([
            tf.ones((self.batch_size(), 1)),
            tf.zeros((self.batch_size(), 1))
        ],
                           axis=0)

        # Add random noise to the labels. (important trick)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self._discriminator(combined_images)
            discriminator_loss = cross_entropy(labels, predictions)

        gradients = tape.gradient(discriminator_loss,
                                  self._discriminator.trainable_weights)
        self._discriminator_optimizer.apply_gradients(
            zip(gradients, self._discriminator.trainable_weights))

        # Sample random points in the latent space.
        generator_inputs = tf.random.normal(
            shape=(self.batch_size(), self.latent_dimension()))

        # Create misleading labels, which will be predicted as real images.
        misleading_labels = tf.zeros((self.batch_size(), 1))

        # Train the generator.
        with tf.GradientTape() as tape:
            # Decode random point to fake images.
            generated_images = self._generator(generator_inputs)

            predictions = self._discriminator(generated_images)
            generator_loss = cross_entropy(misleading_labels, predictions)

        gradients = tape.gradient(generator_loss,
                                  self._generator.trainable_weights)
        self._generator_optimizer.apply_gradients(
            zip(gradients, self._generator.trainable_weights))

        return {
            'generator': generator_loss,
            'discriminator': discriminator_loss
        }
