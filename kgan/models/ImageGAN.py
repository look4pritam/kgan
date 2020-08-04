from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.AbstractGAN import AbstractGAN

import tensorflow as tf
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class ImageGAN(AbstractGAN):
    def __init__(self, input_shape, latent_dimension):
        super(ImageGAN, self).__init__()

        self._input_shape = input_shape
        self._latent_dimension = latent_dimension

    def input_shape(self):
        return (self._input_shape)

    def latent_dimension(self):
        return (self._latent_dimension)

    def _create_summary_writer(self, logdir='logs'):
        self._summary_writer = tf.summary.create_file_writer(logdir)
        return (self._summary_writer)

    def _discriminator_loss(self, real_predictions, fake_predictions):
        # Create labels for real images. - zeros.
        real_labels = tf.zeros_like(real_predictions)

        # Add random noise to the labels.
        real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))

        # Compute discriminator loss for real images.
        real_loss = cross_entropy(real_labels, real_predictions)

        # Create labels for fake images. - ones.
        fake_labels = tf.ones_like(fake_predictions)

        # Add random noise to the labels.
        fake_labels += 0.05 * tf.random.uniform(tf.shape(fake_labels))

        # Compute discriminator loss for fake images.
        fake_loss = cross_entropy(fake_labels, fake_predictions)

        # Compute total discriminator loss.
        discriminator_loss = 0.5 * (real_loss + fake_loss)

        return (discriminator_loss)

    def _generator_loss(self, fake_predictions):
        # Create fake labels, which will be predicted as real images.
        fake_labels = tf.zeros_like(fake_predictions)

        # Compute generator loss.
        generator_loss = cross_entropy(fake_labels, fake_predictions)
        return (generator_loss)

    def _normalize_dataset(self, image, label):
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5
        return (image, label)

    def _decode_image(self, input_image):
        input_image = input_image * 127.5 + 127.5
        return (input_image)

    def _create_generator_inputs(self):
        generator_inputs = tf.random.normal(
            [self.number_of_samples(),
             self.latent_dimension()])
        return (generator_inputs)

    def generate_samples(self, generator_inputs):
        generator_inputs = self._create_generator_inputs()

        generated_images = self._generator.predict(generator_inputs)
        generated_images = generated_images.reshape(self.number_of_samples(),
                                                    self._input_shape[0],
                                                    self._input_shape[1])
        generated_images = self._decode_image(generated_images)
        return (generated_images)

    def _load_weights(self):
        return (False)

    def _print_losses(self, losses):
        print('loss values are -')
        with self._summary_writer.as_default():
            for key_value, loss_value in losses.items():
                print(key_value, '-', loss_value.numpy())
                tf.summary.scalar(
                    key_value, loss_value, step=self.current_step())
        self._summary_writer.flush()
