from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.AbstractGAN import AbstractGAN

import tensorflow as tf

import cv2

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

    def generate(self):
        generator_inputs = tf.random.normal(
            [self.number_of_samples(),
             self.latent_dimension()])
        generated_images = self._generator.predict(generator_inputs)
        generated_images = generated_images.reshape(self.number_of_samples(),
                                                    self._input_shape[0],
                                                    self._input_shape[1])
        generated_images = generated_images * 255.0
        return (generated_images)

    def save_generated(self):
        generated_images = self.generate()
        for index, image in enumerate(generated_images):
            filename = 'image-' + str(index) + '.png'
            cv2.imwrite(filename, image)

    def _print_losses(self, losses):
        for key_value, loss_value in losses.items():
            print(key_value, '-', loss_value.numpy())
