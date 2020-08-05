from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.WGANGP import WGANGP

from kgan.models.discriminators.AttGANDiscriminator import AttGANDiscriminator

from kgan.models.generators.AttGANEncoder import AttGANEncoder
from kgan.models.generators.AttGANDecoder import AttGANDecoder

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np


class AttGAN(WGANGP):
    @classmethod
    def name(cls):
        return ('attgan')

    def __init__(self, input_shape, latent_dimension):
        super(AttGAN, self).__init__(input_shape, latent_dimension)
        self._encoder = None
        self._decoder = None

    def encoder_filename(self):
        return ('encoder.h5')

    def decoder_filename(self):
        return ('decoder.h5')

    def discriminator_filename(self):
        return ('discriminator.h5')

    def generate_samples(self, generator_inputs):
        input_image, image_attributes = generator_inputs

        input_image = tf.expand_dims(input_image, axis=0)
        image_attributes = tf.expand_dims(image_attributes, axis=0)

        image_features = self._encoder.predict(input_image)
        generated_images = self._decoder.predict(
            [image_features, image_attributes])
        generated_images = self._decode_image(generated_images)

        return (generated_images)

    def _create_discriminator(self):
        self._discriminator = AttGANDiscriminator.create(self.input_shape())
        self._discriminator.summary()
        return (True)

    def _create_generator(self):
        self._generator = None

        self._encoder = AttGANEncoder.create()
        self._decoder = AttGANDecoder.create()

        input_images = np.zeros(self.input_shape())
        input_images = np.expand_dims(input_images, axis=0)

        image_attributes = np.zeros(self.latent_dimension())
        image_attributes = np.expand_dims(image_attributes, axis=0)

        image_features = self._encoder.predict(input_images)
        print('number of image features -', len(image_features))
        for index, image_feature in enumerate(image_features):
            print('image feature -', (index + 1), image_feature.shape)

        generated_images = self._decoder.predict(
            [image_features, image_attributes])
        print('generated image shape -', generated_images.shape)
        self._encoder.summary()
        self._decoder.summary()
        return (True)

    def _load_weights(self):
        status = True

        self._discriminator.load_weights(self.discriminator_filename())

        self._encoder.load_weights(self.encoder_filename())
        self._decoder.load_weights(self.decoder_filename())
        return (status)

    def _normalize_dataset(self, image, attributes):
        return (image, attributes)

    def _update_learning_rate(self, current_epoch, number_of_epochs):
        start_decay_epoch = number_of_epochs // 2
        if (current_epoch >= start_decay_epoch):
            self._learning_rate = self.base_learning_rate() * (
                1 - 1 / (number_of_epochs - start_decay_epoch + 1) *
                (current_epoch - start_decay_epoch + 1))
        else:
            self._learning_rate = self.base_learning_rate()

        return (True)

    def _create_generator_optimizer(self, learning_rate):
        # AttGAN - Adam optimizer - base_learning_rate=0.0002, beta_1=0.5, beta_2=0.999
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        # AttGAN - Adam optimizer - base_learning_rate=0.0002, beta_1=0.5, beta_2=0.999
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
        return (optimizer)

    def _update_discriminator(self, input_batch):

        # Extract input images and image attributes from current input batch.
        real_images, input_attributes = input_batch

        # Generate target attributes from input attributes.
        target_attributes = tf.random.shuffle(input_attributes)

        # Transform target attributes.
        scaled_target_attributes = target_attributes * 2. - 1.

        # Generate image features for input image.
        image_features = self._encoder(real_images)

        # Generate fake image using image features and target attributes.
        fake_images = self._decoder([image_features, scaled_target_attributes])

        # Train the discriminator.
        with tf.GradientTape() as tape:

            # Compute discriminator's predictions for real images.
            real_image_prediction, real_image_attributes = self._discriminator(
                real_images)

            # Compute discriminator's predictions for generated images.
            fake_image_prediction, fake_image_attributes = self._discriminator(
                fake_images)

            discriminator_loss = self._discriminator_loss(
                real_image_prediction, fake_image_prediction)

            # Compute gradient penalty using real and fake images.
            gradient_penalty = self._gradient_penalty(real_images, fake_images)

            # Compute image attribute loss for real images.
            real_image_attributes_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
                input_attributes, real_image_attributes)

            # Update discriminator loss using gradient penalty value.
            discriminator_loss = discriminator_loss + self.gradient_penalty_weight(
            ) * gradient_penalty + real_image_attributes_loss * self._discriminator.attribute_loss_weight(
            )

        # Compute gradients of discriminator loss using trainable weights of discriminator model.
        gradients = tape.gradient(discriminator_loss,
                                  self._discriminator.trainable_variables)

        # Apply gradients to trainable weights of discriminator.
        self._discriminator_optimizer.apply_gradients(
            zip(gradients, self._discriminator.trainable_variables))

        return (discriminator_loss)

    def _update_generator(self, input_batch):

        # Extract input images and image attributes from current input batch.
        real_images, input_attributes = input_batch

        # Generate target attributes from input attributes.
        target_attributes = tf.random.shuffle(input_attributes)

        # Transform input and target attributes.
        scaled_input_attributes = input_attributes * 2. - 1.
        scaled_target_attributes = target_attributes * 2. - 1.

        # Train the generator.
        with tf.GradientTape() as tape:
            # Generate image features for input image.
            image_features = self._encoder(real_images)

            # Reconstruct input image.
            reconstructed_imags = self._decoder(
                [image_features, scaled_input_attributes])

            # Generate fake image using image features and target attributes.
            fake_images = self._decoder(
                [image_features, scaled_target_attributes])

            # Generate image predictions and attributes for fake image.
            fake_image_prediction, fake_image_attributes = self._discriminator(
                fake_images)

            # Compute generator loss using these fake image predictions.
            fake_image_prediction_loss = self._generator_loss(
                fake_image_prediction)

            # Compute fake image attribute loss.
            fake_image_attributes_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
                target_attributes, fake_image_attributes)

            # Compute image reconstruction loss.
            image_reconstruction_loss = tf.compat.v1.losses.absolute_difference(
                real_images, reconstructed_imags)

            # Compute generator loss.
            generator_loss = fake_image_prediction_loss + fake_image_attributes_loss * self._encoder.attribute_loss_weight(
            ) + image_reconstruction_loss * self._encoder.reconstruction_loss_weight(
            )

        # Compute gradients of generator loss using trainable weights of encoder and decoder models.
        gradients = tape.gradient(generator_loss, [
            *self._encoder.trainable_variables,
            *self._decoder.trainable_variables
        ])

        # Apply gradients to trainable weights of encoder and decoder models.
        self._generator_optimizer.apply_gradients(
            zip(gradients, [
                *self._encoder.trainable_variables,
                *self._decoder.trainable_variables
            ]))

        return (generator_loss)

    def _train_on_batch(self, input_batch):

        if ((self.current_step() % self.cycle_number()) == 0):
            # Update generator weights.
            generator_loss = self._update_generator(input_batch)
            return {'generator': generator_loss}
        else:
            # Update discriminator weights.
            discriminator_loss = self._update_discriminator(input_batch)
            return {'discriminator': discriminator_loss}

    def _save_samples(self):
        print('generating samples - start')
        print('generating samples - end')
