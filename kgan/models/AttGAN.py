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

        self._d_attribute_loss_weight = 1.0

        self._g_attribute_loss_weight = 10.0
        self._g_reconstruction_loss_weight = 100.0

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
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
        return (optimizer)

    '''
    def compute_generator_loss(input_image, input_attributes):

    target_attributes = tf.random.shuffle(input_attributes)

    scaled_input_attributes = input_attributes * 2. - 1.
    scaled_target_attributes = target_attributes * 2. - 1.

    # Generator
    image_features = encoder_model(input_image)

    reconstructed_image = decoder_model([image_features, scaled_input_attributes])
    fake_image = decoder_model([image_features, scaled_target_attributes])

    # Discriminator
    fake_image_prediction, fake_image_attributes = discriminator_model(fake_image)

    fake_image_prediction_loss = tf.reduce_mean(-fake_image_prediction)
    fake_image_attributes_loss = tf.compat.v1.losses.sigmoid_cross_entropy(target_attributes, fake_image_attributes)  
  
    image_reconstruction_loss = tf.compat.v1.losses.absolute_difference(input_image, reconstructed_image)
   
    generator_loss = (  fake_image_prediction_loss 
                    + fake_image_attributes_loss * g_attribute_loss_weight 
                    + image_reconstruction_loss * g_reconstruction_loss_weight
                    )  
  
    write_generator_loss(fake_image_prediction_loss, fake_image_attributes_loss, image_reconstruction_loss, generator_loss)
  
    return(generator_loss)
    '''

    def _update_generator(self, input_batch):
        # Sample random points in the latent space.
        generator_inputs = self._create_generator_inputs(input_batch)

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

        if ((self.current_step() % self.cycle_number()) == 0):
            # Update generator weights.
            generator_loss = self._update_generator(input_batch)
            return {'generator': generator_loss}
        else:
            # Update discriminator weights.
            discriminator_loss = self._update_discriminator(input_batch)
            return {'discriminator': discriminator_loss}
