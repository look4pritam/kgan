from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from functools import partial
import numpy as np


class AttGANDiscriminator(models.Model):
    @classmethod
    def create(cls,
               input_shape,
               number_of_attributes=40,
               discriminator_dimension=64,
               dense_dimension=1024,
               downsamplings_layers=5):
        discriminator = AttGANDiscriminator(
            number_of_attributes, discriminator_dimension, dense_dimension,
            downsamplings_layers)

        input_images = np.zeros((input_shape))
        input_images = np.expand_dims(input_images, axis=0)
        discriminator_prediction, classifier_predictions = discriminator.predict(
            input_images)
        return (discriminator)

    def __init__(self,
                 number_of_attributes=40,
                 discriminator_dimension=64,
                 dense_dimension=1024,
                 downsamplings_layers=5,
                 name='attgan-discriminator',
                 **kwargs):
        super(AttGANDiscriminator, self).__init__(name=name, **kwargs)

        self._number_of_attributes = number_of_attributes
        self._discriminator_dimension = discriminator_dimension
        self._dense_dimension = dense_dimension
        self._downsamplings_layers = downsamplings_layers

        self._features = None
        self._classifier = None
        self._discriminator = None

        self._create_features()
        self._create_classifier()
        self._create_discriminator()

    def _create_features(self):
        self._features = models.Sequential(name="features")
        filters = self._discriminator_dimension
        for block_index in range(self._downsamplings_layers):
            block_name = 'block-' + str(block_index + 1)

            current_features = self._convolution_block(
                filters, 4, name=block_name)
            self._features.add(current_features)

            filters = filters * 2

    def _create_classifier(self):
        self._classifier = models.Sequential(name='classifier')
        self._classifier.add(
            self._dense_block(
                self._dense_dimension,
                activation_fn=tf.nn.leaky_relu,
                name='dense'))
        self._classifier.add(
            self._dense_block(
                self._number_of_attributes,
                activation_fn=None,
                name='predictions'))

    def _create_discriminator(self):
        self._discriminator = models.Sequential(name='discriminator')
        self._discriminator.add(
            self._dense_block(
                self._dense_dimension,
                activation_fn=tf.nn.leaky_relu,
                name='dense'))
        self._discriminator.add(
            self._dense_block(1, activation_fn=None, name='predictions'))

    def _convolution_block(self,
                           filters,
                           kernel_size,
                           activation_fn=tf.nn.leaky_relu,
                           batch_norm=True,
                           input_shape=None,
                           name=''):

        if (input_shape is None):
            conv = partial(layers.Conv2D)
        else:
            conv = partial(layers.Conv2D, input_shape=input_shape)

        blocks = [
            conv(
                filters, (kernel_size, kernel_size),
                strides=(2, 2),
                padding="same",
                use_bias=True,
                name='conv')
        ]

        if (batch_norm):
            blocks.append(tfa.layers.InstanceNormalization(name='inorm'))

        if (activation_fn is not None):
            if (activation_fn == tf.nn.leaky_relu):
                blocks.append(layers.LeakyReLU(alpha=0.2, name='act'))
            else:
                blocks.append(layers.Activation(activation_fn, name='act'))

        return (models.Sequential(blocks, name=name))

    def _dense_block(self,
                     filters,
                     activation_fn=tf.nn.leaky_relu,
                     batch_norm=False,
                     input_shape=None,
                     name=None):

        if (input_shape is None):
            dense = partial(layers.Dense)
        else:
            dense = partial(layers.Dense, input_shape=input_shape)

        blocks = [dense(filters, use_bias=True, name='dense')]

        if (batch_norm):
            blocks.append(tfa.layers.BatchNormalization(name='bnorm'))

        if (activation_fn is not None):
            if (activation_fn == tf.nn.leaky_relu):
                blocks.append(layers.LeakyReLU(alpha=0.2, name='act'))
            else:
                blocks.append(layers.Activation(activation_fn, name='act'))

        return (models.Sequential(blocks, name=name))

    def call(self, input_image, training=True):

        layer_input = input_image
        layer_input = self._features(layer_input, training=training)
        layer_input = layers.Flatten()(layer_input)

        classifier_predictions = self._classifier(
            layer_input, training=training)
        discriminator_prediction = self._discriminator(
            layer_input, training=training)

        return (discriminator_prediction, classifier_predictions)
