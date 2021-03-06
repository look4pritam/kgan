from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from functools import partial


class AttGANEncoder(models.Model):
    @classmethod
    def create_128(cls, encoder_dimension=64, downsamplings_layers=5):
        encoder = AttGANEncoder(encoder_dimension, downsamplings_layers)
        return (encoder)

    @classmethod
    def create_256(cls, encoder_dimension=64, downsamplings_layers=5):
        encoder = AttGANEncoder(encoder_dimension, downsamplings_layers)
        return (encoder)

    def __init__(self,
                 encoder_dimension,
                 downsamplings_layers,
                 name='attgan-encoder',
                 **kwargs):
        super(AttGANEncoder, self).__init__(name=name, **kwargs)

        self._encoder_dimension = encoder_dimension
        self._downsamplings_layers = downsamplings_layers

        self._image_features = []
        filters = self._encoder_dimension
        for block_index in range(self._downsamplings_layers):
            block_name = 'block-' + str(block_index + 1)

            current_features = self._convolution_block(
                filters, 4, name=block_name)
            self._image_features.append(current_features)

            filters = filters * 2

    def _convolution_block(self,
                           filters,
                           kernel_size,
                           activation_fn=tf.nn.leaky_relu,
                           batch_norm=True,
                           input_shape=None,
                           name=None):

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
            blocks.append(layers.BatchNormalization(name='bnorm'))

        if (activation_fn is not None):
            if (activation_fn == tf.nn.leaky_relu):
                blocks.append(layers.LeakyReLU(alpha=0.2, name='act'))
            else:
                blocks.append(layers.Activation(activation_fn, name='act'))

        return (models.Sequential(blocks, name=name))

    def call(self, input_image, training=True):
        image_features = []

        layer_input = input_image
        for current_features in self._image_features:
            layer_input = current_features(layer_input, training=training)
            image_features.append(layer_input)

        return (image_features)
