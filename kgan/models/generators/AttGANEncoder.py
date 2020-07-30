from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from functools import partial


class AttGANEncoder(models.Model):
    @classmethod
    def create(cls, input_shape, encoder_dimension=64, downsamplings_layers=5):

        encoder = AttGANEncoder(encoder_dimension, downsamplings_layers)

        input_images = np.zeros((input_shape))
        input_images = np.expand_dims(input_images, axis=0)

        image_features = encoder.predict(input_images)

        return (encoder)

    def __init__(self,
                 encoder_dimension,
                 downsamplings_layers,
                 name='attgan-encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self._encoder_dimension = encoder_dimension
        self._downsamplings_layers = downsamplings_layers

        self._encoders = []
        filters = self._encoder_dimension
        for block_index in range(self._downsamplings_layers):
            block_name = 'block-' + str(block_index + 1)

            current_encoder = self._convolution_block(
                filters, 4, name=block_name)
            self._encoders.append(current_encoder)

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
        for current_encoder in self._encoders:
            layer_input = current_encoder(layer_input, training=training)
            image_features.append(layer_input)

        return (image_features)
