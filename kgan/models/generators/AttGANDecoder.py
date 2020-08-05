from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from functools import partial


def concatenate(list_of_features, list_of_attributes, layer_name):
    list_of_features = list(list_of_features) if isinstance(
        list_of_features, (list, tuple)) else [list_of_features]
    list_of_attributes = list(list_of_attributes) if isinstance(
        list_of_attributes, (list, tuple)) else [list_of_attributes]
    for index, attributes in enumerate(list_of_attributes):
        attributes = tf.reshape(
            attributes, [-1, 1, 1, attributes.shape[-1]],
            name=layer_name + 'reshape')
        attributes = tf.tile(
            attributes,
            [1, list_of_features[0].shape[1], list_of_features[0].shape[2], 1],
            name=layer_name + 'tile')
        list_of_attributes[index] = attributes
    return tf.concat(
        list_of_features + list_of_attributes,
        axis=-1,
        name=layer_name + 'concat')


class AttGANDecoder(models.Model):
    @classmethod
    def create_128(cls,
                   decoder_dimension=64,
                   upsamplings_layers=5,
                   shortcut_layers=1,
                   inject_layers=1):
        decoder = AttGANDecoder(decoder_dimension, upsamplings_layers,
                                shortcut_layers, inject_layers)
        return (decoder)

    @classmethod
    def create_256(cls,
                   decoder_dimension=64,
                   upsamplings_layers=5,
                   shortcut_layers=1,
                   inject_layers=1):
        decoder = AttGANDecoder(decoder_dimension, upsamplings_layers,
                                shortcut_layers, inject_layers)
        return (decoder)

    @classmethod
    def create_384(cls,
                   decoder_dimension=48,
                   upsamplings_layers=5,
                   shortcut_layers=1,
                   inject_layers=1):
        decoder = AttGANDecoder(decoder_dimension, upsamplings_layers,
                                shortcut_layers, inject_layers)
        return (decoder)

    def __init__(self,
                 decoder_dimension,
                 upsamplings_layers,
                 shortcut_layers,
                 inject_layers,
                 name='attgan-decoder',
                 **kwargs):
        super(AttGANDecoder, self).__init__(name=name, **kwargs)

        self._attribute_loss_weight = 10.0
        self._reconstruction_loss_weight = 100.0

        self._decoder_dimension = decoder_dimension
        self._upsamplings_layers = upsamplings_layers
        self._shortcut_layers = shortcut_layers
        self._inject_layers = inject_layers

        self._decoders = []
        filters = self._decoder_dimension
        for block_index in range(self._upsamplings_layers - 1):
            block_name = 'block-' + str(block_index + 1)

            current_decoder = self._deconvolution_block(
                filters, 4, activation_fn=tf.nn.relu, name=block_name)
            self._decoders.append(current_decoder)

            filters = filters * 2

        current_decoder = self._deconvolution_block(
            3, 4, activation_fn=tf.nn.tanh, batch_norm=False, name='block-5')
        self._decoders.append(current_decoder)

    def set_attribute_loss_weight(self, attribute_loss_weight):
        self._attribute_loss_weight = attribute_loss_weight

    def attribute_loss_weight(self):
        return (self._attribute_loss_weight)

    def set_reconstruction_loss_weight(self, reconstruction_loss_weight):
        self._reconstruction_loss_weight = reconstruction_loss_weight

    def reconstruction_loss_weight(self):
        return (self._reconstruction_loss_weight)

    def _deconvolution_block(self,
                             filters,
                             kernel_size,
                             activation_fn=tf.nn.leaky_relu,
                             batch_norm=True,
                             input_shape=None,
                             name=''):

        if (input_shape is None):
            dconv = partial(layers.Conv2DTranspose)
        else:
            dconv = partial(layers.Conv2DTranspose, input_shape=input_shape)

        blocks = [
            dconv(
                filters, (kernel_size, kernel_size),
                strides=(2, 2),
                padding="same",
                use_bias=True,
                name='dconv')
        ]

        if (batch_norm):
            blocks.append(layers.BatchNormalization(name='bnorm'))

        if (activation_fn is not None):
            if (activation_fn == tf.nn.leaky_relu):
                blocks.append(layers.LeakyReLU(alpha=0.2, name='act'))
            else:
                blocks.append(layers.Activation(activation_fn, name='act'))

        return (models.Sequential(blocks, name=name))

    def call(self, inputs, training=True):

        input_features, input_attributes = inputs

        layer_name = 'block-0-shortcut-'
        layer_input = concatenate(
            input_features[-1], input_attributes, layer_name=layer_name)
        for block_index in range(self._upsamplings_layers):
            layer_name = 'block-' + str(block_index + 1) + '-'

            decoder_layer = self._decoders[block_index]
            layer_input = decoder_layer(layer_input, training)

            if (self._shortcut_layers > block_index):
                shortcut_name = layer_name + 'shortcut-'
                layer_input = concatenate(
                    [layer_input, input_features[-2 - block_index]], [],
                    layer_name=shortcut_name)

            if (self._inject_layers > block_index):
                inject_name = layer_name + 'inject-'
                layer_input = concatenate(
                    layer_input, input_attributes, layer_name=inject_name)

        return (layer_input)
