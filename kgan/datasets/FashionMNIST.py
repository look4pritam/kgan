from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds

from kgan.datasets.AbstractDataset import AbstractDataset


class FashionMNIST(AbstractDataset):

    __default_image_shape = (28, 28, 1)

    @classmethod
    def default_image_shape(cls):
        return (cls.__default_image_shape)

    @classmethod
    def name(cls):
        return ('fashion_mnist')

    def __init__(self):
        super(FashionMNIST, self).__init__()
        self._image_shape = FashionMNIST.default_image_shape()

    def set_image_shape(self, image_shape):
        self._image_shape = image_shape

    def image_shape(self):
        return (self._image_shape)

    def load(self, batch_size):
        self.set_batch_size(batch_size)

        (train_images, train_labels), (
            validation_images,
            validation_labels) = tf.keras.datasets.fashion_mnist.load_data()

        train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                            1).astype('float32')
        train_labels = tf.one_hot(train_labels, depth=10)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images,
                                                            train_labels))

        validation_images = validation_images.reshape(
            train_images.shape[0], 28, 28, 1).astype('float32')
        validation_labels = tf.one_hot(validation_labels, depth=10)
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (validation_images, validation_labels))

        self._buffer_size = train_images.shape[0]
        train_dataset = train_dataset.shuffle(self.buffer_size()).batch(
            self.batch_size(), drop_remainder=True)
        train_dataset = train_dataset.map(self._augment_image)

        validation_dataset = validation_dataset.batch(self.batch_size())
        validation_dataset = validation_dataset.map(self._normalize_image)

        return (train_dataset, validation_dataset)

    def _normalize_image(self, image, label):
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5
        return (image, label)

    def _augment_image(self, image, label):
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5
        return (image, label)
