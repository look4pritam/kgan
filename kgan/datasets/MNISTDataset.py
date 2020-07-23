from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds


class MNISTDataset(object):
    @classmethod
    def name(cls):
        return ('mnist')

    def __init__(self):
        self._image_shape = None
        self._batch_size = None
        self._buffer_size = None

    def batch_size(self):
        return (self._batch_size)

    def load(self, image_shape, batch_size, buffer_size=1024):
        self._image_shape = image_shape
        self._batch_size = batch_size
        self._buffer_size = buffer_size

        train_dataset, validation_dataset = tfds.load(
            name="mnist", split=['train', 'test'], as_supervised=True)

        train_dataset = train_dataset.shuffle(buffer_size).batch(
            self.batch_size())
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
