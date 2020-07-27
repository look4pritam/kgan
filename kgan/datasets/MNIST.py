from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds

from kgan.datasets.AbstractDataset import AbstractDataset


class MNIST(AbstractDataset):

    __default_image_shape = (28, 28, 1)

    @classmethod
    def default_image_shape(cls):
        return (cls.__default_image_shape)

    @classmethod
    def name(cls):
        return ('mnist')

    def __init__(self):
        super(MNIST, self).__init__()
        self._image_shape = MNIST.default_image_shape()

    def set_image_shape(self, image_shape):
        self._image_shape = image_shape

    def image_shape(self):
        return (self._image_shape)

    def load(self, batch_size):
        self.set_batch_size(batch_size)

        train_dataset, test_dataset = tfds.load(
            name="mnist", split=['train', 'test'], as_supervised=True)
        train_dataset = train_dataset.concatenate(test_dataset)

        train_dataset = train_dataset.shuffle(self.buffer_size())
        train_dataset = train_dataset.batch(
            self.batch_size(), drop_remainder=True)
        train_dataset = train_dataset.map(self._augment_dataset)

        return (train_dataset)

    def _augment_dataset(self, image, label):
        return (image, label)
