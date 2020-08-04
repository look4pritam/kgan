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

    def _augment_dataset(self, image, label):
        return (image, label)

    def load_train_dataset(self, batch_size):
        self.set_batch_size(batch_size)

        number_of_batches = 0
        train_dataset, test_dataset = tfds.load(
            name="fashion_mnist", split=['train', 'test'], as_supervised=True)
        train_dataset = train_dataset.concatenate(test_dataset)

        train_dataset = train_dataset.shuffle(self.buffer_size())
        train_dataset = train_dataset.batch(
            self.batch_size(), drop_remainder=True)
        train_dataset = train_dataset.map(self._augment_dataset)

        return (train_dataset, number_of_batches)

    def load_validation_dataset(self, batch_size):
        validation_dataset = None
        number_of_batches = 0
        return (validation_dataset, number_of_batches)
