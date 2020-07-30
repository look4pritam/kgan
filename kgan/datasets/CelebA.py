from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from kgan.datasets.AbstractDataset import AbstractDataset


class CelebA(AbstractDataset):

    __default_image_shape = (128, 128, 3)

    @classmethod
    def default_image_shape(cls):
        return (cls.__default_image_shape)

    @classmethod
    def name(cls):
        return ('celeba')

    def __init__(self):
        super(CelebA, self).__init__()
        self._image_shape = CelebA.default_image_shape()
        self._image_load_shape = (143, 143, 3)

    def set_image_shape(self, image_shape):
        self._image_shape = image_shape

    def image_shape(self):
        return (self._image_shape)

    def _load_image(self, image_filename):
        input_image = tf.io.read_file(image_filename)
        input_image = tf.image.decode_jpeg(input_image, 3)
        return (input_image)

    def _normalize_image(self, image):
        image = tf.cast(image, tf.float32)
        image = tf.clip_by_value(image, 0, 255) / 127.5 - 1
        return (image)

    def _random_crop(self, image):
        cropped_image = tf.image.random_crop(image, size=image_shape)
        return (cropped_image)

    def _random_jitter(self, image):
        image = tf.image.resize(image,
                                [image_load_shape[0], image_load_shape[1]])
        image = self._random_crop(image)
        image = tf.image.random_flip_left_right(image)
        return (image)

    def _preprocess_attributes(self, attributes_array):
        selected_attributes = []
        for attribute_name in default_attribute_names:
            index = attributes_to_identifiers[attribute_name]
            selected_attributes.append(attributes_array[index])

        selected_attributes = tf.convert_to_tensor(selected_attributes)
        selected_attributes = (selected_attributes + 1) // 2
        selected_attributes = selected_attributes * 1.
        return (selected_attributes)

    def load(self, batch_size):
        self.set_batch_size(batch_size)

        number_of_batches = 0
        train_dataset, test_dataset = tfds.load(
            name="mnist", split=['train', 'test'], as_supervised=True)
        train_dataset = train_dataset.concatenate(test_dataset)

        train_dataset = train_dataset.shuffle(self.buffer_size())
        train_dataset = train_dataset.batch(
            self.batch_size(), drop_remainder=True)
        train_dataset = train_dataset.map(self._augment_dataset)

        return (train_dataset, number_of_batches)

    def _augment_dataset(self, image, attributes):
        image = self._load_image(image_filename)
        image = self._random_jitter(image)
        image = self._normalize_image(image)

        attributes = self._preprocess_attributes(attributes)
        return (image, attributes)
