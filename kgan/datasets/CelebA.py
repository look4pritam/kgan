from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from kgan.datasets.AbstractDataset import AbstractDataset

import tensorflow as tf
import numpy as np


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

        self._attributes_to_identifiers = {
            '5_o_Clock_Shadow': 0,
            'Arched_Eyebrows': 1,
            'Attractive': 2,
            'Bags_Under_Eyes': 3,
            'Bald': 4,
            'Bangs': 5,
            'Big_Lips': 6,
            'Big_Nose': 7,
            'Black_Hair': 8,
            'Blond_Hair': 9,
            'Blurry': 10,
            'Brown_Hair': 11,
            'Bushy_Eyebrows': 12,
            'Chubby': 13,
            'Double_Chin': 14,
            'Eyeglasses': 15,
            'Goatee': 16,
            'Gray_Hair': 17,
            'Heavy_Makeup': 18,
            'High_Cheekbones': 19,
            'Male': 20,
            'Mouth_Slightly_Open': 21,
            'Mustache': 22,
            'Narrow_Eyes': 23,
            'No_Beard': 24,
            'Oval_Face': 25,
            'Pale_Skin': 26,
            'Pointy_Nose': 27,
            'Receding_Hairline': 28,
            'Rosy_Cheeks': 29,
            'Sideburns': 30,
            'Smiling': 31,
            'Straight_Hair': 32,
            'Wavy_Hair': 33,
            'Wearing_Earrings': 34,
            'Wearing_Hat': 35,
            'Wearing_Lipstick': 36,
            'Wearing_Necklace': 37,
            'Wearing_Necktie': 38,
            'Young': 39
        }
        self._identifiers_to_attributes = {
            v: k
            for k, v in self._attributes_to_identifiers.items()
        }

        self._default_attribute_names = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
            'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
            'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
        self._number_of_attributes = len(self._default_attribute_names)
        print('number of attributes -', self.number_of_attributes())

    def number_of_attributes(self):
        return (self._number_of_attributes)

    def set_image_shape(self, image_shape):
        self._image_shape = image_shape

    def image_shape(self):
        return (self._image_shape)

    def image_load_shape(self):
        return (self._image_load_shape)

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
        image = tf.image.resize(
            image, [self._image_load_shape[0], self._image_load_shape[1]])
        image = self._random_crop(image)
        image = tf.image.random_flip_left_right(image)
        return (image)

    def _preprocess_attributes(self, attributes_array):
        selected_attributes = []
        for attribute_name in self._default_attribute_names:
            index = self._attributes_to_identifiers[attribute_name]
            selected_attributes.append(attributes_array[index])

        selected_attributes = tf.convert_to_tensor(selected_attributes)
        selected_attributes = (selected_attributes + 1) // 2
        selected_attributes = selected_attributes * 1.
        return (selected_attributes)

    def _create_dataset(self,
                        image_root_dir='img_align_celeba/images',
                        attribute_filename='img_align_celeba/train_label.txt'):
        image_names = np.genfromtxt(attribute_filename, dtype=str, usecols=0)
        image_filename_array = np.array([
            os.path.join(image_root_dir, image_name)
            for image_name in image_names
        ])

        attributes_array = np.genfromtxt(
            attribute_filename, dtype=float, usecols=range(1, 41))

        number_of_batches = len(image_filename_array) // self.batch_size()

        memory_data = (image_filename_array, attributes_array)
        dataset = tf.data.Dataset.from_tensor_slices(memory_data)

        return (dataset, number_of_batches)

    def load(self, batch_size):
        self.set_batch_size(batch_size)

        train_dataset, number_of_batches = self._create_dataset()

        auto_tune = tf.data.experimental.AUTOTUNE

        train_dataset = train_dataset.map(
            self._augment_dataset, num_parallel_calls=auto_tune)
        train_dataset = train_dataset.shuffle(self.buffer_size())
        train_dataset = train_dataset.batch(
            self.batch_size(), drop_remainder=True)
        train_dataset = train_dataset.prefetch(auto_tune)

        return (train_dataset, number_of_batches)

    def _augment_dataset(self, image_filename, attributes):
        image = self._load_image(image_filename)
        image = self._random_jitter(image)
        image = self._normalize_image(image)

        attributes = self._preprocess_attributes(attributes)
        return (image, attributes)
