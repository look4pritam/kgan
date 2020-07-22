from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.datasets.MNISTDataset import MNISTDataset


class DatasetFactory(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, name, image_shape, batch_size):
        train_dataset = None
        validation_dataset = None

        if (name == MNISTDataset.name()):
            train_dataset, validation_dataset = MNISTDataset(
                image_shape, batch_size)
        else:
            train_dataset, validation_dataset = MNISTDataset(
                image_shape, batch_size)

        return (train_dataset, validation_dataset)
