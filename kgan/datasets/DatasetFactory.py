from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.datasets.MNISTDataset import MNISTDataset


class DatasetFactory(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, name):
        train_dataset = None
        validation_dataset = None

        if (name == MNISTDataset.name()):
            train_dataset, validation_dataset = MNISTDataset()
        else:
            train_dataset, validation_dataset = MNISTDataset()

        return (train_dataset, validation_dataset)
