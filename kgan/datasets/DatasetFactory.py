from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.datasets.MNISTDataset import MNISTDataset


class DatasetFactory(object):

    __datasets = []

    @classmethod
    def datasets(cls):
        if (len(cls.__datasets) == 0):
            cls.__datasets.append(MNISTDataset.name())

        return (cls.__datasets)

    @classmethod
    def default_dataset(cls):
        return (MNISTDataset.name())

    def __init__(self):
        pass

    @classmethod
    def create(cls, name):
        dataset = None

        if (name == MNISTDataset.name()):
            dataset = MNISTDataset()
        else:
            dataset = MNISTDataset()

        return (dataset)
