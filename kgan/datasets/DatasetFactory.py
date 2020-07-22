from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.datasets.MNISTDataset import MNISTDataset


class DatasetFactory(object):
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
