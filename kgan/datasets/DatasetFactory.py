from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.datasets.MNIST import MNIST
from kgan.datasets.FashionMNIST import FashionMNIST


class DatasetFactory(object):

    __datasets = []

    @classmethod
    def datasets(cls):
        if (len(cls.__datasets) == 0):
            cls.__datasets.append(MNIST.name())
            cls.__datasets.append(FashionMNIST.name())

        return (cls.__datasets)

    @classmethod
    def default_dataset(cls):
        return (MNIST.name())

    def __init__(self):
        pass

    @classmethod
    def create(cls, name):
        dataset = None

        if (name == MNIST.name()):
            dataset = MNIST()
        elif (name == FashionMNIST.name()):
            dataset = FashionMNIST()
        else:
            dataset = MNIST()

        return (dataset)
