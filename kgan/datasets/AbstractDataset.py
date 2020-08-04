from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class AbstractDataset(object):

    __default_batch_size = 128
    __default_buffer_size = 1024

    @classmethod
    def default_batch_size(cls):
        return (cls.__default_batch_size)

    @classmethod
    def default_buffer_size(cls):
        return (cls.__default_buffer_size)

    def __init__(self):
        self._batch_size = AbstractDataset.default_batch_size()
        self._buffer_size = AbstractDataset.default_buffer_size()

    def set_batch_size(self, batch_size):
        if (batch_size > 0):
            self._batch_size = batch_size
        else:
            self._batch_size = AbstractDataset.default_batch_size()

    def batch_size(self):
        return (self._batch_size)

    def set_buffer_size(self, buffer_size):
        if (buffer_size > 0):
            self._buffer_size = buffer_size
        else:
            self._buffer_size = AbstractDataset.default_buffer_size()

    def buffer_size(self):
        return (self._buffer_size)

    def load_train_dataset(self, batch_size):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load_validation_dataset(self, batch_size):
        raise NotImplementedError('Must be implemented by the subclass.')
