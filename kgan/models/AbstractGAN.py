from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.optimizers import Adam


class AbstractGAN(object):

    __default_batch_size = 128

    __default_generation_frequency = 1000
    __default_loss_scan_frequency = 1000

    @classmethod
    def default_batch_size(cls):
        return (cls.__default_batch_size)

    @classmethod
    def default_generation_frequency(cls):
        return (cls.__default_generation_frequency)

    @classmethod
    def default_loss_scan_frequency(cls):
        return (cls.__default_loss_scan_frequency)

    def __init__(self):
        self._batch_size = AbstractGAN.default_batch_size()

        self._generation_frequency = AbstractGAN.default_generation_frequency()
        self._loss_scan_frequency = AbstractGAN.default_loss_scan_frequency()

        self._generator_optimizer = None
        self._discriminator_optimizer = None

    def set_batch_size(self, batch_size):
        if (batch_size > 0):
            self._batch_size = batch_size
        else:
            self._batch_size = AbstractDataset.default_batch_size()

    def batch_size(self):
        return (self._batch_size)

    def set_generation_frequency(self, generation_frequency):
        if (generation_frequency > 0):
            self._generation_frequency = generation_frequency
        else:
            self._generation_frequency = AbstractDataset.default_generation_frequency(
            )

    def generation_frequency(self):
        return (self._generation_frequency)

    def set_loss_scan_frequency(self, loss_scan_frequency):
        if (loss_scan_frequency > 0):
            self._loss_scan_frequency = loss_scan_frequency
        else:
            self._loss_scan_frequency = AbstractDataset.default_loss_scan_frequency(
            )

    def loss_scan_frequency(self):
        return (self._loss_scan_frequency)

    def _create_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        return (optimizer)

    def train(self,
              train_dataset,
              batch_size,
              epochs,
              learning_rate=0.0001,
              validation_dataset=None):

        self.set_batch_size(batch_size)
        generation_frequency = 100

        self._generator_optimizer = self._create_optimizer(learning_rate)
        self._discriminator_optimizer = self._create_optimizer(learning_rate)

        batch_index = 0
        for current_epoch in range(epochs):
            for current_batch in train_dataset:

                current_losses = self._train_on_batch(current_batch)
                batch_index = batch_index + 1

                if self.generation_frequency() and (
                        batch_index % self.generation_frequency() == 0):
                    self.save_generated()
                    print('generated samples at', str(batch_index))

                if self.loss_scan_frequency() and (
                        batch_index % self.loss_scan_frequency() == 0):
                    print('current loss values at', str(batch_index))
                    self._print_losses(current_losses)

        return (True)

    def generate(self, number_of_samples):
        raise NotImplementedError('Must be implemented by the subclass.')

    def save_generated(self, number_of_samples=10):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _print_losses(self, losses):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_discriminator(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_generator(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _train_on_batch(self, input_batch):
        raise NotImplementedError('Must be implemented by the subclass.')