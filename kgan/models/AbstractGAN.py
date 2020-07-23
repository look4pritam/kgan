from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.optimizers import Adam


class AbstractGAN(object):

    __default_batch_size = 128

    @classmethod
    def default_batch_size(cls):
        return (cls.__default_batch_size)

    def __init__(self):
        self._batch_size = AbstractGAN.default_batch_size()

        self._generator_optimizer = None
        self._discriminator_optimizer = None

    def set_batch_size(self, batch_size):
        if (batch_size > 0):
            self._batch_size = batch_size
        else:
            self._batch_size = AbstractDataset.default_batch_size()

    def batch_size(self):
        return (self._batch_size)

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
                losses = self._train_on_batch(current_batch)
                batch_index = batch_index + 1
                if (batch_index % generation_frequency == 0):
                    self.save_generated()

            #print('generator loss -', losses['generator_loss'].numpy())
            #print('discriminator loss -', losses['discriminator_loss'].numpy())

        return (True)

    def generate(self, number_of_samples):
        raise NotImplementedError('Must be implemented by the subclass.')

    def save_generated(self, number_of_samples=10):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_discriminator(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_generator(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _train_on_batch(self, input_batch):
        raise NotImplementedError('Must be implemented by the subclass.')
