from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.optimizers import Adam


class AbstractGAN(object):

    __default_number_of_samples = 10

    __default_learning_rate = 0.0001
    __default_batch_size = 128

    __default_generation_frequency = 1000
    __default_loss_scan_frequency = 1000

    __default_save_frequency = 1

    @classmethod
    def default_number_of_samples(cls):
        return (cls.__default_number_of_samples)

    @classmethod
    def default_learning_rate(cls):
        return (cls.__default_learning_rate)

    @classmethod
    def default_batch_size(cls):
        return (cls.__default_batch_size)

    @classmethod
    def default_generation_frequency(cls):
        return (cls.__default_generation_frequency)

    @classmethod
    def default_loss_scan_frequency(cls):
        return (cls.__default_loss_scan_frequency)

    @classmethod
    def default_save_frequency(cls):
        return (cls.__default_save_frequency)

    def __init__(self):
        self._number_of_samples = AbstractGAN.default_number_of_samples()
        self._learning_rate = AbstractGAN.default_learning_rate()
        self._batch_size = AbstractGAN.default_batch_size()

        self._generation_frequency = AbstractGAN.default_generation_frequency()
        self._loss_scan_frequency = AbstractGAN.default_loss_scan_frequency()

        self._generator_optimizer = None
        self._discriminator_optimizer = None

    def set_number_of_samples(self, number_of_samples):
        if (number_of_samples > 0):
            self._number_of_samples = number_of_samples
        else:
            self._number_of_samples = AbstractGAN.default_number_of_samples()

    def number_of_samples(self):
        return (self._number_of_samples)

    def set_learning_rate(self, learning_rate):
        if (learning_rate > 0):
            self._learning_rate = learning_rate
        else:
            self._learning_rate = AbstractDataset.default_learning_rate()

    def learning_rate(self):
        return (self._learning_rate)

    def set_batch_size(self, batch_size):
        if (batch_size > 0):
            self._batch_size = batch_size
        else:
            self._batch_size = AbstractDataset.default_batch_size()

    def batch_size(self):
        return (self._batch_size)

    def set_save_frequency(self, save_frequency):
        if (save_frequency > 0):
            self._save_frequency = save_frequency
        else:
            self._save_frequency = AbstractDataset.default_save_frequency()

    def save_frequency(self):
        return (self._save_frequency)

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

    def _create_generator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        return (optimizer)

    def _create_models(self):
        self._discriminator = self._create_discriminator()
        self._discriminator.summary()

        self._generator = self._create_generator()
        self._generator.summary()
        return (True)

    def _normalize_dataset(self, image, label):
        raise NotImplementedError('Must be implemented by the subclass.')

    def train(self, train_dataset, batch_size, epochs, learning_rate=0.0001):
        status = True

        train_dataset = train_dataset.map(self._normalize_dataset)
        self.set_learning_rate(learning_rate)
        self.set_batch_size(batch_size)
        status = self._create_models() and status

        batch_index = 0
        for current_epoch in range(epochs):

            # Create optimizers with learning rate for each epoch.
            self._generator_optimizer = self._create_generator_optimizer(
                self.learning_rate())

            self._discriminator_optimizer = self._create_discriminator_optimizer(
                self.learning_rate())

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

            if self.save_frequency() and (
                    current_epoch % self.save_frequency() == 0):
                self.save_models()
                print('models are saved at', str(current_epoch))

            # Update learning rate at end of each epoch.
            self._update_learning_rate()

        return (True)

    def _update_learning_rate(self):
        return (True)

    def save_models(self):
        pass

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
