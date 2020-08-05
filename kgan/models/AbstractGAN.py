from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.optimizers import Adam

import cv2


class AbstractGAN(object):

    __default_number_of_samples = 10

    __default_base_learning_rate = 0.0001
    __default_batch_size = 128

    __default_discriminator_number = 1
    __default_generator_number = 1

    __default_save_frequency = 1
    __default_loss_scan_frequency = 100

    @classmethod
    def default_number_of_samples(cls):
        return (cls.__default_number_of_samples)

    @classmethod
    def default_base_learning_rate(cls):
        return (cls.__default_base_learning_rate)

    @classmethod
    def default_batch_size(cls):
        return (cls.__default_batch_size)

    @classmethod
    def default_generator_number(cls):
        return (cls.__default_generator_number)

    @classmethod
    def default_discriminator_number(cls):
        return (cls.__default_discriminator_number)

    @classmethod
    def default_save_frequency(cls):
        return (cls.__default_save_frequency)

    @classmethod
    def default_loss_scan_frequency(cls):
        return (cls.__default_loss_scan_frequency)

    def __init__(self):
        self._current_step = 0
        self._summary_writer = None

        self._number_of_samples = AbstractGAN.default_number_of_samples()
        self._learning_rate = self._base_learning_rate = AbstractGAN.default_base_learning_rate(
        )
        self._batch_size = AbstractGAN.default_batch_size()

        self._discriminator_number = AbstractGAN.default_discriminator_number()
        self._generator_number = AbstractGAN.default_generator_number()
        self._cycle_number = self.discriminator_number(
        ) + self.generator_number()

        self._generator_optimizer = None
        self._discriminator_optimizer = None

    def cycle_number(self):
        return (self.discriminator_number() + self.generator_number())

    def current_step(self):
        return (self._current_step)

    def set_number_of_samples(self, number_of_samples):
        if (number_of_samples > 0):
            self._number_of_samples = number_of_samples
        else:
            self._number_of_samples = AbstractGAN.default_number_of_samples()

    def number_of_samples(self):
        return (self._number_of_samples)

    def set_base_learning_rate(self, base_learning_rate):
        if (base_learning_rate > 0):
            self._learning_rate = self._base_learning_rate = base_learning_rate
        else:
            self._learning_rate = self._base_learning_rate = AbstractDataset.default_base_learning_rate(
            )

    def base_learning_rate(self):
        return (self._base_learning_rate)

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

    def set_loss_scan_frequency(self, loss_scan_frequency):
        if (loss_scan_frequency > 0):
            self._loss_scan_frequency = loss_scan_frequency
        else:
            self._loss_scan_frequency = AbstractDataset.default_loss_scan_frequency(
            )

    def loss_scan_frequency(self):
        return (self._loss_scan_frequency)

    def set_generator_number(self, generator_number):
        if (generator_number > 0):
            self._generator_number = generator_number
        else:
            self._generator_number = AbstractDataset.default_generator_number()

    def generator_number(self):
        return (self._generator_number)

    def set_discriminator_number(self, discriminator_number):
        if (discriminator_number > 0):
            self._discriminator_number = discriminator_number
        else:
            self._discriminator_number = AbstractDataset.default_discriminator_number(
            )

    def discriminator_number(self):
        return (self._discriminator_number)

    def _create_generator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        return (optimizer)

    def _create_discriminator_optimizer(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        return (optimizer)

    def _create_models(self):
        status = True

        status = self._create_discriminator() and status
        status = self._create_generator() and status

        return (status)

    def _normalize_dataset(self, image, label):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_summary_writer(self, logdir='logs'):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _preprocess_train_dataset(self, dataset, batch_size):
        # Load train dataset split.
        train_dataset, number_of_batches = dataset.load_train_dataset(
            batch_size)

        # Normalize_ the dataset.
        train_dataset = train_dataset.map(self._normalize_dataset)

        return (train_dataset, number_of_batches)

    def train(self,
              dataset,
              batch_size,
              number_of_epochs,
              start_epoch=0,
              base_learning_rate=0.0001):
        status = True

        # Preprocess train dataset.
        train_dataset, number_of_batches = self._preprocess_train_dataset(
            dataset, batch_size)

        # Set parameters used for model training.
        self.set_base_learning_rate(base_learning_rate)
        self.set_batch_size(batch_size)

        # Create models.
        status = self._create_models() and status

        # Load model weights.
        status = self._load_weights() and status

        # Compute current step using start epoch and number of batches.
        self._current_step = start_epoch * number_of_batches + 1

        # Create summary writer.
        self._summary_writer = self._create_summary_writer()

        # Train the model starting from start_epoch upto number_of_epochs.
        for current_epoch in range(start_epoch, number_of_epochs):
            print('epoch', str(current_epoch), '- start')

            # Create generator optimizer with learning rate using current epoch.
            self._generator_optimizer = self._create_generator_optimizer(
                self.learning_rate())

            # Create discriminator optimizer with learning rate using current epoch.
            self._discriminator_optimizer = self._create_discriminator_optimizer(
                self.learning_rate())

            # Train the model for all batches in the train dataset split.
            for current_batch in train_dataset:
                # Train the model on current batch.
                current_losses = self._train_on_batch(current_batch)

                if self.loss_scan_frequency() and (
                        self.current_step() % self.loss_scan_frequency() == 0):
                    self._print_losses(current_losses)

                # Increment current step by 1.
                self._current_step = self._current_step + 1

            self._print_losses(current_losses)
            self._save_samples()

            if self.save_frequency() and (
                    current_epoch % self.save_frequency() == 0):
                self.save_models()

            # Update learning rate at end of each epoch.
            self._update_learning_rate(current_epoch, number_of_epochs)

            print('epoch', str(current_epoch), '- end')

        return (status)

    def _update_learning_rate(self, current_epoch, number_of_epochs):
        return (True)

    def _save_models(self):
        return (True)

    def save_models(self):
        print('saving models - start')
        self._save_models()
        print('saving models - end')

    def load(self):
        status = True

        status = self._create_models() and status
        status = self._load_weights() and status

        return (status)

    def generate_samples(self, generator_inputs):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_generator_inputs(self, number_of_samples):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _save_samples(self):
        print('generating samples - start')
        generator_inputs = self._create_generator_inputs(
            self.number_of_samples())

        generated_images = self.generate_samples(generator_inputs)
        for index, image in enumerate(generated_images):
            filename = 'image-' + str(index) + '.png'
            cv2.imwrite(filename, image)
        print('generating samples - end')

    def _print_losses(self, losses):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_discriminator(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _create_generator(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _train_on_batch(self, input_batch):
        raise NotImplementedError('Must be implemented by the subclass.')

    def _load_weights(self):
        raise NotImplementedError('Must be implemented by the subclass.')
