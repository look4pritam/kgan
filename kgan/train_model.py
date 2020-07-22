from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.GANFactory import GANFactory
from kgan.datasets.DatasetFactory import DatasetFactory

input_shape = (28, 28, 1)
latent_dimension = 100

gan = GANFactory.create('gan', input_shape, latent_dimension)

batch_size = 64
dataset = DatasetFactory.create('mnist')
train_dataset, validation_dataset = dataset.load(input_shape, batch_size)

epochs = 10
learning_rate = 0.0002
status = gan.train(train_dataset, batch_size, epochs, learning_rate,
                   validation_dataset)
print(status)
