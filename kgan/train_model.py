from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import argparse

from kgan.models.GANFactory import GANFactory
from kgan.datasets.DatasetFactory import DatasetFactory


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        choices=GANFactory.models(),
        help='GAN model.',
        default=GANFactory.default_model())

    parser.add_argument(
        '--dataset',
        type=str,
        choices=DatasetFactory.datasets(),
        help='Dataset used for training the model.',
        default=DatasetFactory.default_dataset())

    parser.add_argument(
        '--latent_dimension',
        type=int,
        help='Latent dimension used for generating the image.',
        default=100)

    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate used for training the model.',
        default=0.0002)

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size used for training the model.',
        default=64)

    parser.add_argument(
        '--maximum_epochs',
        type=int,
        help='Maximum epochs used for training the model.',
        default=100)

    parser.add_argument(
        '--generation_frequency',
        type=int,
        help=
        'Sample generation frequency in terms of number of batches processed.',
        default=1000)

    return (parser.parse_args(argv))


def main(args):

    model_shape = (28, 28, 1)
    gan = GANFactory.create(args.model, model_shape, args.latent_dimension)
    gan.set_generation_frequency(args.generation_frequency)

    dataset = DatasetFactory.create(args.dataset)
    train_dataset, validation_dataset = dataset.load(args.batch_size)

    status = gan.train(train_dataset, args.batch_size, args.maximum_epochs,
                       args.learning_rate, validation_dataset)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(parse_arguments(sys.argv[1:]))
