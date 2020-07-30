from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import argparse

from kgan.models.GANFactory import GANFactory
from kgan.datasets.DatasetFactory import DatasetFactory

default_model_shape = [28, 28, 1]


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
        '--model_shape',
        nargs='+',
        type=int,
        help='Input shape used for training the model.',
        default=default_model_shape)

    parser.add_argument(
        '--latent_dimension',
        type=int,
        help='Latent dimension used for generating samples.',
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
        '--start_epoch',
        type=int,
        help='Starting epoch used for training the model.',
        default=0)

    parser.add_argument(
        '--discriminator_number',
        type=int,
        help=
        'Number of times discriminator model is updated while training the model.',
        default=1)

    parser.add_argument(
        '--generator_number',
        type=int,
        help=
        'Number of times generator model is updated while training the model.',
        default=1)

    parser.add_argument(
        '--save_frequency',
        type=int,
        help='Model saving frequency in terms of number of epochs processed.',
        default=1)

    return (parser.parse_args(argv))


def main(args):

    print('creating the model - start')
    model_shape = args.model_shape[:3]
    gan = GANFactory.create(args.model, model_shape, args.latent_dimension)

    gan.set_discriminator_number(args.discriminator_number)
    gan.set_generator_number(args.generator_number)

    gan.set_save_frequency(args.save_frequency)
    print('creating the model - end')

    print('processing the dataset - start')
    dataset = DatasetFactory.create(args.dataset)
    print('processing the dataset - end')

    print('training the model - start')
    status = gan.train(dataset, args.batch_size, args.maximum_epochs,
                       args.start_epoch, args.learning_rate)
    print('training the model - end')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(parse_arguments(sys.argv[1:]))
