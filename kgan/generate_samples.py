from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import argparse

from kgan.models.GANFactory import GANFactory

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

    return (parser.parse_args(argv))


def main(args):

    print('creating the model - start')
    model_shape = args.model_shape[:3]
    gan = GANFactory.create(args.model, model_shape, args.latent_dimension)
    print('creating the model - end')

    print('loading the model - start')
    status = gan.load()
    print('loading the model - end')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(parse_arguments(sys.argv[1:]))
