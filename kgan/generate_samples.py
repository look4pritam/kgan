from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import cv2

from kgan.models.GANFactory import GANFactory

default_model_shape = [128, 128, 3]


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
        default=40)

    return (parser.parse_args(argv))


def main(args):

    print('creating the model - start')
    model_shape = args.model_shape[:3]
    gan = GANFactory.create(args.model, model_shape, args.latent_dimension)
    print('creating the model - end')

    print('loading the model - start')
    status = gan.load()
    print('loading the model - end')

    test_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    test_attributes = [
        -1., -1., 1., -1., -1., -1., -1., -1., -1., 1., -1., 1., -1., -1., -1.,
        4., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., 1., -1., -1., -1.,
        -1., 1., 1., -1., -1., -1., -1., -1., -1., 1.
    ]

    dataset_sample = [test_image, test_attributes]
    dataset_sample = gan.preprocess_sample(dataset_sample)
    generated_samples = gan.generate_samples(dataset_sample)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(parse_arguments(sys.argv[1:]))
