from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.GAN import GAN


class ModelFactory(object):
    def __init__(self):
        pass

    @classmethod
    def create_discriminator(cls, name, input_shape):
        discriminator = None
        if (name == GAN.name()):
            discriminator = GAN.create_discriminator(input_shape)
        else:
            discriminator = GAN.create_discriminator(input_shape)

        return (discriminator)

    @classmethod
    def create_generator(cls, name, noise_shape, input_shape):
        generator = None
        if (name == GAN.name()):
            generator = GAN.create_generator(noise_shape, input_shape)
        else:
            generator = GAN.create_generator(noise_shape, input_shape)

        return (generator)
