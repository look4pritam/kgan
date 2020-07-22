from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.GAN import GAN


class ModelFactory(object):
    def __init__(self):
        pass

    @classmethod
    def create_gan(cls, name, input_shape, latent_shape):
        gan = None

        if (name == GAN.name()):
            gan = GAN(input_shape, latent_shape)
        else:
            gan = GAN(input_shape, latent_shape)

        return (gan)
