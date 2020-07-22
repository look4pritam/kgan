from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.SimpleGAN import SimpleGAN


class GANFactory(object):
    def __init__(self):
        pass

    @classmethod
    def create(cls, name, input_shape, latent_dimension):
        gan = None

        if (name == SimpleGAN.name()):
            gan = SimpleGAN(input_shape, latent_dimension)
        else:
            gan = SimpleGAN(input_shape, latent_dimension)

        return (gan)
