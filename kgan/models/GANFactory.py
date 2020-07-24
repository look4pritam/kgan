from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.SimpleGAN import SimpleGAN
from kgan.models.DCGAN import DCGAN
from kgan.models.WGANGP import WGANGP


class GANFactory(object):

    __models = []

    def __init__(self):
        pass

    @classmethod
    def models(cls):
        if (len(cls.__models) == 0):
            cls.__models.append(SimpleGAN.name())
            cls.__models.append(DCGAN.name())
            cls.__models.append(WGANGP.name())

        return (cls.__models)

    @classmethod
    def default_model(cls):
        return (SimpleGAN.name())

    @classmethod
    def create(cls, name, input_shape, latent_dimension):
        gan = None

        if (name == SimpleGAN.name()):
            gan = SimpleGAN(input_shape, latent_dimension)
        elif (name == DCGAN.name()):
            gan = DCGAN(input_shape, latent_dimension)
        elif (name == WGANGP.name()):
            gan = WGANGP(input_shape, latent_dimension)
        else:
            gan = SimpleGAN(input_shape, latent_dimension)

        return (gan)
