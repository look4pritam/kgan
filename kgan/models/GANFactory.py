from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.GAN import GAN
from kgan.models.DCGAN import DCGAN
from kgan.models.WGANGP import WGANGP
from kgan.models.AttGAN import AttGAN


class GANFactory(object):

    __models = []

    def __init__(self):
        pass

    @classmethod
    def models(cls):
        if (len(cls.__models) == 0):
            cls.__models.append(GAN.name())
            cls.__models.append(DCGAN.name())
            cls.__models.append(WGANGP.name())
            cls.__models.append(AttGAN.name())

        return (cls.__models)

    @classmethod
    def default_model(cls):
        return (GAN.name())

    @classmethod
    def create(cls, name, input_shape, latent_dimension):
        gan = None

        if (name == GAN.name()):
            gan = GAN(input_shape, latent_dimension)
        elif (name == DCGAN.name()):
            gan = DCGAN(input_shape, latent_dimension)
        elif (name == WGANGP.name()):
            gan = WGANGP(input_shape, latent_dimension)
        elif (name == AttGAN.name()):
            gan = AttGAN(input_shape, latent_dimension)
        else:
            gan = GAN(input_shape, latent_dimension)

        return (gan)
