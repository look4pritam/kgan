from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kgan.models.WGANGP import WGANGP

from kgan.models.discriminators.AttGANDiscriminator import AttGANDiscriminator
from kgan.models.generators.ConvolutionalGenerator import ConvolutionalGenerator


class AttGAN(WGANGP):
    @classmethod
    def name(cls):
        return ('attgan')

    def __init__(self, input_shape, latent_dimension):
        super(AttGAN, self).__init__(input_shape, latent_dimension)
        pass

    def _create_discriminator(self):
        discriminator = AttGANDiscriminator.create(self.input_shape())
        return (discriminator)

    def _create_generator(self):
        generator = ConvolutionalGenerator.create(self.latent_dimension())
        return (generator)
