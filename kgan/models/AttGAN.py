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

    def _train_on_batch(self, input_batch):

        if ((self.current_step() % self.cycle_number()) == 0):
            # Update generator weights.
            generator_loss = self._update_generator(input_batch)
            return {'generator': generator_loss}
        else:
            # Update discriminator weights.
            discriminator_loss = self._update_discriminator(input_batch)
            return {'discriminator': discriminator_loss}
