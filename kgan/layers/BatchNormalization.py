from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras.layers as klayers


class BatchNormalization(klayers.Layer):
    def __init__(self, is_training=False):
        super(BatchNormalization, self).__init__()
        self.bn = klayers.BatchNormalization(
            epsilon=1e-5, momentum=0.9, scale=True, trainable=is_training)

    def call(self, inputs, training):
        x = self.bn(inputs, training=training)
        return x
