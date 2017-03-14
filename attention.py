"""
The AttentionLayer implements attention mechanism, as in
Soren Kaae Sonderby, Casper Kaae Sonderby, Henrik Nielsen, Ole Winther,
Convolutional LSTM Networks for Subcellular Localization of Proteins, 2015,
https://arxiv.org/abs/1503.01919

"""

import numpy as np
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init

from lasagne.layers import MergeLayer

__all__ = [
    "AttentionLayer"
]


class AttentionLayer(MergeLayer):
    def __init__(self, incoming, num_units, mask_input=None, W=init.GlorotUniform(),
                 v=init.GlorotUniform(), b=init.Constant(0.), nonlinearity=nonlinearities.tanh,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1

        super(AttentionLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)

        self.num_units = num_units

        input_shape = self.input_shapes[0]
        num_inputs = int(np.prod(input_shape[2:]))

        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.v = self.add_param(v, (num_units, 1), name='v')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[2]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        original_shape = input.shape

        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # reshape input
        input = input.reshape((input.shape[0] * input.shape[1], input.shape[2]))

        # apply mask
        if mask is not None:
            mask = mask.reshape((mask.shape[0] * mask.shape[1], 1))
            input *= mask

        # compute g(W*x+b)*v
        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        activation = self.nonlinearity(activation)
        activation = T.dot(activation, self.v)

        # apply softmax - acquiring attention weights for each letter in each tweet
        activation = activation.reshape((original_shape[0], original_shape[1]))
        attention_w = nonlinearities.softmax(activation)
        attention_w = attention_w.reshape((original_shape[0] * original_shape[1], 1))

        # get weighted sum of each hidden state according to attention weights
        context = input * attention_w
        context = context.reshape(original_shape)
        context = T.sum(context, axis=1)

        return context
