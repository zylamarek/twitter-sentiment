"""
The BlockLayer

"""

import numpy as np
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init

from lasagne.layers import MergeLayer

__all__ = [
    "BlockLayer"
]


class BlockLayer(MergeLayer):
    def __init__(self, incoming, control_incoming, num_units, mask_input=None, W=init.GlorotUniform(),
                 v=init.GlorotUniform(), b=init.Constant(0.), nonlinearity=nonlinearities.tanh,
                 **kwargs):

        incomings = [incoming, control_incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 2

        super(BlockLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)

        self.num_units = num_units

        control_shape = self.input_shapes[1]
        num_inputs = int(np.prod(control_shape[2:]))

        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.v = self.add_param(v, (num_units, 1), name='v')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        original_shape = input.shape
        in_len = input.shape[0] * input.shape[1]

        control_input = inputs[1]

        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # reshape inputs
        input = input.reshape((in_len, input.shape[2]))
        control_input = control_input.reshape((in_len, control_input.shape[2]))

        # apply mask
        if mask is not None:
            mask = mask.reshape((in_len, 1))
            input *= mask
            control_input *= mask

        # compute g[(W*x_c+b)*v]
        control_input = T.dot(control_input, self.W)
        if self.b is not None:
            control_input = control_input + self.b.dimshuffle('x', 0)
        control_input = T.dot(control_input, self.v)
        c_activation = self.nonlinearity(control_input)

        # apply block
        blocked = T.switch(T.lt(c_activation, 0),
                           T.zeros((in_len, original_shape[2])),
                           input)

        blocked = blocked.reshape(original_shape)
        return blocked
