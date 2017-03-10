"""
Word dropout layer

This layer should be used with character input. It uses two incoming layers:
incoming - the previous layer, to which dropout is to be applied;
word_input - the original input; it is used to determine positions of words, together with the space code.

Regular dropout would set each input to zero with some probability p. This layer sets groups of inputs (each group
corresponding to a word in the original input) to zero, each with some probability p.

"""

import theano
import theano.tensor as T

from lasagne.layers import MergeLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng

__all__ = [
    "WordDropoutLayer"
]


class WordDropoutLayer(MergeLayer):
    def __init__(self, incoming, word_input, space, p=0.5, rescale=True, **kwargs):

        incomings = [incoming, word_input]
        super(WordDropoutLayer, self).__init__(incomings, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.space = space

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input = inputs[0]
        if deterministic or self.p == 0:
            return input
        else:
            # Get words positions
            word = inputs[1]
            word = T.argmax(word, axis=-1)
            word = T.neq(self.space, word).astype(dtype=input.dtype)

            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p

            # Rescale input
            if self.rescale:
                input /= retain_prob

            batch_size = input.shape[0]

            mask_prev_init = T.zeros((batch_size, ), dtype=input.dtype)
            word_prev_init = T.zeros((batch_size, ), dtype=input.dtype)

            # Define step function that samples dropout only for new words
            def step(word_n, mask_prev, word_prev, *args):
                mask = T.switch(T.eq(word_n, 0),
                                T.ones((batch_size, )),
                                T.switch(T.eq(word_prev, 1),
                                         mask_prev,
                                         self._srng.binomial((batch_size,), p=retain_prob, dtype=input.dtype)))
                return mask, word_n

            word = word.dimshuffle(1, 0)

            # Iterate through the sequences and apply the step function
            mask_out, word_out = theano.scan(fn=step, sequences=[word],
                                             outputs_info=[mask_prev_init, word_prev_init])[0]

            mask_out = mask_out.dimshuffle(1, 0, 'x')
            return input * mask_out
