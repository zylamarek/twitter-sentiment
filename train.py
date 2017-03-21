"""
The script trains a LSTM neural network model with attention mechanism and convolution inputs on SemEval2017 Task 4A
data. Batches of sequences of characters (tweets) are input to the network. The model tries to predict the overall
sentiment (positive, neutral, negative) of each tweet. The network uses a softmax layer with 3 outputs, one for each
sentiment. The summed crossentropy is used as the training loss function. RMSProp algorithm is used for parameters
update.

Structure of the full model (DRP - dropout, DT - deep transition, DI - deep input, DO - deep output):
INPUT - DRP - CONVOLUTION - N*(- DRP - DI - LSTM(+/-DT) - DO) - ATTENTION - DENSE - SOFTMAX

Parameters:
    BATCH_SIZE - number of tweets in one training batch
    TOL - a small number added to some denominators to prevent division by zero errors, bringing numerical stability
    REC_NUM_UNITS - a list of numbers of units in each LSTM layer (input side first)
    DENSE_NUM_UNITS - a list of numbers of units in each dense layer (input side first)
    ATTENTION_NUM_UNITS - number of units in each attention layer
    ATTENTION_NUM_LAYERS - number of layers in the attention mechanism
    DROPOUT_FRACTION - probability of setting a connection to zero
    LEARNING_RATE - RMSProp parameter
    DECAY - LEARNING_RATE is multiplied by this factor in each epoch after NO_DECAY_EPOCHS
    NO_DECAY_EPOCHS - as above
    MAX_GRAD - gradient is limited to this value in LSTM layers
    NUM_EPOCHS - max number of training epochs
    SEED - random number generator seed (use to reproduce random results)
    INIT_RANGE - range of initial values of parameters
    EARLY_STOPPING - number of epochs without recall increase after which the training stops
    CONV_SIZES - a list containing sizes of convolution filters
    NUM_CONV_EACH - number of filters for each filter size
    CONV_NUM_LAYERS - number of stacked convolution layers
    DT_NUM_LAYERS - number of feed forward layers between consecutive LSTM hidden states (deep transition)
    DI_NUM_LAYERS - number of feed forward layers before LSTM input (deep input)
    DO_NUM_LAYERS - number of feed forward layers after LSTM output (deep output)
    PRINT_PROGRESS - print training, evaluation and data loading progress (a dot every 10% for each dataset)
    DATA_PATH - path to the directory containing the data (each file in this directory will get an id)
    CREATE_DATA_CHUNKS - reload all the data (first run with True and then set to False to speed up data loading)
    PARAMS_TO_LOAD - path to a file with parameters to load (use if you want to continue an experiment)
    DROPOUT_TYPE - {char, word, 1st_word_only} - which dropout to use; 1st_word_only uses word dropout right after input
        layer and char in the rest
    TRAIN_IDS - which datafiles use for training
    VALID_ID - which for validation
    TEST_ID - and testing (value to report for best validation result)
    EVAL_IDS - extra datasets to evaluate

"""

from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import os
import time
import lasagne
import scipy.io as sio
import cPickle as pickle
import logging
import argparse
import shutil
import sys

import data_wrapper
import attention
import word_dropout
import evaluation_helper
import lstm_dt_layer

# Parse settings
parser = argparse.ArgumentParser()

parser.add_argument('-BATCH_SIZE', type=int, default=150)
parser.add_argument('-TOL', type=float, default=1e-7)
parser.add_argument('-REC_NUM_UNITS', nargs='+', type=int, default=[219, 219])
parser.add_argument('-DENSE_NUM_UNITS', nargs='+', type=int, default=[219])
parser.add_argument('-ATTENTION_NUM_UNITS', type=int, default=219)
parser.add_argument('-ATTENTION_NUM_LAYERS', type=int, default=1)
parser.add_argument('-DROPOUT_FRACTION', type=float, default=0.2)
parser.add_argument('-LEARNING_RATE', type=float, default=1e-3)
parser.add_argument('-DECAY', type=float, default=0.99)
parser.add_argument('-NO_DECAY_EPOCHS', type=int, default=100)
parser.add_argument('-MAX_GRAD', type=float, default=5)
parser.add_argument('-NUM_EPOCHS', type=int, default=100)
parser.add_argument('-SEED', type=int, default=1234)
parser.add_argument('-INIT_RANGE', type=float, default=0.08)
parser.add_argument('-EARLY_STOPPING', type=int, default=10)
parser.add_argument('-CONV_SIZES', nargs='+', type=int, default=[3, 5, 7, 13, 21])
parser.add_argument('-NUM_CONV_EACH', type=int, default=10)
parser.add_argument('-CONV_NUM_LAYERS', type=int, default=1)
parser.add_argument('-DT_NUM_LAYERS', type=int, default=0)
parser.add_argument('-DI_NUM_LAYERS', type=int, default=0)
parser.add_argument('-DO_NUM_LAYERS', type=int, default=0)
parser.add_argument('-PRINT_PROGRESS', type=lambda x: x.lower() == 'true', default=True)
parser.add_argument('-DATA_PATH', default='data\\semeval_subA\\2017\\production')
parser.add_argument('-CREATE_DATA_CHUNKS', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('-PARAMS_TO_LOAD', default=None)
parser.add_argument('-DROPOUT_TYPE', choices=['char', 'word', '1st_word_only'], default='char')
parser.add_argument('-TRAIN_IDS', nargs='+', type=int, default=1)
parser.add_argument('-VALID_ID', type=int, default=2)
parser.add_argument('-TEST_ID', type=int, default=3)
parser.add_argument('-EVAL_IDS', nargs='+', type=int, default=None)

args = parser.parse_args()
np.random.seed(args.SEED)
INI = lasagne.init.Uniform(args.INIT_RANGE)

# Check for existence here, so you don't waste time on data loading
# and compilation just to get a silly error afterwards
if args.PARAMS_TO_LOAD is not None:
    if not os.path.isfile(args.PARAMS_TO_LOAD):
        print('Specified parameters file doesn\'t exist.')
        exit()

# Create output folder
folder_name = time.strftime('%Y.%m.%d-%H.%M.%S')
folder_name += '_semeval_%dLSTM' % len(args.REC_NUM_UNITS)
folder_name += '_'.join([str(rnu) for rnu in args.REC_NUM_UNITS])
folder_name += '_D' + '_'.join([str(dnu) for dnu in args.DENSE_NUM_UNITS])
folder_name += '_A%dx%d' % (args.ATTENTION_NUM_UNITS, args.ATTENTION_NUM_LAYERS)
folder_name += '_C%d' % args.CONV_NUM_LAYERS
folder_name += '_NCE%d' % args.NUM_CONV_EACH
folder_name += '_DT%d' % args.DT_NUM_LAYERS
folder_name += '_DI%d' % args.DI_NUM_LAYERS
folder_name += '_DO%d' % args.DO_NUM_LAYERS
folder_name += '_%.2f' % args.DROPOUT_FRACTION
folder_name += '_%.6f' % args.LEARNING_RATE
folder_name = os.path.join('output', folder_name)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Copy the script itself
shutil.copy(__file__, os.path.join(folder_name, os.path.basename(__file__)))

# Initialize logger
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(folder_name + '/train_output.log', mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# Log parameters
logger.info('Settings')
col_wid = max([len(name) for name, _ in args._get_kwargs()]) + 1
logger.info('Theano optimizer\t: '.expandtabs(col_wid) + str(theano.config.optimizer))
logger.info('folder_name\t: '.expandtabs(col_wid) + str(folder_name))
for name, value in args._get_kwargs():
    logger.info((name + '\t: ' + str(value)).expandtabs(col_wid))

# Get floatX type
floatX_dtype = np.dtype(theano.config.floatX).type

# Load data
logger.info('Loading data...')
t0 = time.time()
data_files = [_file for _file in os.listdir(args.DATA_PATH) if os.path.isfile(os.path.join(args.DATA_PATH, _file))]
data = data_wrapper.DataWrapper([os.path.join(args.DATA_PATH, file_name) for file_name in data_files],
                                train_ids=args.TRAIN_IDS, valid_id=args.VALID_ID,
                                test_id=args.TEST_ID, eval_ids=args.EVAL_IDS,
                                batch_size=args.BATCH_SIZE, temp_dir=os.path.join(args.DATA_PATH, 'temp'),
                                create_chunks=args.CREATE_DATA_CHUNKS, rng_seed=args.SEED,
                                shuffle_chunks_on_load=True, shuffle_in_chunks_on_load=True,
                                shuffle_batch_on_return=True, shuffle_in_chunk_on_chunk_reload=True,
                                print_progress=args.PRINT_PROGRESS
                                )
logger.info('Loading data took %.2fs' % (time.time() - t0))

# Log data properties
logger.info('-' * 80)
logger.info('DATA PROPERTIES')
logger.info('n_tweet     : ' + str(data.n_tweets))
logger.info('n_batches   : ' + str(data.n_batches))
logger.info('max_len     : ' + str(data.max_len))
logger.info('n_labels    : ' + str(data.n_labels))
logger.info('charset_size: ' + str(data.charset_size))
logger.info('charset     : ' + str(data.charset_map.keys()))
logger.info('-' * 80)

# Define Theano symbolic variables
sym_x = T.tensor3(dtype=theano.config.floatX)
sym_x_mask = T.matrix(dtype=theano.config.floatX)
sym_y = T.ivector()

# Build the model
t0 = time.time()

# Input
l_inp = lasagne.layers.InputLayer(shape=(args.BATCH_SIZE, data.max_len, data.charset_size), input_var=sym_x)
l_mask = lasagne.layers.InputLayer(shape=(args.BATCH_SIZE, data.max_len), input_var=sym_x_mask)

# Convolution layers
l_conv = l_inp
for _ in range(args.CONV_NUM_LAYERS):
    # Dropout
    if args.DROPOUT_TYPE == 'word' or args.DROPOUT_TYPE == '1st_word_only':
        l_conv = word_dropout.WordDropoutLayer(incoming=l_conv, word_input=l_inp, space=data.charset_map[' '],
                                               p=args.DROPOUT_FRACTION)
    else:
        l_conv = lasagne.layers.DropoutLayer(incoming=l_conv, p=args.DROPOUT_FRACTION)

    # Convolution
    l_sh = lasagne.layers.DimshuffleLayer(incoming=l_conv, pattern=(0, 2, 1))
    l_convs = [lasagne.layers.Conv1DLayer(incoming=l_sh, num_filters=args.NUM_CONV_EACH,
                                          filter_size=conv_size, stride=1, pad='same')
               for conv_size in args.CONV_SIZES]
    l_concat = lasagne.layers.ConcatLayer(incomings=l_convs, axis=1)
    l_conv = lasagne.layers.DimshuffleLayer(incoming=l_concat, pattern=(0, 2, 1))

# LSTM layers
l_lstm = l_conv
l_conv_num = int(np.prod(l_conv.output_shape[2:]))
for i_layer, rec_num_units in enumerate(args.REC_NUM_UNITS):
    only_return_final = args.ATTENTION_NUM_UNITS == 0 and i_layer == len(args.REC_NUM_UNITS) - 1

    # Dropout
    if args.DROPOUT_TYPE == 'word':
        l_lstm = word_dropout.WordDropoutLayer(incoming=l_lstm, word_input=l_inp, space=data.charset_map[' '],
                                               p=args.DROPOUT_FRACTION)
    else:
        l_lstm = lasagne.layers.DropoutLayer(incoming=l_lstm, p=args.DROPOUT_FRACTION)

    # Deep input
    for _ in range(args.DI_NUM_LAYERS):
        l_lstm = lasagne.layers.DenseLayer(incoming=l_lstm, num_leading_axes=2,
                                           num_units=l_conv_num if i_layer == 0 else args.REC_NUM_UNITS[i_layer - 1])

    # LSTM
    l_lstm = lstm_dt_layer.LSTMDTLayer(incoming=l_lstm, mask_input=l_mask, num_units=rec_num_units,
                                       ingate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
                                       forgetgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
                                       outgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
                                       cell=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=None,
                                                                nonlinearity=lasagne.nonlinearities.tanh),
                                       learn_init=True, precompute_input=True,
                                       grad_clipping=args.MAX_GRAD, only_return_final=only_return_final,
                                       num_dt_layers=args.DT_NUM_LAYERS)

    # Deep output
    for _ in range(args.DO_NUM_LAYERS):
        l_lstm = lasagne.layers.DenseLayer(incoming=l_lstm, num_units=args.REC_NUM_UNITS[i_layer], num_leading_axes=2)

# Attention
l_att = l_lstm
if args.ATTENTION_NUM_UNITS > 0 and args.ATTENTION_NUM_LAYERS > 0:
    if args.DROPOUT_TYPE == 'word':
        l_att = word_dropout.WordDropoutLayer(incoming=l_att, word_input=l_inp, space=data.charset_map[' '],
                                               p=args.DROPOUT_FRACTION)
    else:
        l_att = lasagne.layers.DropoutLayer(incoming=l_att, p=args.DROPOUT_FRACTION)
    l_att = attention.AttentionLayer(incoming=l_att, num_units=args.ATTENTION_NUM_UNITS, mask_input=l_mask,
                                      W=INI, v=INI, b=INI, num_att_layers=args.ATTENTION_NUM_LAYERS)

# Dense layer
l_dense = l_att
for dense_num_units in args.DENSE_NUM_UNITS:
    l_dense = lasagne.layers.DenseLayer(incoming=l_dense, num_units=dense_num_units)

# Softmax output
l_out = lasagne.layers.DenseLayer(incoming=l_dense, num_units=data.n_labels,
                                  nonlinearity=lasagne.nonlinearities.softmax)


# Define loss function
def cross_ent(net_output, target, mask):
    mask = T.sgn(T.sum(mask, axis=-1))
    net_output += args.TOL
    cost_ce = T.nnet.categorical_crossentropy(net_output, target)
    cost_ce = mask * cost_ce
    return T.sum(cost_ce)


# Define Theano variables for training and evaluation
train_out = lasagne.layers.get_output(l_out, deterministic=False)
cost_train = cross_ent(train_out, sym_y, sym_x_mask)

eval_out = lasagne.layers.get_output(l_out, deterministic=True)
cost_eval = cross_ent(eval_out, sym_y, sym_x_mask)

# Get all parameters of the network
all_params = lasagne.layers.get_all_params(l_out, trainable=True)

# Log the parameters
logger.info('-' * 80)
total_params = sum([p.get_value().size for p in all_params])
logger.info('#NETWORK params: %d' % total_params)
logger.info('Parameters:')
logger.info([{a.name, a.get_value().shape} for a in all_params])
logger.info('-' * 80)

# Get gradient
all_grads = T.grad(cost_train, all_params)

# Define update rules
sh_lr = theano.shared(lasagne.utils.floatX(args.LEARNING_RATE))
updates = lasagne.updates.rmsprop(all_grads, all_params, learning_rate=sh_lr)

# Define evaluation and train functions
fun_inp = [sym_x, sym_y, sym_x_mask]

logger.info('compiling f_eval...')
f_eval = theano.function(fun_inp, [cost_eval, eval_out], allow_input_downcast=True)

logger.info('compiling f_train...')
f_train = theano.function(fun_inp, [cost_train], updates=updates, allow_input_downcast=True)

logger.info('Building the model took %.2fs' % (time.time() - t0))

evaluation = evaluation_helper.EvaluationHelper(f_eval=f_eval, data=data, val_id=data.valid_id, test_id=data.test_id,
                                                eval_ids=data.train_ids + data.eval_ids,
                                                print_progress=args.PRINT_PROGRESS)

# Load network parameters
if args.PARAMS_TO_LOAD is not None:
    logger.info("Loading network params...")
    l_params = pickle.load(open(args.PARAMS_TO_LOAD, "rb"))
    if l_params.__len__() == all_params.__len__():
        logger.info('Numbers of variables the same.')
        f_shapes = True
        for i_l in range(l_params.__len__()):
            if l_params[i_l].get_value().shape == all_params[i_l].get_value().shape:
                all_params[i_l].set_value(l_params[i_l].get_value())
            else:
                f_shapes = False
        if f_shapes:
            logger.info('All shapes the same - the parameters were loaded.')
        else:
            logger.info('At least one shape incompatible - the parameters were not loaded correctly.')
    else:
        logger.info('Number of variables incompatible, nothing loaded.')

    logger.info('Evaluating loaded network... (compare the results to make sure the network was loaded properly)')
    evaluation.evaluate()
    for i_data, cm in zip(evaluation.eval_ids, evaluation.CM[-1]):
        logger.info('Confusion matrix %d - ' % i_data + data.dataset_names[i_data])
        logger.info(evaluation.conf_matrix_to_str(cm))

    logger.info('Crossentropy: ' + '/'.join(['%.6f' % x for x in evaluation.ce[-1]]))
    logger.info('Accuracy    : ' + '/'.join(['%.3f' % x for x in evaluation.acc[-1]]))
    logger.info('F^PN_1      : ' + '/'.join(['%.3f' % x for x in evaluation.F[-1]]))
    logger.info('Recall      : ' + '/'.join(['%.3f' % x for x in evaluation.R[-1]]))
    evaluation.reset()

# Main training loop
logger.info('Begin training...')

for epoch in range(args.NUM_EPOCHS):
    logger.info('>------------ Epoch: %d' % epoch)
    batch_time = time.time()

    # Train all the batches in training datasets
    for i_data in data.train_ids:
        if args.PRINT_PROGRESS:
            print('train ' + data.dataset_names[i_data] + ': ', end='')
        step = int(data.n_batches[i_data] / 10)
        data.set_current_data(i_data)
        for i_batch, (x_batch, x_mask_batch, y_batch) in enumerate(data):
            f_train(x_batch, y_batch, x_mask_batch)
            if args.PRINT_PROGRESS and not (i_batch + 1) % step:
                print('.', end='')
        if args.PRINT_PROGRESS:
            print('\n', end='')

    # Apply learning rate decay
    if epoch > (args.NO_DECAY_EPOCHS - 1):
        current_lr = sh_lr.get_value()
        sh_lr.set_value(lasagne.utils.floatX(current_lr * float(args.DECAY)))

    elapsed_train = time.time() - batch_time

    # Evaluate and log
    batch_time = time.time()

    evaluation.evaluate()
    for i_data, cm in zip(evaluation.eval_ids, evaluation.CM[-1]):
        logger.info('Confusion matrix %d - ' % i_data + data.dataset_names[i_data])
        logger.info(evaluation.conf_matrix_to_str(cm))

    logger.info('Crossentropy  : ' + '/'.join(['%.6f' % x for x in evaluation.ce[-1]]))
    logger.info('Accuracy      : ' + '/'.join(['%.3f' % x for x in evaluation.acc[-1]]))
    logger.info('F^PN_1        : ' + '/'.join(['%.3f' % x for x in evaluation.F[-1]]))
    logger.info('Recall        : ' + '/'.join(['%.3f' % x for x in evaluation.R[-1]]))

    elapsed_eval = time.time() - batch_time
    logger.info('Time elapsed  : %.0f + %.0f = %.0fs ~= %.1fmin' % (elapsed_train, elapsed_eval,
                                                                    elapsed_train + elapsed_eval,
                                                                    (elapsed_train + elapsed_eval) / 60.))
    eta = (elapsed_train + elapsed_eval) * (args.NUM_EPOCHS - epoch - 1) / 60.
    logger.info('ETA           : %.0fmin ~= %.1fh' % (eta, eta / 60.))
    eta_early = (elapsed_train + elapsed_eval) * (args.EARLY_STOPPING - epoch + evaluation.best_R[0]) / 60.
    logger.info('ETA early stop: %.0fmin ~= %.1fh' % (eta_early, eta_early / 60.))

    logger.info('Best valid R = %.3f with test R = %.3f in epoch %d.' % (evaluation.best_R[1], evaluation.best_R[2],
                                                                         evaluation.best_R[0]))

    # Store cross entropy, correct rates and F scores
    sio.savemat(os.path.join(folder_name, 'performance.mat'), evaluation.measurements)

    # Store parameters
    pickle.dump(all_params, open(folder_name + '/params_' + str(epoch) + '.p', 'wb'))

    # Early stopping
    if epoch - evaluation.best_R[0] == args.EARLY_STOPPING:
        logger.info('Validation set recall did not increase in last %d epochs. Stopping.' % args.EARLY_STOPPING)
        break

logger.info('done.')
