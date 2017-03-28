"""
The script trains a LSTM neural network model with attention mechanism and convolution inputs on SemEval2017 Task 4A
data. Batches of sequences of characters (tweets) are input to the network. The model tries to predict the overall
sentiment (positive, neutral, negative) of each tweet. The network uses a softmax layer with 3 outputs, one for each
sentiment. The summed crossentropy is used as the training loss function. RMSProp algorithm is used for parameters
update.

Structure of the full model (DRP - dropout, DT - deep transition, DI - deep input, DO - deep output):
INPUT - DRP - CONVOLUTION - MAXPOOL - N*(- DRP - DI - LSTM(+/-DT) - DO) - ATTENTION - DENSE - SOFTMAX

Run the script with --help for parameters description.

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
description = 'The script trains a LSTM neural network model with attention mechanism and convolution inputs.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('-BATCH_SIZE', type=int, default=150, help='number of tweets in one training batch')
parser.add_argument('-CONV_SIZES', nargs='+', type=int, default=[3, 5, 7, 13, 21],
                    help='a list containing sizes of convolution filters')
parser.add_argument('-CREATE_DATA_CHUNKS', action='store_const', const=True, default=False,
                    help='reload all the data (first run with the flag set and then without to speed up data loading)')
parser.add_argument('-DATA_PATH', default='data\\semeval_subA\\2017\\production',
                    help='path to the directory containing the data (each file in this directory will get an id)')
parser.add_argument('-DECAY', type=float, default=0.99,
                    help='LEARNING_RATE is multiplied by DECAY in each epoch after NO_DECAY_EPOCHS')
parser.add_argument('-CONV_STRIDE_2', action='store_const', const=True, default=False,
                    help='uses stride=2 in each convolution layer, except the first')
parser.add_argument('-DROPOUT_FRACTION', type=float, default=0.2, help='probability of setting a connection to zero')
parser.add_argument('-DROPOUT_TYPE', choices=['char', 'word', '1st_word_only'], default='char',
                    help='which dropout to use; 1st_word_only uses word dropout right after input layer and char in \
                    the rest')
parser.add_argument('-EARLY_STOPPING', type=int, default=10,
                    help='number of epochs without recall increase after which the training stops')
parser.add_argument('-EVAL_IDS', nargs='+', type=int, default=None, help='extra datasets to evaluate')
parser.add_argument('-INIT_RANGE', type=float, default=0.08, help='range of initial values of parameters')
parser.add_argument('-LEARNING_RATE', type=float, default=1e-3, help='RMSProp parameter')
parser.add_argument('-MAX_GRAD', type=float, default=5, help='gradient is limited to this value in LSTM layers')
parser.add_argument('-NO_DECAY_EPOCHS', type=int, default=100,
                    help='LEARNING_RATE is multiplied by DECAY in each epoch after NO_DECAY_EPOCHS')
parser.add_argument('-NUM_CONV_EACH', type=int, default=10, help='number of convolution filters for each filter size')
parser.add_argument('-NUM_EPOCHS', type=int, default=100, help='max number of training epochs')
parser.add_argument('-NUM_LAYERS_ATTENTION', type=int, default=1,
                    help='number of stacked dense layers in the attention mechanism')
parser.add_argument('-NUM_LAYERS_CONV', type=int, default=1, help='number of stacked convolution layers')
parser.add_argument('-NUM_LAYERS_DENSE', type=int, default=1, help='number of stacked dense layers')
parser.add_argument('-NUM_LAYERS_DI', type=int, default=0,
                    help='number of stacked feed forward layers before LSTM input (deep input)')
parser.add_argument('-NUM_LAYERS_DT', type=int, default=1,
                    help='number of stacked feed forward layers between consecutive LSTM hidden states (deep \
                    transition)')
parser.add_argument('-NUM_LAYERS_DO', type=int, default=0,
                    help='number of stacked feed forward layers after LSTM output (deep output)')
parser.add_argument('-NUM_LAYERS_LSTM', type=int, default=3,
                    help='number of stacked LSTM layers')
parser.add_argument('-NUM_LAYERS_MAXPOOL', type=int, default=0,
                    help='number of stacked maxpool layers')
parser.add_argument('-NUM_UNITS', type=int, default=219, help='number of units in each layer')
parser.add_argument('-PARAMS_TO_LOAD', default=None,
                    help='path to a file with parameters to load (use if you want to continue an experiment)')
parser.add_argument('-SEED', type=int, default=1234,
                    help='random number generator seed (use to reproduce random results)')
parser.add_argument('-SHOW_MODEL', action='store_const', const=True, default=False,
                    help='lists all the layers and their output sizes')
parser.add_argument('-SUPPRESS_PRINT_PROGRESS', action='store_const', const=True, default=False,
                    help='if not set, training, evaluation and data loading prints a dot every 10%% for each dataset')
parser.add_argument('-TEST_ID', type=int, default=3,
                    help='datafiles to be used for testing (value to report for best validation result)')
parser.add_argument('-TOL', type=float, default=1e-7,
                    help='a small number added to some denominators to prevent division by zero errors, bringing \
                    numerical stability')
parser.add_argument('-TRAIN_IDS', nargs='+', type=int, default=1, help='datafiles to be used for training')
parser.add_argument('-VALID_ID', type=int, default=2, help='datafiles to be used for validation')

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
folder_name += '_semeval'
folder_name += '_%d' % args.NUM_UNITS
folder_name += '_%dC%d' % (args.NUM_LAYERS_CONV, args.NUM_CONV_EACH)
if args.CONV_STRIDE_2:
    folder_name += 's2'
folder_name += '_MP%d' % args.NUM_LAYERS_MAXPOOL
folder_name += '_DI%d' % args.NUM_LAYERS_DI
folder_name += '_%dLSTM' % args.NUM_LAYERS_LSTM
folder_name += '_DT%d' % args.NUM_LAYERS_DT
folder_name += '_DO%d' % args.NUM_LAYERS_DO
folder_name += '_A%d' % args.NUM_LAYERS_ATTENTION
folder_name += '_D%d' % args.NUM_LAYERS_DENSE
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
                                print_progress=not args.SUPPRESS_PRINT_PROGRESS
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
for i_layer in range(args.NUM_LAYERS_CONV):
    # Dropout
    if args.DROPOUT_TYPE == 'word' or args.DROPOUT_TYPE == '1st_word_only':
        l_conv = word_dropout.WordDropoutLayer(incoming=l_conv, word_input=l_inp, space=data.charset_map[' '],
                                               p=args.DROPOUT_FRACTION)
    else:
        l_conv = lasagne.layers.DropoutLayer(incoming=l_conv, p=args.DROPOUT_FRACTION)

    if i_layer > 0 and args.CONV_STRIDE_2:
        conv_stride = 2
    else:
        conv_stride = 1

    # Convolution
    l_sh = lasagne.layers.DimshuffleLayer(incoming=l_conv, pattern=(0, 2, 1))
    l_convs = [lasagne.layers.Conv1DLayer(incoming=l_sh, num_filters=args.NUM_CONV_EACH,
                                          filter_size=conv_size, stride=conv_stride, pad='same')
               for conv_size in args.CONV_SIZES]
    l_concat = lasagne.layers.ConcatLayer(incomings=l_convs, axis=1)
    l_conv = lasagne.layers.DimshuffleLayer(incoming=l_concat, pattern=(0, 2, 1))

    # Shorten the mask
    if conv_stride > 1:
        l_mask = lasagne.layers.DimshuffleLayer(incoming=l_mask, pattern=(0, 'x', 1))
        l_mask = lasagne.layers.MaxPool1DLayer(l_mask, 2, stride=conv_stride, pad=l_mask.output_shape[2] % conv_stride)
        l_mask = lasagne.layers.DimshuffleLayer(incoming=l_mask, pattern=(0, 2))

# Max pool layers
l_mp = l_conv
for _ in range(args.NUM_LAYERS_MAXPOOL):
    l_mp = lasagne.layers.DimshuffleLayer(incoming=l_mp, pattern=(0, 2, 1))
    l_mp = lasagne.layers.MaxPool1DLayer(l_mp, 2, stride=2, pad=l_mp.output_shape[2] % 2)
    l_mp = lasagne.layers.DimshuffleLayer(incoming=l_mp, pattern=(0, 2, 1))

    l_mask = lasagne.layers.DimshuffleLayer(incoming=l_mask, pattern=(0, 'x', 1))
    l_mask = lasagne.layers.MaxPool1DLayer(l_mask, 2, stride=2, pad=l_mask.output_shape[2] % 2)
    l_mask = lasagne.layers.DimshuffleLayer(incoming=l_mask, pattern=(0, 2))

# LSTM layers
l_lstm = l_mp
l_conv_num = int(np.prod(l_conv.output_shape[2:]))
for i_layer in range(args.NUM_LAYERS_LSTM):
    only_return_final = args.NUM_LAYERS_ATTENTION == 0 and i_layer == args.NUM_LAYERS_LSTM - 1

    # Dropout
    if args.DROPOUT_TYPE == 'word':
        l_lstm = word_dropout.WordDropoutLayer(incoming=l_lstm, word_input=l_inp, space=data.charset_map[' '],
                                               p=args.DROPOUT_FRACTION)
    else:
        l_lstm = lasagne.layers.DropoutLayer(incoming=l_lstm, p=args.DROPOUT_FRACTION)

    # Deep input
    for _ in range(args.NUM_LAYERS_DI):
        l_lstm = lasagne.layers.DenseLayer(incoming=l_lstm, num_leading_axes=2,
                                           num_units=l_conv_num if i_layer == 0 else args.NUM_UNITS)

    # LSTM
    l_lstm = lstm_dt_layer.LSTMDTLayer(incoming=l_lstm, mask_input=l_mask, num_units=args.NUM_UNITS,
                                       ingate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
                                       forgetgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
                                       outgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
                                       cell=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=None,
                                                                nonlinearity=lasagne.nonlinearities.tanh),
                                       learn_init=True, precompute_input=True,
                                       grad_clipping=args.MAX_GRAD, only_return_final=only_return_final,
                                       num_dt_layers=args.NUM_LAYERS_DT)

    # Deep output
    for _ in range(args.NUM_LAYERS_DO):
        l_lstm = lasagne.layers.DenseLayer(incoming=l_lstm, num_units=args.NUM_UNITS, num_leading_axes=2)

# Attention
l_att = l_lstm
if args.NUM_LAYERS_ATTENTION > 0:
    if args.DROPOUT_TYPE == 'word':
        l_att = word_dropout.WordDropoutLayer(incoming=l_att, word_input=l_inp, space=data.charset_map[' '],
                                              p=args.DROPOUT_FRACTION)
    else:
        l_att = lasagne.layers.DropoutLayer(incoming=l_att, p=args.DROPOUT_FRACTION)
    l_att = attention.AttentionLayer(incoming=l_att, num_units=args.NUM_UNITS, mask_input=l_mask,
                                     W=INI, v=INI, b=INI, num_att_layers=args.NUM_LAYERS_ATTENTION)

# Dense layer
l_dense = l_att
for _ in range(args.NUM_LAYERS_DENSE):
    l_dense = lasagne.layers.DenseLayer(incoming=l_dense, num_units=args.NUM_UNITS)

# Softmax output
l_out = lasagne.layers.DenseLayer(incoming=l_dense, num_units=data.n_labels,
                                  nonlinearity=lasagne.nonlinearities.softmax)

if args.SHOW_MODEL:
    for layer in lasagne.layers.get_all_layers(l_out):
        logger.info(str(layer.output_shape) + ' ' + layer.__class__.__name__)

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

# Define update rules
sh_lr = theano.shared(lasagne.utils.floatX(args.LEARNING_RATE))
updates = lasagne.updates.rmsprop(cost_train, all_params, learning_rate=sh_lr)

# Define evaluation and train functions
fun_inp = [sym_x, sym_y, sym_x_mask]

logger.info('compiling f_eval...')
f_eval = theano.function(fun_inp, [cost_eval, eval_out], allow_input_downcast=True)

logger.info('compiling f_train...')
f_train = theano.function(fun_inp, [], updates=updates, allow_input_downcast=True)

logger.info('Building the model took %.2fs' % (time.time() - t0))

evaluation = evaluation_helper.EvaluationHelper(f_eval=f_eval, data=data, val_id=data.valid_id, test_id=data.test_id,
                                                eval_ids=data.train_ids + data.eval_ids,
                                                print_progress=not args.SUPPRESS_PRINT_PROGRESS)

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
        if not args.SUPPRESS_PRINT_PROGRESS:
            print('train ' + data.dataset_names[i_data] + ': ', end='')
        step = max(data.n_batches[i_data] // 10 + 1, 1)
        offset = step * 10 - data.n_batches[i_data] + 1
        if not args.SUPPRESS_PRINT_PROGRESS and data.n_batches[i_data] < 10:
            print('.' * (10 - data.n_batches[i_data]), end='')
        data.set_current_data(i_data)
        for i_batch, (x_batch, x_mask_batch, y_batch) in enumerate(data):
            f_train(x_batch, y_batch, x_mask_batch)
            if not args.SUPPRESS_PRINT_PROGRESS and not (i_batch + offset) % step:
                print('.', end='')
        if not args.SUPPRESS_PRINT_PROGRESS:
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
