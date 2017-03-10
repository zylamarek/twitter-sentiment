""" Data wrapper class

Manages loading data and prepping batches. Dynamically stores and loads chunks of data to disk to support datasets
larger than the available RAM. Uses numpy for saving and loading data, as it is much faster than pickle and allows
compression.

Parameters:
    data_path - str or list - path to file(s) containing data
    train_ids - int or list - ids specifying datasets used for training, may be one or multiple files
    valid_id - int - id specifying dataset used for validation, has to be one file only
    test_id - int or None - id specifying dataset used for testing, may be one file or none
    eval_ids - int or list or None - ids specifying datasets that are used for neither of the above purposes, but just
        evaluation
    create_chunks - bool - if False, does not load the data from the files specified, but from temporary chunks. If
        chunks do not exist the program fails. Use it to speed up loading huge datasets.
    chunk_size - int - number of tweets in one chunk
    batch_size - int - number of tweets in one training batch
    shuffle_chunks_on_load - bool - if True, shuffles the chunks while loading data from files
    shuffle_in_chunks_on_load - bool - if True, shuffles tweets inside chunks while loading data from files
    shuffle_batch_on_return - bool - if True, shuffles tweets inside batch while iterating on dataset
    shuffle_in_chunk_on_chunk_reload - bool - if True, shuffles tweets inside the chunk whenever chunk is loaded
    rng_seed - int or None - random number generator seed
    temp_dir - str - path to the directory to store the chunks in

"""

from __future__ import print_function
import cPickle as pickle
import numpy as np
import os

__all__ = [
    "DataWrapper"
]


class DataWrapper:
    def __init__(self,
                 data_path,
                 train_ids,
                 valid_id,
                 test_id=None,
                 eval_ids=None,
                 create_chunks=True,
                 chunk_size=10000,
                 batch_size=200,
                 shuffle_chunks_on_load=True,
                 shuffle_in_chunks_on_load=True,
                 shuffle_batch_on_return=True,
                 shuffle_in_chunk_on_chunk_reload=True,
                 rng_seed=None,
                 temp_dir='temp_chunks',
                 print_progress=True):

        self.data_path = data_path
        if isinstance(self.data_path, basestring):
            self.data_path = [self.data_path]
        self.dataset_names = []
        for path in self.data_path:
            self.dataset_names.append(os.path.basename(path))

        self.temp_dir = temp_dir
        self.chunk_size = chunk_size // batch_size * batch_size  # make chunk_size a multiple of batch_size
        self.batch_size = batch_size

        self.shuffle_chunks_on_load = shuffle_chunks_on_load
        self.shuffle_in_chunks_on_load = shuffle_in_chunks_on_load
        self.shuffle_batch_on_return = shuffle_batch_on_return
        self.shuffle_in_chunk_on_chunk_reload = shuffle_in_chunk_on_chunk_reload

        if rng_seed is not None:
            np.random.seed(rng_seed)
        self.rng_seed = rng_seed

        self.create_chunks = create_chunks
        self.n_datasets = len(self.data_path)
        self.print_progress = print_progress

        if train_ids is None:
            raise ValueError('Specify at least one train id.')
        if isinstance(train_ids, (int, long)):
            train_ids = [train_ids]
        self.train_ids = train_ids
        if valid_id is None:
            raise ValueError('Specify at least one validation id.')
        self.valid_id = valid_id
        self.test_id = test_id
        if isinstance(eval_ids, (int, long)):
            eval_ids = [eval_ids]
        if eval_ids is None:
            eval_ids = []
        self.eval_ids = eval_ids

        self.max_len = 0
        self.labels = []
        self.n_labels = 0
        self.charset_map = {}
        self.charset_size = 0
        self.n_tweets = []
        self.n_chunks = []
        self.n_batches = []

        self.x = None
        self.x_mask = None
        self.y = None

        self.current_batch = 0
        self.current_chunk = 0
        self.current_data = 0

        self.__load_data_params()
        self.__load_data()

    def __iter__(self):
        return self

    def next(self):
        if self.current_batch < self.n_batches[self.current_data]:
            batch = self.__get_batch(self.current_batch)
            self.current_batch += 1
            return batch
        else:
            self.current_batch = 0
            raise StopIteration()

    def set_current_data(self, no):
        if 0 <= no < len(self.data_path):
            self.current_data = no
            self.current_batch = 0
            self.current_chunk = 0
            self.__load_chunk(0)

    def __get_batch(self, batch_id):
        if self.n_chunks[self.current_data] == 1:
            current_batch_in_chunk = batch_id
        else:
            # Load another chunk if necessary
            if not self.__is_batch_in_chunk(batch_id, self.current_chunk):
                self.__load_chunk(self.__get_chunk_id_of_batch(batch_id))
            current_batch_in_chunk = batch_id % (self.chunk_size / self.batch_size)

        current_slice = range(current_batch_in_chunk * self.batch_size,
                              (current_batch_in_chunk + 1) * self.batch_size)
        if self.shuffle_batch_on_return:
            np.random.shuffle(current_slice)
        return self.x[current_slice], self.x_mask[current_slice], self.y[current_slice]

    def __is_batch_in_chunk(self, batch_id, chunk_id):
        return self.chunk_size * chunk_id <= batch_id * self.batch_size < self.chunk_size * (chunk_id + 1)

    def __get_chunk_id_of_batch(self, batch_id):
        return batch_id * self.batch_size // self.chunk_size

    def __load_data_params(self):
        if self.create_chunks:
            for i_path, path in enumerate(self.data_path):
                with open(path, 'rb') as pfile:
                    tweets = pickle.load(pfile)
                    self.n_tweets.append(len(tweets))
                    for iTweet, tweet_entry in enumerate(tweets):
                        tweet_text = tweet_entry[1]
                        tweet_sentiment = tweet_entry[2]

                        if len(tweet_text) > self.max_len:
                            self.max_len = len(tweet_text)

                        for symbol in tweet_text:
                            if symbol not in self.charset_map:
                                self.charset_map[symbol] = self.charset_size
                                self.charset_size += 1

                        if tweet_sentiment not in self.labels:
                            self.labels.append(tweet_sentiment)
                            self.n_labels += 1
                self.n_chunks.append((self.n_tweets[i_path] - 1) / self.chunk_size + 1)
                self.n_batches.append((self.n_tweets[i_path] - 1) / self.batch_size + 1)
            self.__save_chunk_info()
        else:
            self.__load_chunk_info()

    def __save_chunk_info(self):
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        with open(os.path.join(self.temp_dir, 'chunk_info.p'), 'wb') as pfile:
            pickle.dump([self.max_len,
                         self.labels,
                         self.n_labels,
                         self.charset_map,
                         self.charset_size,
                         self.n_tweets,
                         self.n_chunks,
                         self.n_batches], pfile)

    def __load_chunk_info(self):
        with open(os.path.join(self.temp_dir, 'chunk_info.p'), 'rb') as pfile:
            [self.max_len,
             self.labels,
             self.n_labels,
             self.charset_map,
             self.charset_size,
             self.n_tweets,
             self.n_chunks,
             self.n_batches] = pickle.load(pfile)

    def __load_data(self):
        if self.create_chunks:
            self.symbols_loaded = 0
            for i_path, path in enumerate(self.data_path):
                self.current_data = i_path
                with open(path, 'rb') as pfile:
                    if self.print_progress:
                        print(self.dataset_names[i_path] + ': ', end='')
                    step = max(self.n_tweets[i_path] // 10, 1)

                    chunk_ids = range(self.n_chunks[i_path])
                    if self.shuffle_chunks_on_load:
                        # leave the last chunk at its place as it is most probably not full
                        last_id = chunk_ids[-1]
                        chunk_ids = chunk_ids[:-1]
                        np.random.shuffle(chunk_ids)
                        chunk_ids.append(last_id)

                    # limit the size in case there is not enough data to fill the whole chunk
                    if self.n_chunks[i_path] > 1:
                        data_size = self.chunk_size
                    else:
                        data_size = self.n_batches[i_path] * self.batch_size

                    tweets = pickle.load(pfile)
                    self.__reset_data(data_size)
                    chunk_id = 0

                    for iTweet, tweet_entry in enumerate(tweets):
                        if self.print_progress and not (iTweet + 1) % step:
                            print('.', end='')

                        iTweet %= self.chunk_size
                        tweet_text = tweet_entry[1]
                        tweet_sentiment = tweet_entry[2]

                        for iSym, symbol in enumerate(tweet_text):
                            self.x[iTweet, iSym] = self.charset_map[symbol]
                            self.x_mask[iTweet, iSym] = 1
                            self.symbols_loaded += 1

                        self.y[iTweet] = int(tweet_sentiment)

                        if iTweet == self.chunk_size - 1:
                            # chunk full - save

                            if self.shuffle_in_chunks_on_load:
                                self.__shuffle_data()

                            self.__save_chunk(chunk_ids[chunk_id])

                            if chunk_id == self.n_chunks[self.current_data] - 2:
                                # the last chunk may be smaller
                                data_size = (self.n_batches[i_path] * self.batch_size) % self.chunk_size

                            self.__reset_data(data_size)
                            chunk_id += 1

                    if chunk_id == self.n_chunks[self.current_data] - 1:
                        if self.shuffle_in_chunks_on_load:
                            self.__shuffle_data()
                            # self.__shuffle_data(chunk_ids[chunk_id])
                        self.__save_chunk(chunk_ids[chunk_id])

                    if self.print_progress:
                        print('')

        self.current_data = 0
        self.__load_chunk(0)

    def __encode1hot(self):
        x_1hot = np.zeros((self.x.shape[0], self.x.shape[1], self.charset_size))
        for iTweet, tweet in enumerate(self.x):
            for iSym, symbol in enumerate(tweet):
                if self.x_mask[iTweet, iSym] == 1:
                    x_1hot[iTweet, iSym, symbol] = 1
        return x_1hot

    def __reset_data(self, data_size):
        self.x = np.zeros((data_size, self.max_len), dtype=np.uint32)
        self.x_mask = np.zeros((data_size, self.max_len), dtype=np.uint32)
        self.y = np.zeros(data_size, dtype=np.uint32)

    def __shuffle_data(self):  #, chunk_id=None):
        # if chunk_id is None:
        #     chunk_id = self.current_chunk
        # if chunk_id == self.n_chunks[self.current_data] - 1 and self.y.shape[0] == self.chunk_size:
        #     shuffle_up_to = self.n_batches[self.current_data] * self.batch_size % self.chunk_size
        #     current_slice = range(shuffle_up_to)
        #     np.random.shuffle(current_slice)
        #     current_slice += range(shuffle_up_to, self.chunk_size)
        # else:
        #     current_slice = range(self.y.shape[0])
        #     np.random.shuffle(current_slice)
        current_slice = range(self.y.shape[0])
        np.random.shuffle(current_slice)
        self.x = self.x[current_slice]
        self.x_mask = self.x_mask[current_slice]
        self.y = self.y[current_slice]

    def __save_chunk(self, chunk_id):
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)

        file_path = os.path.join(self.temp_dir, 'chunk_' + str(self.current_data) + '_' + str(chunk_id) + '.npz')
        with open(file_path, 'wb') as pfile:
            np.savez_compressed(pfile, x=self.x, x_mask=self.x_mask, y=self.y)

    def __load_chunk(self, chunk_id):
        file_path = os.path.join(self.temp_dir, 'chunk_' + str(self.current_data) + '_' + str(chunk_id) + '.npz')
        with np.load(file_path) as vals:
            self.x = vals['x']
            self.x_mask = vals['x_mask']
            self.y = vals['y']

        self.current_chunk = chunk_id

        if self.shuffle_in_chunk_on_chunk_reload:
            self.__shuffle_data()

        self.x = self.__encode1hot()
