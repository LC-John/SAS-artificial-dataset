# -*- coding:utf-8 -*-

"""
@file:sas_dataset.py
@author:DrLC(Zhang Huangzhao)
@email:zhang_hz@pku.edu.cn
@time:2018/3/5 21:22
"""

import pickle, gzip
import random
import numpy

class SAS(object):
    '''
    The simple add sequence dataset

    Load the SAS dataset and split original train-set into train-set and
    valid-set.
    This class generate batch randomly, by calling "minibatch", "test_batch"
    or "valid_batch", and convert batch back to sequence data pair by calling
    "batch2raw".

    Parameters:

        path: path of the generated dataset, "./sas.pkl.gz" as default.
        valid_ratio: ratio of the valid-set in the original train-set.
        rand_seed: random seed for numpy.random and random.
    '''

    def __init__(self, path,
                 valid_ratio=0.2,
                 rand_seed=1234):

        self.set_rand_seed(rand_seed)

        if valid_ratio > 1:
            valid_ratio = 1
        elif valid_ratio < 0:
            valid_ratio = 0

        f = gzip.open(path, "rb")
        d = pickle.load(f)
        f.close()

        valid_size = int(valid_ratio * len(d["train"]))

        self.__test = d["test"]
        self.__valid = d["train"][:valid_size]
        self.__train = d["train"][valid_size:]
        self.__test_size = len(self.__test)
        self.__valid_size = len(self.__valid)
        self.__train_size = len(self.__train)

        self.__epoch = random.sample(list(range(self.__train_size)),
                                     self.__train_size)
        self.__dictionary = ["1", "2", "3", "4", "5",
                             "6", "7", "8", "9", "0",
                             " "]

    def minibatch(self, batch_size, max_len=15):

        '''
        Generate a minibatch from the current epoch. If the current epoch
        is smaller than batch size, generate a new epoch and extract a
        minibatch from the new epoch.
        The minibatch includes sequences in the form of a numpy.2darray,
        labels in the form of a numpy.1darray, lengths of each sequence in
        the form of a numpy.1darray.

        Parameter:

            batch_size: size of the minibatch.
            max_len: the final length of the sequences, with padding. Of
                course, max_len >= max(len(any sequence)). If the length
                of a sequence is less than max_len, pad blank at the end
                until the length reaches max_len.

        Returns:

            A batch, which is a list of [X, Y, L]
            X: sequence array, a numpy.2darray with the shape of
                [batch size, max length]. Each row is a padded sequence.
            Y: label vector, a numpy.1darray with the shape of [batch size].
                Each element is a label.
            L: length vector, a numpy.1darray with the shape of [batch size].
                Each element is the valid length (no padding length) of the
                corresponding sequence.
        '''

        if batch_size > self.__train_size:
            batch_size = self.__train_size
        if batch_size > len(self.__epoch):
            self.__epoch = random.sample(list(range(self.__train_size)),
                                         self.__train_size)

        idx = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]

        X = []
        Y = []
        L = []
        for i in idx:

            if self.__train[i][1] == "GE":
                Y.append(0)
            elif self.__train[i][1] == "LT":
                Y.append(1)
            else:
                assert False

            assert len(self.__train[i][0]) <= max_len
            L.append(len(self.__train[i][0]))

            tmp_seq = []
            for t in self.__train[i][0]:
                assert t in self.__dictionary
                tmp_seq.append(self.__dictionary.index(t))
            tmp_seq += [10 for iii in range(max_len - L[-1])]
            X.append(tmp_seq)

        X = numpy.asarray(X, dtype=numpy.int32)
        Y = numpy.asarray(Y, dtype=numpy.int32)
        L = numpy.asarray(L, dtype=numpy.int32)

        return X, Y, L

    def test_batch(self, batch_size, max_len=15):

        '''
        Generate a batch from test-set. Similar to minibatch, but there is
        no such thing like "epoch" in test_batch.

        Parameters:

            batch_size: size of the test batch.
            max_len: length of the sequences after padding.

        Returns:

            Same as minibatch
        '''

        if batch_size > self.__test_size:
            batch_size = self.__test_size

        idx = random.sample(list(range(self.__test_size)),
                            batch_size)

        X = []
        Y = []
        L = []
        for i in idx:

            if self.__test[i][1] == "GE":
                Y.append(0)
            elif self.__test[i][1] == "LT":
                Y.append(1)
            else:
                assert False

            assert len(self.__test[i][0]) <= max_len
            L.append(len(self.__test[i][0]))

            tmp_seq = []
            for t in self.__test[i][0]:
                assert t in self.__dictionary
                tmp_seq.append(self.__dictionary.index(t))
            tmp_seq += [10 for iii in range(max_len - L[-1])]
            X.append(tmp_seq)

        X = numpy.asarray(X, dtype=numpy.int32)
        Y = numpy.asarray(Y, dtype=numpy.int32)
        L = numpy.asarray(L, dtype=numpy.int32)

        return X, Y, L

    def valid_batch(self, batch_size, max_len=15):

        '''
        Generate a batch from valid-set. Same as test_batch

        Parameters:

            batch_size: size of the test batch.
            max_len: length of the sequences after padding.

        Returns:

            Same as test_batch
        '''

        if batch_size > self.__valid_size:
            batch_size = self.__valid_size

        idx = random.sample(list(range(self.__valid_size)),
                            batch_size)

        X = []
        Y = []
        L = []
        for i in idx:

            if self.__valid[i][1] == "GE":
                Y.append(0)
            elif self.__valid[i][1] == "LT":
                Y.append(1)
            else:
                assert False

            assert len(self.__valid[i][0]) <= max_len
            L.append(len(self.__valid[i][0]))

            tmp_seq = []
            for t in self.__valid[i][0]:
                assert t in self.__dictionary
                tmp_seq.append(self.__dictionary.index(t))
            tmp_seq += [10 for iii in range(max_len - L[-1])]
            X.append(tmp_seq)

        X = numpy.asarray(X, dtype=numpy.int32)
        Y = numpy.asarray(Y, dtype=numpy.int32)
        L = numpy.asarray(L, dtype=numpy.int32)

        return X, Y, L

    def batch2raw(self, batch):

        '''
        Convert a batch back to raw sequence data., for debugging usage.

        Parameters:

            batch: a batch generated from minibatch, test_batch or
                valid_batch.

        Returns:

            A sequence data pair, as a list of [sequences, labels].
        '''

        x, y, l = batch

        seq = []
        for i in range(len(l)):
            tmp_seq = ""
            for j in range(l[i]):
                tmp_seq += self.__dictionary[x[i][j]]
            seq.append(tmp_seq)

        label = []
        for i in y:
            if i == 0:
                label.append("GE")
            elif i == 1:
                label.append("LT")
            else:
                assert False

        return seq, label

    def set_rand_seed(self, rand_seed):

        '''
        Set random seed for numpy.random and random.

        Parameters:

            rand_seed: random seed.

        Returns:

            None
        '''

        random.seed(rand_seed)
        numpy.random.seed(rand_seed)

    def get_test_size(self):

        '''
        Get test-set size

        Parameters:

            None

        Returns:

            Size of test-set.
        '''

        return self.__test_size

    def get_train_size(self):

        '''
        Get train-set size

        Parameters:

            None

        Returns:

            Size of train-set.
        '''

        return self.__train_size

    def get_valid_size(self):

        '''
        Get valid-set size

        Parameters:

            None

        Returns:

            Size of valid-set.
        '''

        return self.__valid_size


class generator(object):
    '''
    The simple add sequence generator

    Randomly generate pieces of data for sas GE/LT classification task.
    Each piece of data has an input string and an output label.

    The input string has 4 integer numbers, seperated by a blank. There
    are two candidate output labels -- "GE" and "LT".
    Denote the 4 integer as a, b, c, ans (in string sequential order).
    If a+b-c >= ans, the output label should be "GE"; otherwise, the
    output label should be "LT".

    eg. Input: "65 235 328 905"; output: "LT".
        Input: "744 25 3 376";   output: 'GE'.
    '''

    def __init__(self, num_lim=[0, 1000],
                 seq_num=120000,
                 test_num=20000,
                 train_num=100000,
                 rand_seed=1234):

        # set parameters
        self.set_param(num_lim, seq_num, test_num, train_num, rand_seed)
        # generate sequences
        self.generate()

    def set_param(self, num_lim=[0, 1000],
                  seq_num=100000,
                  test_num=20000,
                  train_num=100000,
                  rand_seed=1234):

        '''
        Set the parameters, including boundary of a, b, c and ans, sequence
        number, and random seed.

        Parameters:

            num_lim: boundary of a, b, c and ans. The first element is the
                lower bound, and the second is the upper bound.
            seq_num: sequence number to be generated.
            test_num: sequence number for test set.
            train_num: sequence number for train set.
            rand_seed: random seed for numpy.random

        Returns:

            None
        '''

        # notice that sequence number must eqaul to
        # the sum of test set size and train set size
        assert test_num + train_num == seq_num

        # set random seed
        numpy.random.seed(rand_seed)
        # set the parameters
        self.__min_num = num_lim[0]
        self.__max_num = num_lim[1]
        self.__seq_num = seq_num
        self.__test_num = test_num
        self.__train_num = train_num

    def generate(self):

        '''
        Generate data pairs (input string and output label), and do some
        statistics on the generated dataset.

        Parameters:

            None

        Returns:

            None
        '''

        self.__seq = []
        for cnt in range(self.__seq_num):
            # generate a, b, c, ans randomly
            a, b, c, ans = self.__generate_random_number_set()
            tmp_seq = str(a) + " " + str(b) + " " + str(c) + " " + str(ans)
            # compare a+b-c with ans
            if a + b - c >= ans:
                tmp_label = "GE"  # greater or equal
            else:
                tmp_label = "LT"  # less than
            self.__seq.append([tmp_seq, tmp_label])

        self.test = self.__seq[:self.__test_num]
        self.train = self.__seq[self.__test_num:]

        # do some statistics works
        self.__stat_seq_len()
        self.__stat_label()

    def __generate_random_number_set(self):

        '''
        Generate the four numbers for each data pair.

        Parameters:

            None

        Returns:

            a, b, c and ans for one input string sequence
        '''

        tmp_arr = numpy.random.uniform(self.__min_num,
                                       self.__max_num,
                                       size=[4])
        a, b, c, d = numpy.asarray(tmp_arr, dtype=numpy.int32)
        return a, b, c, d

    def __stat_seq_len(self):

        '''
        Statistics. Get the distribution of sequence lengths.

        Parameters:

            None

        Returns:

            None
        '''

        self.seq_len = {}
        for d in self.__seq:
            tmp_l = len(d[0])
            if tmp_l not in self.seq_len.keys():
                self.seq_len[tmp_l] = 1
            else:
                self.seq_len[tmp_l] += 1

    def __stat_label(self):

        '''
        Statistics. Get the distribution of GE/LT labels.

        Parameters:

            None

        Returns:

            None
        '''

        self.label_cnt = {"GE": 0, "LT": 0}
        for d in self.__seq:
            self.label_cnt[d[1]] += 1

    def save_sas(self, path):

        f = gzip.open(path, "wb")
        pickle.dump({"train": self.train,
                     "test": self.test}, f)
        f.close()

    def get_sas(self):

        return {"train": self.train,
                "test": self.test}