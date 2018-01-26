# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:56:59 2018

@author: DrLC
"""

import numpy
import gzip, pickle

class simple_add_seq_generator(object):
    
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
    
    def __init__(self, num_lim=[0,1000],
                 seq_num=120000,
                 test_num=20000,
                 train_num=100000,
                 rand_seed=1234):
        
        # set parameters
        self.set_param(num_lim, seq_num, test_num, train_num, rand_seed)
        # generate sequences
        self.generate()
    
    def set_param(self, num_lim=[0,1000],
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
            tmp_seq = str(a)+" "+str(b)+" "+str(c)+" "+str(ans)
            # compare a+b-c with ans
            if a + b - c >= ans:
                tmp_label = "GE" # greater or equal
            else:
                tmp_label = "LT" # less than
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

    def save(self, path="./sas.pkl.gz"):
        
        f = gzip.open(path, "wb")
        pickle.dump({"train": self.train,
                     "test": self.test}, f)
        f.close()

if __name__ == "__main__":
    
    sas_gen = simple_add_seq_generator()
    sas_gen.save()