'''
refer: SpecAugment: ASimpleDataAugmentationMethod for Automatic Speech Recognition
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import numbers
import random


def get_mask_par(x_size, rate):
    # build a masking 0-1 matrix
    ones = np.ones(x_size)
    MAX_FREQ_MASK = 5
    MAX_TIME_MASK = 10
    aug_num = int(x_size[0]*rate)
    random_index_mask = random.sample(range(0, x_size[0]), aug_num)
    #print(random_index_mask) # wrong: only run 1 time
    for i in random_index_mask:
        start_t = np.random.randint(0,x_size[1]-MAX_TIME_MASK)
        strat_f = np.random.randint(0,x_size[2]-MAX_FREQ_MASK)
        len_t = np.random.randint(2, MAX_TIME_MASK)
        len_f = np.random.randint(2, MAX_FREQ_MASK)
        ones[i][start_t:start_t+len_t, :] = 0.
        ones[i][:, strat_f:strat_f+len_f] = 0.
    return ones

def get_warp_par(x_size, rate):
    time_middle = x_size[1]//2
    MAX_W = 5 
    aug_num = int(x_size[0]*rate)
    w = np.random.randint(-MAX_W,MAX_W+1, [x_size[0],]) # left or right
    random_index_not_warp = random.sample(range(0, x_size[0]), x_size[0] - aug_num)
    w[random_index_not_warp] = 0
    boundary = np.array([[[0,0], [0, x_size[2]-1], [x_size[1]-1,0], [x_size[1]-1, x_size[2]-1]]])
    
    # to build the source and dest array for sparse_image_warp()
    s = np.array([[[time_middle-w[0], 0],[time_middle-w[0], x_size[2]-1]]])
    s = np.concatenate([s, boundary], axis=1)

    d = np.array([[[time_middle+w[0], 0],[time_middle+w[0], x_size[2]-1]]])
    d = np.concatenate([d, boundary], axis=1)
    for i in range(1, x_size[0]):
        temp_s = np.array([[[time_middle-w[i], 0],[time_middle-w[i], x_size[2]-1]]])
        temp_s = np.concatenate([temp_s, boundary], axis=1)
        s = np.concatenate([s, temp_s], axis=0)
        
        temp_d = np.array([[[time_middle+w[i], 0],[time_middle+w[i], x_size[2]-1]]])
        temp_d = np.concatenate([temp_d, boundary], axis=1)
        d = np.concatenate([d, temp_d], axis=0)
    return s, d

def mask(x, mask_matrix):
    x_mask = x * mask_matrix
    return x_mask

def time_warp(x, x_size, s, d):
    x = tf.reshape(x, shape=[x_size[0], x_size[1], x_size[2], 1])
    x_warp = tf.contrib.image.sparse_image_warp(x, s, d)
    x_warp = tf.reshape(x_warp[0], shape = x_size)
    return x_warp

def data_augment(x, x_size, mask_matrix, warp_source, warp_dest):
    '''
    ' param.:
    '  rate: the percentage of x to be augmented
    '''
    # default x_size = [128, 65, 40] in current code
    x = tf.reshape(x, shape=x_size)
    x_warp = time_warp(x, x_size, warp_source, warp_dest)
    x_mask = mask(x_warp, mask_matrix)
    return x_mask
