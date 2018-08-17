'''
Created on Jun 11, 2018

@author: nishilab
'''
import tensorflow as tf
from builtins import property


class NoiseVarianceGenerator(object):
    
    def __init__(self, x, y, batch_size):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._epoch_size = len(x) // batch_size
        self._num_features = x.shape[1]
    
    @property
    def epoch_size(self):
        return self._epoch_size
    
    @property
    def num_features(self):
        return self._num_features
    
    def get_data(self):
        ds = tf.data.Dataset.from_tensor_slices((self._x, self._y))
        ds = ds.batch(self._batch_size).repeat().shuffle(buffer_size = 1000)
        return ds.make_one_shot_iterator().get_next()