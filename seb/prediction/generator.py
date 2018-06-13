'''
Created on Jun 11, 2018

@author: nishilab
'''

import numpy as np
import tensorflow as tf

class Generator(object):

    def __init__(self, data, batch_size, num_steps, prediction_size=1):
        self._inputs, self._targets = data
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._data_length = self._inputs.shape[0]
        self._prediction_size = prediction_size
        self._num_batches = (self._data_length - prediction_size) // batch_size
        self._epoch_size = (self._data_length - prediction_size) // (num_steps * batch_size)
        assert (self._epoch_size > 0), "Epoch size is zero, num_steps or batch_size are to big"
        self._pos = -1
        self._x_data, self._y_data = self._format_data()

    @property
    def epoch_size(self):
        return self._epoch_size
        
    def _format_data(self):
        x_data = []
        y_data = []
        for num_batch in range(self._num_batches):
            x_batch = []
            y_batch = []
            for batch_pos in range(self._batch_size):
                pos =  batch_pos * self._num_batches + num_batch
                x_batch.append(self._inputs[pos])
                y_batch.append(self._targets[pos + 1])
            x_data.append(x_batch)
            y_data.append(y_batch)
        return np.asarray(x_data), np.asarray(y_data)
    
    def get_data(self):
        ds = tf.data.Dataset.from_tensor_slices((self._x_data, self._y_data))
        ds = ds.batch(self._num_steps).repeat()
        return ds.make_one_shot_iterator().get_next()