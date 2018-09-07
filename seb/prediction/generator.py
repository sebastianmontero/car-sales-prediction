'''
Created on Jun 11, 2018

@author: nishilab
'''

import math
import numpy as np
import tensorflow as tf

class Generator(object):
    
    PREDICTED_VARS_START_POS = 2

    def __init__(self, data, batch_size, num_steps, num_predicted_features, num_predicted_vars, prediction_size=1, multi_month_prediction=False, for_test=False):
        
        prediction_data_size = (prediction_size * 2 - 1) if for_test and multi_month_prediction else prediction_size
        residual = (data.shape[0] - prediction_data_size) % batch_size
        self._data = data[residual:]
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._data_length = data.shape[0]
        self._num_predicted_features = num_predicted_features
        self._num_predicted_vars = num_predicted_vars
        self._prediction_size = prediction_size
        self._multi_month_prediction = multi_month_prediction
        self._for_test = for_test
        self._num_batches = (self._data_length - prediction_data_size) // batch_size
        self._epoch_size = math.ceil((self._data_length - prediction_data_size) / (num_steps * batch_size))
        assert (self._epoch_size > 0), "Epoch size is zero, num_steps or batch_size are to big"
        self._pos = -1
        self._x_data, self._y_data = self._format_data()

    @property
    def epoch_size(self):
        return self._epoch_size
    
    @property
    def num_predicted_features(self):
        return self._num_predicted_features
    
    @property
    def num_predicted_vars(self):
        return self._num_predicted_vars
        
    def _format_data(self):
        x_data = []
        y_data = []
        predictions = [i + 1 for i in range(self._prediction_size)] if self._multi_month_prediction else [self._prediction_size]
        for num_batch in range(self._num_batches - 1):
            x_batch, y_batch = self._create_batch(num_batch, predictions)        
            x_data.append(x_batch)
            y_data.append(y_batch)
        
        final_batch_pos = self._num_batches - 1
        if self._for_test and self._multi_month_prediction:
            final_batch_pos += self._prediction_size - 1
        x_batch, y_batch = self._create_batch(final_batch_pos, predictions)        
        x_data.append(x_batch)
        y_data.append(y_batch)
        
        return np.asarray(x_data), np.asarray(y_data)
    
    def _create_batch(self, num_batch, predictions):
        x_batch = []
        y_batch = []
        for batch_pos in range(self._batch_size):
            pos =  batch_pos * self._num_batches + num_batch
            x_batch.append(self._data[pos])
            y = []
            for prediction in predictions:
                y = np.concatenate((y, self._data[pos + prediction][self.PREDICTED_VARS_START_POS:self.PREDICTED_VARS_START_POS + self._num_predicted_features]))
            y_batch.append(y)
        return x_batch, y_batch
    
    def get_data(self):
        ds = tf.data.Dataset.from_tensor_slices((self._x_data, self._y_data))
        ds = ds.batch(self._num_steps).repeat()
        return ds.make_one_shot_iterator().get_next()