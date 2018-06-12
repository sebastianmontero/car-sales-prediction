'''
Created on Jun 11, 2018

@author: nishilab
'''

class Generator(object):

    def __init__(self, data, batch_size, num_steps, prediction_size=1):
        self._data = data
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._data_length = data.shape[0]
        self._prediction_size = prediction_size
        self._num_batches = (self._data_length - prediction_size) // batch_size
        self._epoch_size = (self._data_length - prediction_size) // (num_steps * batch_size)
        assert (self._epoch_size > 0), "Epoch size is zero, num_steps or batch_size are to big"
        self._pos = -1
        self._x_data = None
        self._x_data = None
        self._x_data, self._y_data = self._format_data()

    def _reset(self):
        self._pos = -1
        
    def _format_data(self):
        x_data = []
        y_data = []
        for num_batch in range(self._num_batches):
            x_batch = []
            y_batch = []
            for batch_pos in range(self._batch_size):
                pos =  batch_pos * self._batch_size + num_batch
                x_batch.append(self._data[pos])
                y_batch.append(self._data[pos + 1])
            x_data.append(x_batch)
            y_data.append(y_batch)
        return x_data, y_data

    def next_epoch_stage(self):
        self._pos += 1
        return self.has_more_epoch_stages()
    
    def get_stage(self):
        start_pos = self._pos * self._num_steps
        end_pos = (self._pos  + 1) * self._num_steps
        return self._x_data[start_pos:end_pos], self._y_data[start_pos:end_pos] 
            
    def has_more_epoch_stages(self):
        return self._pos < self._epoch_size