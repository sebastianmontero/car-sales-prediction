'''
Created on Jun 11, 2018

@author: nishilab
'''

class Generator(object):

    def __init__(self, inputs, targets, iterator_initializer, num_predicted_vars, epoch_size):
        
        self._inputs = inputs
        self._targets = targets
        self._iterator_initializer = iterator_initializer
        self._num_predicted_vars = num_predicted_vars
        self._epoch_size = epoch_size

    @property
    def iterator_initializer(self):
        return self._iterator_initializer
    
    @property
    def epoch_size(self):
        return self._epoch_size
    
    @property
    def num_predicted_vars(self):
        return self._num_predicted_vars
    
    def get_data(self):
        return self._inputs, self._targets
        