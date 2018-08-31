'''
Created on Jun 11, 2018

@author: nishilab
'''

class Generator(object):

    def __init__(self, inputs, targets, num_predicted_vars):
        
        self._inputs = inputs
        self._targets = targets
        self._num_predicted_vars = num_predicted_vars

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
        