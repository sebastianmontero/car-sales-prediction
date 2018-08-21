'''
Created on Jun 15, 2018

@author: nishilab
'''


import numpy as np
from utils import Utils
from base_evaluator import BaseEvaluator

class Evaluator(BaseEvaluator):

    def __init__(self, reader, predictions, end_window_pos, global_step=None):
        BaseEvaluator.__init__(self)
        self._reader = reader
        self._unscaled_predictions = self._reader.unscale_sales(predictions)
        self._predictions = np.reshape(predictions, [-1])
        self._end_window_pos = end_window_pos
        self._window_length = len(predictions)
        self._global_step = global_step
        
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def global_step(self):
        return self._global_step
    
    @property
    def unscaled_predictions(self):
        return self._unscaled_predictions
    
    def get_predictions(self, scaled=False):
        return self._predictions if scaled else self._unscaled_predictions