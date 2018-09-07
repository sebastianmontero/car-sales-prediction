'''
Created on Jun 15, 2018

@author: nishilab
'''

import math
import seaborn as sns
import numpy as np

from utils import Utils

class BaseEvaluator(object):

    def __init__(self):
        sns.set()
        
    @property
    def start_window_pos(self):
        return self.reader.process_absolute_pos(self.end_window_pos) - self.window_length
    
    @property
    def end_window_pos(self):
        return self.reader.process_absolute_pos(self._end_window_pos)
    
    @property
    def window_length(self):
        return self._window_length - self.reader.num_base_ensembles
    
    @property
    def reader(self):
        return self._reader
    
    @property
    def predicted_features(self):
        return self.reader.predicted_features
    
    @property
    def num_predicted_features(self):
        return self.reader.num_predicted_features
    
    def predictions_by_absolute_pos(self, pos, scaled=False):
        start = self.start_window_pos
        if start <= pos and pos < self.end_window_pos:
            return self.get_predictions_by_row(pos - start, scaled)
        return None
    
    def get_predictions_by_row(self, row, scaled=False):
        return self.predictions(scaled)[row]
    
    def get_predicted_var_name(self, feature_pos):
        return self.reader.get_predicted_var_name(feature_pos)
    
    def _get_feature_values(self, data, feature_pos):
        return np.take(data, feature_pos, axis=1)
    
    def _unscale_features(self, features, round_=True):
        return self._reader.unscale_features(features, round_)    
        
    def _calculate_absolute_error_by_pos(self, targets, predictions, pos):
        return math.fabs(targets[pos]- predictions[pos])
    
    def calculate_absolute_error(self, targets, predictions):
        return np.abs(targets - predictions)
    
    def calculate_relative_error(self, targets, predictions):
        errors = []
        for target, prediction in zip(targets, predictions):
            if target == prediction:
                errors.append(0)
            else:
                errors.append(((target - prediction)/(math.fabs(target) + math.fabs(prediction))) * 200) 
            
        return errors
    
    def absolute_error(self, feature_pos=0, scaled=False):
        targets = self._get_target_by_pos(feature_pos, scaled=scaled)
        predictions = self.get_predictions(feature_pos, scaled=scaled)
        return self.calculate_absolute_error(targets, predictions)
    
    def relative_error(self, feature_pos=0, scaled=False):
        targets = self._get_target_by_pos(feature_pos, scaled=scaled)
        predictions = self.get_predictions(feature_pos, scaled=scaled)
        return self.calculate_relative_error(targets, predictions)
    
    def _calculate_absolute_mean_error(self, targets, predictions):
        return np.mean(self.calculate_absolute_error(targets, predictions))
    
    def _calculate_relative_mean_error(self, targets, predictions):
        return np.mean(np.absolute(self.calculate_relative_error(targets, predictions)))
        
    def _get_target_by_pos(self, target_pos, scaled=False, length=None):
        return self.get_target(self.get_predicted_var_name(target_pos), scaled, length)

    def get_target(self, target, scaled=False, length=None):
        return self._get_data(scaled, length)[target].values
    
    def get_months(self, length=None):
        return self._get_data(length=length)['month_id'].values
    
    def get_predicted_targets(self, scaled=False):
        return self._get_data(scaled)[self.predicted_features].values
    
    def _get_data(self, scaled=False, length=None):
        length = length if length else self.window_length
        return self._reader.get_data(self._end_window_pos, length, scaled)
    
    def get_target_data_length(self, tail=False):
        return self._end_window_pos if tail else self.window_length
    
    def get_predictions(self, feature_pos=0, scaled=False):
        raise NotImplementedError("Child classes must implement this method")
    
    def _generate_feature_name(self, feature_name, scaled=None):
        name = ''
        if scaled is not None:
            name += 'Scaled ' if scaled else 'Real '
        name +=  self.format_name(feature_name)
        return name.strip()
    
    def generate_feature_name(self, feature_pos, scaled=None):
        return self._generate_feature_name(self.get_predicted_var_name(feature_pos), scaled)
    
    def format_name(self, name):
        ns = name.split('_')
        fname = ''
        
        for n in ns:
            fname += n[0].capitalize() + n[1:] + ' '
            
        return fname  
    
    def absolute_mean_error(self, feature_pos=0, scaled=False):
        return self._calculate_absolute_mean_error(self._get_target_by_pos(feature_pos, scaled=scaled), self.get_predictions(feature_pos, scaled=scaled))
    
    def window_real_absolute_mean_error(self):
        return (self.absolute_mean_error(0) + self.absolute_error_by_pos(-1)) / 2
    
    def relative_mean_error(self, feature_pos=0, scaled=False):
        return self._calculate_relative_mean_error(self._get_target_by_pos(feature_pos, scaled=scaled), self.get_predictions(feature_pos, scaled=scaled))
    
    def _get_absolute_error_by_pos(self, pos, feature_pos=0,  scaled=False):
        return self._calculate_absolute_error_by_pos(self._get_target_by_pos(feature_pos, scaled=scaled), self.get_predictions(feature_pos,scaled=scaled), pos)
    
    def absolute_error_by_pos(self, pos, feature_pos=0, scaled=False):
        return self._get_absolute_error_by_pos(pos, feature_pos, scaled=scaled)
    
    
    