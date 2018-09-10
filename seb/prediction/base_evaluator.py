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
    def num_predictions(self):
        return self.prediction_size if self.multi_month_prediction else 1
    
    def prediction_indexes(self, prediction_indexes=[]):
        if len(prediction_indexes) == 0:
            prediction_indexes = [i for i in range(self.num_predictions)]
        return prediction_indexes
    
    def window_shift(self, prediction_index=0):
        return self.num_predictions() - prediction_index - 1
        
    def start_window_pos(self, prediction_index=0):
        return self.end_window_pos(prediction_index) - self.window_length
    
    def end_window_pos(self, prediction_index=0):
        return self.reader.process_absolute_pos(self._end_window_pos) - self.window_shift(prediction_index)
    
    def feature_index(self, feature_pos=0, prediction_index=0):
        return prediction_index * self.num_predicted_features + feature_pos
    
    @property
    def window_length(self):
        return self._window_length - self.reader.num_base_ensembles
    
    @property
    def reader(self):
        return self._reader
    
    @property
    def multi_month_prediction(self):
        return self.reader.multi_month_prediction
    
    @property
    def prediction_size(self):
        return self.reader.prediction_size
    
    @property
    def predicted_features(self):
        return self.reader.predicted_features
    
    @property
    def num_predicted_features(self):
        return self.reader.num_predicted_features
    
    def predictions_by_absolute_pos(self, pos, scaled=False):
        prediction_index = self.num_predictions - 1
        start = self.start_window_pos(prediction_index)
        if start <= pos and pos < self.end_window_pos(prediction_index):
            return self.get_predictions_by_row(pos - start, scaled)
        return None
    
    def get_predictions_by_row(self, row, scaled=False):
        return self.predictions(scaled)[row]
    
    def get_predicted_var_name(self, feature_pos):
        return self.reader.get_predicted_var_name(feature_pos)
    
    def _get_feature_values(self, data, feature_pos):
        return np.take(data, feature_pos, axis=1)
    
    def _unscale_features(self, features, round_=True):
        unscaled = []
        for pi in self.prediction_indexes:
            unscaled.append(self.reader.unscale_features(features[:, self.feature_index(prediction_index=pi):self.feature_index(prediction_index=pi+1)], round_))
        unscaled = np.concatenate(unscaled, axis=1)
        return unscaled     
        
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
    
    def absolute_error(self, feature_pos=0, scaled=False, prediction_index=0):
        targets = self._get_target_by_pos(feature_pos, scaled=scaled, prediction_index=prediction_index)
        predictions = self.get_predictions(feature_pos, scaled=scaled, prediction_index=prediction_index)
        return self.calculate_absolute_error(targets, predictions)
    
    def relative_error(self, feature_pos=0, scaled=False):
        targets = self._get_target_by_pos(feature_pos, scaled=scaled)
        predictions = self.get_predictions(feature_pos, scaled=scaled)
        return self.calculate_relative_error(targets, predictions)
    
    def _calculate_absolute_mean_error(self, targets, predictions):
        return np.mean(self.calculate_absolute_error(targets, predictions))
    
    def _calculate_relative_mean_error(self, targets, predictions):
        return np.mean(np.absolute(self.calculate_relative_error(targets, predictions)))
        
    def _get_target_by_pos(self, target_pos, scaled=False, length=None, prediction_index=0):
        return self.get_target(self.get_predicted_var_name(target_pos), scaled, length=length, prediction_index=prediction_index)

    def get_target_by_start_end(self, target, start_pos, end_pos, scaled=False):
        return self.get_target(target, scaled, end_pos, end_pos - start_pos)
    
    def get_target(self, target, scaled=False, end_window_pos=None, length=None, prediction_index=0):
        return self._get_data(scaled, end_window_pos, length, prediction_index)[target].values
    
    def get_months(self, end_window_pos=None, length=None, prediction_index=0):
        return self._get_data(end_window_pos=end_window_pos, length=length, prediction_index=prediction_index)['month_id'].values
    
    def get_predicted_targets(self, scaled=False):
        targets = []
        for pi in self.prediction_indexes:
            targets.append(self._get_data(scaled, prediction_index=pi)[self.predicted_features].values)
        
        targets = np.concatenate(targets, axis=1)
        return targets
    
    def _get_data(self, scaled=False, end_window_pos=None, length=None, prediction_index=0):
        
        end_window_pos = self.end_window_pos(prediction_index) if end_window_pos is None else end_window_pos
        length = self.window_length if length is None else length 
        return self.reader.get_data(end_window_pos, length, scaled)
    
    def get_target_data_length(self, tail=False, prediction_index=0):
        return self.end_window_pos(prediction_index) if tail else self.window_length
    
    def get_predictions(self, feature_pos=0, scaled=False, prediction_index=0):
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
        return self.absolute_mean_error_detail(feature_pos, scaled)['total']
        
    def absolute_mean_error_detail(self, feature_pos=0, scaled=False, prediction_indexes=[]):
        
        return self._mean_error_detail(self._calculate_absolute_mean_error, feature_pos, scaled, prediction_indexes) 
    
    def _mean_error_detail(self, fn, feature_pos=0, scaled=False, prediction_indexes=[]):
        
        mes = []
        prediction_indexes = self.prediction_indexes(prediction_indexes)
        
        for pi in prediction_indexes:
            mes.append(fn(self._get_target_by_pos(feature_pos, scaled=scaled, prediction_index=pi), 
                                                   self.get_predictions(feature_pos, scaled=scaled, prediction_index=pi)))
        return {
            'detail': mes,
            'total': np.average(mes)
        } 
    
    def window_real_absolute_mean_error(self):
        return (self.absolute_mean_error(0) + self.absolute_error_by_pos(-1)) / 2
    
    def relative_mean_error(self, feature_pos=0, scaled=False):
        return self.relative_mean_error_detail(feature_pos, scaled)['total']
    
    def relative_mean_error_detail(self, feature_pos=0, scaled=False, prediction_indexes=[]):
        return self._mean_error_detail(self._calculate_relative_mean_error, feature_pos, scaled, prediction_indexes) 
    
    
    def _get_absolute_error_by_pos(self, pos, feature_pos=0,  scaled=False, prediction_index=0):
        return self._calculate_absolute_error_by_pos(self._get_target_by_pos(feature_pos, scaled=scaled, prediction_index=prediction_index), 
                                                     self.get_predictions(feature_pos,scaled=scaled, prediction_index=prediction_index), pos)
    
    def absolute_error_by_pos(self, pos, feature_pos=0, scaled=False):
        return self._get_absolute_error_by_pos(pos, feature_pos, scaled=scaled)
    
    
    