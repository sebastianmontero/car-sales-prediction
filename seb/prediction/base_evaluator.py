'''
Created on Jun 15, 2018

@author: nishilab
'''

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from multiprocessing import Process

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
        return self._window_length
    
    @property
    def reader(self):
        return self._reader
    
    @property
    def predicted_vars(self):
        return self.reader.predicted_vars
    
    @property
    def num_predicted_vars(self):
        return self.reader.num_predicted_vars
    
    def predictions_by_absolute_pos(self, pos, scaled=False):
        start = self.start_window_pos
        if start <= pos and pos < self.end_window_pos:
            return self.get_predictions_by_row(pos - start, scaled)
        return None
    
    def get_predictions_by_row(self, row, scaled=False):
        return self.predictions(scaled)[row]
    
    def _get_predicted_var_name(self, feature_pos):
        return self.reader.get_predicted_var_name(feature_pos)
    
    def _get_feature_values(self, data, feature_pos):
        return np.take(data, feature_pos, axis=1)
    
    def _unscale_features(self, features, round_=True):
        return self._reader.unscale_features(features, round_)
    
    def _run_in_new_process(self, target, args=()):
        p = Process(target=target, args=args)
        p.start()
    
    def _plot_target_vs_predicted_new_process(self, real, predictions, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_predicted, args=(real, predictions, ylabel, title))
    
    def _plot_target_vs(self, real, vs, ylabel, title):
        vs['Real'] = real
        self._plot_by_month('Real', vs, ylabel, title)
        
    def _plot_by_month(self, ref_key, series, ylabel, title, yfix=False):
        ref_vals = series[ref_key]
        if isinstance(ref_vals, dict):
            ref_vals = ref_vals['values']
        months = self._get_months(len(ref_vals))
        num_months = len(months)
        
        min_val = None
        max_val = None
        for label, obj in series.items():
            vals = obj
            plt_type = 'line'
            if isinstance(obj, dict):
                vals = obj['values']
                plt_type = obj.get('type', 'line')
            
            tmin=min(vals)
            tmax=max(vals)
            if min_val is None or tmin < min_val:
                min_val = tmin
            if max_val is None or tmax > max_val:
                max_val = tmax
            
            if plt_type == 'bar':
                plt.bar(range(num_months - len(vals), num_months), vals, label=(label))
            else:
                plt.plot(range(num_months - len(vals), num_months), vals, label=(label))
        
        if yfix:
            plt.ylim([min_val-0.5*(max_val-min_val), max_val+0.5*(max_val-min_val)])
        
        plt.ylabel(ylabel)
        plt.xticks(range(num_months), months, rotation='vertical')
        plt.title(title)
        plt.legend()
        plt.show()
        
    def _plot_target_vs_predicted(self, real, predictions, ylabel, title):
        self._plot_target_vs(real, {'Predicted' : predictions}, ylabel, title)
    
    def _plot_errors_new_process(self, absolute, relative, ylabel_absolute, ylabel_relative, title):
        self._run_in_new_process(target=self._plot_errors, args=(absolute, relative, ylabel_absolute, ylabel_relative, title))
            
    def _plot_errors(self, absolute, relative, ylabel_absolute, ylabel_relative, title):
        months = self._get_months()
        color = 'tab:blue'
        fig, ax1 = plt.subplots()
        ax1.set_xticks(range(len(months)))
        ax1.set_xticklabels(months)
        ax1.tick_params(axis='x', labelrotation=90)
        
        ax1.set_ylabel(ylabel_absolute, color=color)
        ax1.plot(absolute, 'o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        color = 'tab:green'
        ax2 = ax1.twinx()
        ax2.plot(relative, 'o', color=color)
        ax2.set_ylabel(ylabel_relative, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
            
        plt.suptitle(title)
        plt.show()
        
        
    def _calculate_absolute_error_by_pos(self, targets, predictions, pos):
        return math.fabs(targets[pos]- predictions[pos])
    
    def _calculate_absolute_error(self, targets, predictions):
        return np.abs(targets - predictions)
    
    def _calculate_relative_error(self, targets, predictions):
        errors = []
        for target, prediction in zip(targets, predictions):
            if target == prediction:
                errors.append(0)
            else:
                errors.append(((target - prediction)/(math.fabs(target) + math.fabs(prediction))) * 200) 
            
        return errors
    
    def _calculate_absolute_mean_error(self, targets, predictions):
        return np.mean(self._calculate_absolute_error(targets, predictions))
    
    def _calculate_relative_mean_error(self, targets, predictions):
        return np.mean(np.absolute(self._calculate_relative_error(targets, predictions)))
        
    def _get_target_by_pos(self, target_pos, scaled=False, length=None):
        return self._get_target(self.reader.get_predicted_var_name(target_pos), scaled, length)

    def _get_target(self, target, scaled=False, length=None):
        return self._get_data(scaled, length)[target].values
    
    def _calculate_and_plot_errors(self, feature_pos=0, scaled=False):
        feature_name = self.reader.get_predicted_var_name(feature_pos)
        targets = self._get_target(feature_name, scaled=scaled)
        predictions = self.get_predictions(feature_pos, scaled=scaled)
        absolute = self._calculate_absolute_error(targets, predictions)
        relative = self._calculate_relative_error(targets, predictions)
        title = self._generate_feature_name(feature_name, scaled) + ' Target vs Prediction Errors'
        self._plot_errors_new_process(absolute, relative, 'Absolute Error', 'Relative Error', title)
    
    def _get_months(self, length=None):
        return self._get_data(length=length)['month_id'].values
    
    def get_predicted_targets(self, scaled=False):
        return self._get_data(scaled)[self.predicted_vars].values
    
    def _get_data(self, scaled=False, length=None):
        length = length if length else self._window_length
        return self._reader.get_data(self._end_window_pos, length, scaled)
    
    def _get_target_data_length(self, tail=False):
        return self._end_window_pos if tail else self._window_length
    
    def get_predictions(self, feature_pos=0, scaled=False):
        raise NotImplementedError("Child classes must implement this method")
    
    def _generate_feature_name(self, feature_name, scaled=None):
        name = ''
        if scaled is not None:
            name += 'Scaled ' if scaled else 'Real '
        name +=  self.format_name(feature_name)
        return name.strip()
    
    def generate_feature_name(self, feature_pos, scaled=None):
        return self._generate_feature_name(self.reader.get_predicted_var_name(feature_pos), scaled)
    
    def format_name(self, name):
        ns = name.split('_')
        fname = ''
        
        for n in ns:
            fname += n[0].capitalize() + n[1:] + ' '
            
        return fname  
    
    def plot_target_vs_predicted(self, feature_pos=0, scaled=False, tail=False):
        feature_name = self._get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled) 
        self._plot_target_vs_predicted_new_process(self._get_target(feature_name, scaled=scaled, length=self._get_target_data_length(tail)), 
                                                   self.get_predictions(feature_pos, scaled), 
                                                   formatted_feature_name, 'Real vs Predicted ' + formatted_feature_name) 
    
    def plot_errors(self, feature_pos=0, scaled=False):
        return self._calculate_and_plot_errors(feature_pos, scaled)
    
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
    
    
    