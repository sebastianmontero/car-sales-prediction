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
    def end_window_pos(self):
        return self._end_window_pos
    
    @property
    def window_length(self):
        return self._window_length
    
    @property
    def reader(self):
        return self._reader
    
    def _unscale_sales(self, sales):
        return self._reader.unscale_sales(sales)
    
    def _run_in_new_process(self, target, args=()):
        p = Process(target=target, args=args)
        p.start()
    
    def _plot_target_vs_predicted_new_process(self, real, predictions, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_predicted, args=(real, predictions, ylabel, title))
    
    
    def _plot_target_vs(self, real, vs, ylabel, title):
        vs['Real'] = real
        self._plot_by_month('Real', vs, ylabel, title)
        
    def _plot_by_month(self, ref_key, series, ylabel, title):
        months = self._get_months(len(series[ref_key]))
        num_months = len(months)
        for label, vals in series.items():
            plt.plot(range(num_months - len(vals), num_months), vals, label=(label))
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
        
    def _get_target_sales(self, scaled=False, length=None):
        return self._get_data(scaled, length)['sales'].values

    def _calculate_and_plot_errors(self, scaled=False):
        targets = self._get_target_sales(scaled)
        predictions = self.get_predictions(scaled)
        absolute = self._calculate_absolute_error(targets, predictions)
        relative = self._calculate_relative_error(targets, predictions)
        title = 'Target vs Prediction Errors' + (' (Scaled)' if scaled else '')
        self._plot_errors_new_process(absolute, relative, 'Absolute Error', 'Relative Error', title)
    
    def _get_months(self, length=None):
        return self._get_data(length=length)['month_id'].values
    
    def _get_data(self, scaled=False, length=None):
        length = length if length else self._window_length
        return self._reader.get_data(self._end_window_pos, length, scaled)
    
    def _get_target_data_length(self, tail=False):
        return self._end_window_pos if tail else self._window_length
    
    def get_predictions(self, scaled=False):
        raise NotImplementedError("Child classes must implement this method")
        
    def plot_real_target_vs_predicted(self, tail=False):
        self._plot_target_vs_predicted_new_process(self._get_target_sales(length=self._get_target_data_length(tail)), self.get_predictions(), 'Sales', 'Real vs Predicted Sales')
        
    def plot_scaled_target_vs_predicted(self, tail=False):
        self._plot_target_vs_predicted_new_process(self._get_target_sales(scaled=True, length=self._get_target_data_length(tail)), self.get_predictions(scaled=True), 'Sales', 'Scaled Real vs Predicted Sales')
    
    def plot_real_errors(self):
        return self._calculate_and_plot_errors()
    
    def plot_scaled_errors(self):
        return self._calculate_and_plot_errors(scaled=True)
    
    def real_absolute_mean_error(self):
        return self._calculate_absolute_mean_error(self._get_target_sales(), self.get_predictions())
    
    def window_real_absolute_mean_error(self):
        return (self.real_absolute_mean_error() + self.real_absolute_error_by_pos(-1)) / 2
    
    def scaled_absolute_mean_error(self):
        return self._calculate_absolute_mean_error(self._get_target_sales(scaled=True), self.get_predictions(scaled=True))
    
    def real_relative_mean_error(self):
        return self._calculate_relative_mean_error(self._get_target_sales(), self.get_predictions())
    
    def scaled_relative_mean_error(self):
        return self._calculate_relative_mean_error(self._get_target_sales(scaled=True), self.get_predictions(scaled=True))
    
    def _get_absolute_error_by_pos(self, pos, scaled=False):
        return self._calculate_absolute_error_by_pos(self._get_target_sales(scaled), self.get_predictions(scaled), pos)
    
    def real_absolute_error_by_pos(self, pos):
        return self._get_absolute_error_by_pos(pos)
    
    def scaled_absolute_error_by_pos(self, pos):
        return self._get_absolute_error_by_pos(pos, True)
    
    