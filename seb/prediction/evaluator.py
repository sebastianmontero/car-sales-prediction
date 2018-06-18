'''
Created on Jun 15, 2018

@author: nishilab
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Evaluator(object):

    def __init__(self, reader, predictions, end_window_pos):
        self._reader = reader
        self._unscaled_predictions = self._get_real_sales_from_predictions(predictions)
        self._predictions = np.reshape(predictions, [-1])
        self._end_window_pos = end_window_pos
        self._window_length = len(predictions)
        sns.set()
        
    def _plot_target_vs_predicted(self, real, predictions, ylabel, title):
        months = self._get_months()
        plt.plot(real, label=('Real'))
        plt.plot(predictions, label=('Predicted'))
        plt.ylabel(ylabel)
        plt.xticks(range(len(months)), months, rotation='vertical')
        plt.title(title)
        plt.legend()
        plt.show()
        
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
        
    def _calculate_absolute_error(self, target, predictions):
        return np.abs(target - predictions)
    
    def _calculate_relative_error(self, target, predictions):
        return np.abs((target - predictions)/target)
        
    def _get_real_sales_from_predictions(self, predictions):
        unscaled = np.reshape(self._reader.unscale_sales(predictions), [-1])
        return [round(max(prediction,0)) for prediction in unscaled]
        
    def _get_target_sales(self, scaled=False):
        return self._get_data(scaled)['sales'].values

    def _calculate_and_plot_errors(self, scaled=False):
        targets = self._get_target_sales(scaled)
        predictions = self._get_predictions(scaled)
        absolute = self._calculate_absolute_error(targets, predictions)
        relative = self._calculate_relative_error(targets, predictions)
        title = 'Target vs Prediction Errors' + (' (Scaled)' if scaled else '')
        self._plot_errors(absolute, relative, 'Absolute Error', 'Relative Error', title)
    
    def _get_months(self):
        return self._get_data()['month_id'].values
    
    def _get_data(self, scaled=False):
        return self._reader.get_data(self._end_window_pos, self._window_length, scaled)
    
    def _get_predictions(self, scaled=False):
        return self._predictions if scaled else self._unscaled_predictions
    
    def plot_real_target_vs_predicted(self):
        self._plot_target_vs_predicted(self._get_target_sales(), self._unscaled_predictions, 'Sales', 'Real vs Predicted Sales')
        
    def plot_scaled_target_vs_predicted(self):
        self._plot_target_vs_predicted(self._get_target_sales(scaled=True), self._predictions, 'Sales', 'Scaled Real vs Predicted Sales')
    
    def plot_real_errors(self):
        return self._calculate_and_plot_errors()
    
    def plot_scaled_errors(self):
        return self._calculate_and_plot_errors(scaled=True)
     
        
        