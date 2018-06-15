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
        self._predictions = predictions
        self._unscaled_predictions = self._get_real_sales(predictions)
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
        
    def _get_real_sales(self, predictions):
        unscaled = np.reshape(self._reader.unscale_sales(predictions), [-1])
        return [round(max(prediction,0)) for prediction in unscaled]
        
    def plot_real_target_vs_predicted(self):
        self._plot_target_vs_predicted(self._get_data()['sales'].values, self._unscaled_predictions, 'Sales', 'Real vs Predicted Sales')
        
    def plot_scaled_target_vs_predicted(self):
        self._plot_target_vs_predicted(self._get_data(scaled=True)['sales'].values, self._predictions, 'Sales', 'Scaled Real vs Predicted Sales')
    
    def _get_months(self):
        return self._get_data()['month_id'].values
    
    def _get_data(self, scaled = False):
        return self._reader.get_data(self._end_window_pos, self._window_length, scaled)
    
        
        