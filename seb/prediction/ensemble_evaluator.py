'''
Created on Jun 15, 2018

@author: nishilab
'''

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import Utils
from base_evaluator import BaseEvaluator
from scipy.stats import t

class EnsembleEvaluator(BaseEvaluator):

    def __init__(self, evaluators):
        BaseEvaluator.__init__(self)
        assert (len(evaluators) > 0), "There must be at least one evaluator in the ensemble"
        best_network = evaluators[0]
        self._quantile = 0.975 #0.95 confidence interval 
        self._best_network = best_network
        self._end_window_pos = best_network.end_window_pos
        self._window_length = best_network.window_length
        self._num_networks = len(evaluators)
        self._reader = best_network.reader
        self._mean = None
        self._mean_u = None
        self._variance = None
        self._std = None
        self._std_u = None
        self._min = None
        self._max = None
        self._min_u = None
        self._max_u = None
        self._lower = None
        self._lower_u = None
        self._upper = None
        self._upper_u = None
        
        self._process_evaluators(evaluators)
        
    def _process_evaluators(self, evaluators):
        predictions = self._generate_predictions_array(evaluators)
        self._mean = self._calculate_mean(predictions)
        self._variance = self._calculate_variance(predictions)
        self._std = self._calculate_std(predictions)
        self._min = self._get_min(predictions)
        self._max = self._get_max(predictions)
        self._lower, self._upper = self._calculate_interval(self._mean, self._std)
        self._mean_u = self._unscale_sales(self._mean)
        self._std_u = self._unscale_sales(self._std)
        self._min_u = self._unscale_sales(self._min)
        self._max_u = self._unscale_sales(self._max)
        self._lower_u = self._unscale_sales(self._lower)
        self._upper_u = self._unscale_sales(self._upper)
    
    def _generate_predictions_array(self, evaluators):
        predictions = []
        for e in evaluators:
            predictions.append(e.predictions)
        
        return np.array(predictions)
    
    def _calculate_mean(self, predictions):
        return np.mean(predictions, axis=0)
    
    def _calculate_variance(self, predictions):
        return np.var(predictions, axis=0, ddof=1)
    
    def _calculate_std(self, predictions):
        return np.std(predictions, axis=0, ddof=1)
    
    def _calculate_interval(self, mean, std):
        range_ = np.array(std) * t.ppf(self._quantile, self._num_networks - 1)
        return mean - range_, mean + range_ 
    
    def _get_min(self, predictions):
        return np.amin(predictions, axis=0)
    
    def _get_max(self, predictions):
        return np.amax(predictions, axis=0)
        
    def get_predictions(self, scaled=False):
        return self._mean if scaled else self._mean_u
    
    def get_std(self, scaled=False):
        return self._std if scaled else self._std_u
    
    def get_min(self, scaled=False):
        return self._min if scaled else self._min_u
    
    def get_max(self, scaled=False):
        return self._max if scaled else self._max_u
    
    def get_lower(self, scaled=False):
        return self._lower if scaled else self._lower_u
    
    def get_upper(self, scaled=False):
        return self._upper if scaled else self._upper_u
    
    def _plot_target_vs_mean_best_new_process(self, real, mean, best, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_mean_best, args=(real, mean, best, ylabel, title))
        
    def _plot_target_vs_mean_best(self, real, mean, best, ylabel, title):
        self._plot_target_vs(real,{'Ensemble Mean':mean, 'Best Network': best},ylabel, title)
        
    def plot_real_target_vs_mean_best(self, tail=False):
        self._plot_target_vs_mean_best_new_process(self._get_target_sales(length=self._get_target_data_length(tail)), self.get_predictions(), self._best_network.get_predictions(), 'Sales', 'Real vs Ensemble Mean and Best Network Sales')
        
    def plot_scaled_target_vs_mean_best(self, tail=False):
        self._plot_target_vs_mean_best_new_process(self._get_target_sales(scaled=True, length=self._get_target_data_length(tail)), self.get_predictions(scaled=True), self._best_network.get_predictions(scaled=True), 'Scaled Sales', 'Scaled Real vs Ensemble Mean and Best Network Sales')
    
    
    def _plot_target_vs_mean_min_max_new_process(self, real, mean, min_, max_, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_mean_min_max, args=(real, mean, min_, max_, ylabel, title))
        
    def _plot_target_vs_mean_min_max(self, real, mean, min_, max_, ylabel, title):
        self._plot_target_vs(real,{'Ensemble Mean':mean, 'Min': min_, 'Max': max_},ylabel, title)
        
    def plot_real_target_vs_mean_min_max(self, tail=False):
        self._plot_target_vs_mean_min_max_new_process(self._get_target_sales(length=self._get_target_data_length(tail)), self.get_predictions(), self.get_min(), self.get_max() , 'Sales', 'Real vs Ensemble Mean, Min and Max Sales')
        
    def plot_scaled_target_vs_mean_min_max(self, tail=False):
        self._plot_target_vs_mean_min_max_new_process(self._get_target_sales(scaled=True, length=self._get_target_data_length(tail)), self.get_predictions(scaled=True), self.get_min(scaled=True), self.get_max(scaled=True), 'Scaled Sales', 'Scaled Real vs Ensemble Mean, Min and Max Sales')
        
    def _plot_target_vs_mean_interval_new_process(self, real, mean, lower, upper, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_mean_interval, args=(real, mean, lower, upper, ylabel, title))
        
    def _plot_target_vs_mean_interval(self, real, mean, lower, upper, ylabel, title):
        self._plot_target_vs(real,{'Ensemble Mean':mean, 'Lower Limit': lower, 'Upper Limit': upper}, ylabel, title)
        
    def plot_real_target_vs_mean_interval(self, tail=False):
        self._plot_target_vs_mean_interval_new_process(self._get_target_sales(length=self._get_target_data_length(tail)), self.get_predictions(), self.get_lower(), self.get_upper() , 'Sales', 'Real vs Ensemble Mean and Interval Sales')
        
    def plot_scaled_target_vs_mean_interval(self, tail=False):
        self._plot_target_vs_mean_interval_new_process(self._get_target_sales(scaled=True, length=self._get_target_data_length(tail)), self.get_predictions(scaled=True), self.get_lower(scaled=True), self.get_upper(scaled=False), 'Scaled Sales', 'Scaled Real vs Ensemble Mean and Interval Saless')    
    
    
    