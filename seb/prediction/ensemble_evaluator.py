'''
Created on Jun 15, 2018

@author: nishilab
'''

import numpy as np
import pandas as pd

from utils import Utils
from base_evaluator import BaseEvaluator
from scipy.stats import t

class EnsembleEvaluator(BaseEvaluator):

    def __init__(self, evaluators, find_best_ensemble=False):
        BaseEvaluator.__init__(self)
        if find_best_ensemble:
            evaluators = self._find_best_ensemble(evaluators)
        assert (len(evaluators) > 1), "There must be at least two evaluators in the ensemble"
        best_network = evaluators[0]
        self._quantile = 0.975 #0.95 confidence interval 
        self._best_network = best_network
        self._end_window_pos = best_network.end_window_pos
        self._window_length = best_network.window_length
        self._num_networks = len(evaluators)
        self._reader = best_network.reader
        self._mean = None
        self._mean_u = None
        self._model_variance = None
        self._noise_variance = None
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
        
    def _find_best_ensemble(self, evaluators):
        best = []
        candidate_pos = 0
        rme = evaluators[0].real_relative_mean_error()
        while len(evaluators) > 0 and candidate_pos is not None:
            best.append(evaluators[candidate_pos])
            del evaluators[candidate_pos]
            print('Chosen: ', candidate_pos)
            candidate_pos = None
            print('Current best relative mean error: {}, networks: {}'.format(rme, len(best)))
            for pos, evaluator in enumerate(evaluators):
                ensemble = EnsembleEvaluator([evaluator] + best)
                erme = ensemble.real_relative_mean_error()
                if erme < rme:
                    candidate_pos = pos
                    rme = erme
                    
        return best
        
    def _process_evaluators(self, evaluators):
        predictions = self._generate_predictions_array(evaluators)
        self._mean = self._calculate_mean(predictions)
        self._model_variance = self._calculate_model_variance(predictions)
        self._noise_variance = self._calculate_noise_variance(self._get_target_sales(scaled=True),self._mean, self._model_variance)
        self._std = self._calculate_std(predictions)
        self._min = self._get_min(predictions)
        self._max = self._get_max(predictions)
        self._lower, self._upper = self._calculate_interval(self._mean, self._std)
        self._mean_u = self._unscale_sales(self._mean)
        self._std_u = self._unscale_sales(self._std, round_=False)
        self._min_u = self._unscale_sales(self._min)
        self._max_u = self._unscale_sales(self._max)
        self._lower_u = self._unscale_sales(self._lower)
        self._upper_u = self._unscale_sales(self._upper)
        
    @property
    def best_network(self):
        return self._best_network
    
    def _generate_predictions_array(self, evaluators):
        predictions = []
        for e in evaluators:
            predictions.append(e.predictions)
        
        return np.array(predictions)
    
    def _calculate_mean(self, predictions):
        return np.mean(predictions, axis=0)
    
    def _calculate_model_variance(self, predictions):
        return np.var(predictions, axis=0, ddof=1)
    
    def _calculate_noise_variance(self, targets, mean, model_variance):
        return [max((t - m) ** 2 - v, 0) for t, m, v in zip(targets, mean, model_variance)]
    
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
    
    def get_noise_variance_dataset(self):
        data = self._reader.get_data(self._end_window_pos, self._window_length, scaled=True).reset_index(drop=True)
        data = data.join(pd.DataFrame({'variance':self._noise_variance}))
        return data;
    
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
        self._plot_target_vs_mean_interval_new_process(self._get_target_sales(scaled=True, length=self._get_target_data_length(tail)), self.get_predictions(scaled=True), self.get_lower(scaled=True), self.get_upper(scaled=True), 'Scaled Sales', 'Scaled Real vs Ensemble Mean and Interval Saless')    
    
    def _plot_variance_errors_new_process(self, model_variance, noise_variance, ylabel, title):
        self._run_in_new_process(target=self._plot_variance_errors, args=(model_variance, noise_variance, ylabel, title))
        
    def _plot_variance_errors(self, model_variance, noise_variance, ylabel, title):
        self._plot_by_month('Model Variance',{'Model Variance':model_variance, 'Noise Variance': noise_variance}, ylabel, title)
        
    def plot_variance_errors(self):
        self._plot_variance_errors_new_process(self._model_variance, self._noise_variance, 'variance', 'Model and Noise Variance')
    
    def _plot_std_new_process(self, std, ylabel, title):
        self._run_in_new_process(target=self._plot_std, args=(std, ylabel, title))
        
    def _plot_std(self, std, ylabel, title):
        self._plot_by_month('Standard Deviation',{'Standard Deviation':{'values': std, 'type': 'bar'}}, ylabel, title, yfix=True)
        
    def plot_scaled_std(self):
        self._plot_std_new_process(self.get_std(scaled=True),'Standard Deviation', 'Scaled Standard Deviation')
    
    def plot_real_std(self):
        self._plot_std_new_process(self.get_std(scaled=False),'Standard Deviation', 'Real Standard Deviation')
    
    