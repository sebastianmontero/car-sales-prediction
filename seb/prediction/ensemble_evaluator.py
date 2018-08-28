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
            #print('Chosen: ', candidate_pos)
            candidate_pos = None
            #print('Current best relative mean error: {}, networks: {}'.format(rme, len(best)))
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
        self._noise_variance = self._calculate_noise_variance(self._get_predicted_targets(scaled=True),self._mean, self._model_variance)
        self._std = self._calculate_std(predictions)
        self._min = self._get_min(predictions)
        self._max = self._get_max(predictions)
        self._lower, self._upper = self._calculate_interval(self._mean, self._std)
        self._mean_u = self._unscale_features(self._mean)
        self._std_u = self._unscale_features(self._std, round_=False)
        self._min_u = self._unscale_features(self._min)
        self._max_u = self._unscale_features(self._max)
        self._lower_u = self._unscale_features(self._lower)
        self._upper_u = self._unscale_features(self._upper)
        
        
    @property
    def best_network(self):
        return self._best_network
    
    def _generate_predictions_array(self, evaluators):
        predictions = []
        for e in evaluators:
            predictions.append(np.reshape(e.predictions(scaled=True), [-1, 1, e.num_predicted_vars]))
        
        return np.concatenate(predictions, axis=1)
    
    def _calculate_mean(self, predictions):
        return np.mean(predictions, axis=1)
    
    def _calculate_model_variance(self, predictions):
        return np.var(predictions, axis=1, ddof=1)
    
    def _calculate_noise_variance(self, targets, mean, model_variance):
        return [np.maximum((t - m) ** 2 - v, np.zeros(t.shape)) for t, m, v in zip(targets, mean, model_variance)]
    
    def _calculate_std(self, predictions):
        return np.std(predictions, axis=1, ddof=1)
    
    def _calculate_interval(self, mean, std):
        range_ = std * t.ppf(self._quantile, self._num_networks - 1)
        return mean - range_, mean + range_ 
    
    def _get_min(self, predictions):
        return np.amin(predictions, axis=1)
    
    def _get_max(self, predictions):
        return np.amax(predictions, axis=1)
    
    def mean(self, scaled=False):
        return self._mean if scaled else self._mean_u
    
    def std(self, scaled=False):
        return self._std if scaled else self._std_u
    
    def min(self, scaled=False):
        return self._min if scaled else self._min_u
    
    def max(self, scaled=False):
        return self._max if scaled else self._max_u
    
    def min_max_range(self, scaled=False):
        return self.get_max(scaled) - self.get_min(scaled)
    
    def lower(self, scaled=False):
        return self._lower if scaled else self._lower_u
    
    def upper(self, scaled=False):
        return self._upper if scaled else self._upper_u
    
    def get_predictions(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.mean(scaled), feature_pos)
    
    def get_std(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.std(scaled), feature_pos)
    
    def get_min(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.min(scaled), feature_pos)
    
    def get_max(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.max(scaled), feature_pos)
    
    def get_min_max_range(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.min_max_range(scaled), feature_pos)
    
    def get_lower(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.lower(scaled), feature_pos)
    
    def get_upper(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.upper(scaled), feature_pos)
    
    def get_noise_variance(self, feature_pos=0):
        return self._get_feature_values(self._noise_variance, feature_pos)
    
    def get_noise_variance_dataset(self):
        data = self._reader.get_data(self._end_window_pos, self._window_length, scaled=True).reset_index(drop=True)
        data = data.join(pd.DataFrame({'variance':self.get_noise_variance()}))
        return data;
    
    def _plot_target_vs_mean_best_new_process(self, real, mean, best, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_mean_best, args=(real, mean, best, ylabel, title))
        
    def _plot_target_vs_mean_best(self, real, mean, best, ylabel, title):
        self._plot_target_vs(real,{'Ensemble Mean':mean, 'Best Network': best},ylabel, title)
        
    def plot_target_vs_mean_best(self, feature_pos=0, scaled=False, tail=False):
        feature_name = self._get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_mean_best_new_process(self._get_target(feature_name, scaled=scaled,length=self._get_target_data_length(tail)), 
                                                   self.get_predictions(feature_pos, scaled), 
                                                   self._best_network.get_predictions(feature_pos, scaled), 
                                                   formatted_feature_name, 
                                                   'Target vs Ensemble Mean and Best Network ' + formatted_feature_name)
        
    
    def _plot_target_vs_mean_min_max_new_process(self, real, mean, min_, max_, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_mean_min_max, args=(real, mean, min_, max_, ylabel, title))
        
    def _plot_target_vs_mean_min_max(self, real, mean, min_, max_, ylabel, title):
        self._plot_target_vs(real,{'Ensemble Mean':mean, 'Min': min_, 'Max': max_},ylabel, title)
        
    def plot_target_vs_mean_min_max(self, feature_pos=0, scaled=False, tail=False):
        feature_name = self._get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_mean_min_max_new_process(self._get_target(feature_name, scaled=scaled, length=self._get_target_data_length(tail)), 
                                                      self.get_predictions(feature_pos, scaled), 
                                                      self.get_min(feature_pos, scaled), 
                                                      self.get_max(feature_pos, scaled) , 
                                                      formatted_feature_name, 
                                                      'Target vs Ensemble Mean, Min and Max ' + formatted_feature_name)
        
    def _plot_target_vs_mean_interval_new_process(self, real, mean, lower, upper, ylabel, title):
        self._run_in_new_process(target=self._plot_target_vs_mean_interval, args=(real, mean, lower, upper, ylabel, title))
        
    def _plot_target_vs_mean_interval(self, real, mean, lower, upper, ylabel, title):
        self._plot_target_vs(real,{'Ensemble Mean':mean, 'Lower Limit': lower, 'Upper Limit': upper}, ylabel, title)
        
    def plot_target_vs_mean_interval(self, feature_pos=0, scaled=False, tail=False):
        feature_name = self._get_predicted_var_name(feature_pos)
        formatted_feature_name = self._generate_feature_name(feature_name, scaled)
        self._plot_target_vs_mean_interval_new_process(self._get_target(feature_name, scaled=scaled,length=self._get_target_data_length(tail)), 
                                                       self.get_predictions(feature_pos, scaled), 
                                                       self.get_lower(feature_pos, scaled), 
                                                       self.get_upper(feature_pos, scaled) , 
                                                       formatted_feature_name, 
                                                       'Target vs Ensemble Mean and Interval ' + formatted_feature_name)
        
    
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
    
    def _plot_min_max_range_new_process(self, mm_range, ylabel, title):
        self._run_in_new_process(target=self._plot_min_max_range, args=(mm_range, ylabel, title))
        
    def _plot_min_max_range(self, mm_range, ylabel, title):
        self._plot_by_month('Min Max Range',{'Min Max Range':{'values': mm_range, 'type': 'bar'}}, ylabel, title, yfix=True)
        
    def plot_scaled_min_max_range(self):
        self._plot_min_max_range_new_process(self.get_min_max_range(scaled=True),'Min Max Range', 'Scaled Min Max Range')
    
    def plot_real_min_max_range(self):
        self._plot_min_max_range_new_process(self.get_min_max_range(scaled=False),'Min Max Range', 'Real Min Max Range')
    