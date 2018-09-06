'''
Created on Jun 15, 2018

@author: nishilab
'''

import numpy as np
import pandas as pd

from utils import Utils
from base_evaluator import BaseEvaluator
from scipy.stats import t
from sklearn.cluster import MeanShift

class EnsembleEvaluator(BaseEvaluator):

    def __init__(self, evaluators, operator='mean', find_best_ensemble=False):
        BaseEvaluator.__init__(self)
        self._operator = operator
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
        self._mode = None
        self._mode_u = None
        self._median = None
        self._median_u = None
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
        
    @property
    def window_length(self):
        return self._window_length
        
    def _find_best_ensemble(self, evaluators):
        best = []
        candidate_pos = 0
        rme = None
        while len(evaluators) > 0 and candidate_pos is not None:
            best.append(evaluators[candidate_pos])
            del evaluators[candidate_pos]
            #print('Chosen: ', candidate_pos)
            candidate_pos = None
            #if rme:
            #    print('Current best relative mean error: {}, networks: {}'.format(rme, len(best)))
            for pos, evaluator in enumerate(evaluators):
                ensemble = EnsembleEvaluator([evaluator] + best, operator=self._operator)
                erme = ensemble.relative_mean_error()
                if rme is None or erme < rme:
                    candidate_pos = pos
                    rme = erme
                    
        return best
        
    def _process_evaluators(self, evaluators):
        predictions = self._generate_predictions_array(evaluators)
        self._mean = self._calculate_mean(predictions)
        self._median = self._calculate_median(predictions)
        
        self._model_variance = self._calculate_model_variance(predictions)
        self._noise_variance = self._calculate_noise_variance(self.get_predicted_targets(scaled=True),self._mean, self._model_variance)
        self._std = self._calculate_std(predictions)
        self._min = self._get_min(predictions)
        self._max = self._get_max(predictions)
        self._lower, self._upper = self._calculate_interval(self._mean, self._std)
        self._mean_u = self._unscale_features(self._mean)
        self._median_u = self._unscale_features(self._median)
        self._std_u = self._unscale_features(self._std, round_=False)
        self._min_u = self._unscale_features(self._min)
        self._max_u = self._unscale_features(self._max)
        self._lower_u = self._unscale_features(self._lower)
        self._upper_u = self._unscale_features(self._upper)
        
        if self._operator == 'mode':
            self._mode = self._calculate_mode(predictions)
            self._mode_u = self._unscale_features(self._mode)
        
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
    
    def _calculate_median(self, predictions):
        return np.median(predictions, axis=1)
    
    def _calculate_mode(self, predictions):
        mode = []
        prev_month_predictions = None
        for month_predictions in predictions:
            ms = MeanShift()
            try:
                ms.fit(month_predictions)
            except:
                ms.set_params(bandwidth=0.75)
                ms.fit(month_predictions)
            if prev_month_predictions is not None:
                cc = np.array(ms.cluster_centers_)
                distances = np.linalg.norm(cc - prev_month_predictions, axis=1)
                sel_predictions = cc[np.argmin(distances)]
            else:
                sel_predictions = np.array(ms.cluster_centers_[0])
            prev_month_predictions = sel_predictions
            mode.append(sel_predictions)
        
        return mode
    
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
    
    def median(self, scaled=False):
        return self._median if scaled else self._median_u
    
    def mode(self, scaled=False):
        return self._mode if scaled else self._mode_u
    
    def std(self, scaled=False):
        return self._std if scaled else self._std_u
    
    def min(self, scaled=False):
        return self._min if scaled else self._min_u
    
    def max(self, scaled=False):
        return self._max if scaled else self._max_u
    
    def min_max_range(self, scaled=False):
        return self.max(scaled) - self.min(scaled)
    
    def lower(self, scaled=False):
        return self._lower if scaled else self._lower_u
    
    def upper(self, scaled=False):
        return self._upper if scaled else self._upper_u
    
    #Has to be defined so that its compatible with evaluator, this method is used in BaseEvaluators
    def predictions(self, scaled=False):
        fn = {
                'mean': self.mean,
                'median' : self.median,
                'mode' : self.mode
              }
        return fn[self._operator](scaled)
    
    def get_predictions(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.predictions(scaled), feature_pos)
    
    def get_std(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.std(scaled), feature_pos)
    
    def get_min(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.min(scaled), feature_pos)
    
    def get_max(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.max(scaled), feature_pos)
    
    def get_min_max_range(self, feature_pos=0, scaled=False):
        return self.get_max(feature_pos, scaled) - self.get_min(feature_pos, scaled)
    
    def get_lower(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.lower(scaled), feature_pos)
    
    def get_upper(self, feature_pos=0, scaled=False):
        return self._get_feature_values(self.upper(scaled), feature_pos)
    
    def get_model_variance(self, feature_pos=0):
        return self._get_feature_values(self._model_variance, feature_pos)
    
    def get_noise_variance(self, feature_pos=0):
        return self._get_feature_values(self._noise_variance, feature_pos)
    
    def get_noise_variance_dataset(self):
        data = self._reader.get_data(self._end_window_pos, self._window_length, scaled=True).reset_index(drop=True)
        data = data.join(pd.DataFrame({'variance':self.get_noise_variance()}))
        return data;
    